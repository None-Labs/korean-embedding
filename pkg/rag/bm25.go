package rag

import (
	"math"
	"strings"
	"sync"

	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
)

// BM25Index implements Korean morphological stem-based BM25 scoring.
type BM25Index struct {
	mu        sync.RWMutex
	segmenter *morphology.Segmenter
	docTokens map[string][]string // chunkID -> morpheme stems
	docLens   map[string]int      // chunkID -> token length
	df        map[string]int      // stem -> document frequency
	totalDocs int
	avgDocLen float64

	k1 float64 // default: 1.5
	b  float64 // default: 0.75
}

// NewBM25Index creates an initialized BM25 index.
func NewBM25Index() *BM25Index {
	return &BM25Index{
		segmenter: morphology.DefaultSegmenter(),
		docTokens: make(map[string][]string),
		docLens:   make(map[string]int),
		df:        make(map[string]int),
		k1:        1.5,
		b:         0.75,
	}
}

// extractMorphemeStems analyzes text and extracts substantive/predicate roots and numbers.
func (idx *BM25Index) extractMorphemeStems(text string) []string {
	morphs := idx.segmenter.Analyze(text)
	var stems []string

	for _, m := range morphs {
		tag := m.Tag
		// Keep substantives (nouns), predicates (verbs/adjectives), numbers, and roots
		if tag.IsSubstantive() || tag.IsPredicate() || tag == morphology.TagSN ||
			tag == morphology.TagXR || tag == morphology.TagSL {
			s := strings.TrimSpace(m.Surface)
			if s != "" {
				stems = append(stems, s)
			}
		}
	}
	return stems
}

// AddDocument indexes a chunk's text for BM25 search.
func (idx *BM25Index) AddDocument(id, text string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	stems := idx.extractMorphemeStems(text)
	idx.docTokens[id] = stems
	idx.docLens[id] = len(stems)

	seen := make(map[string]bool)
	for _, st := range stems {
		if !seen[st] {
			idx.df[st]++
			seen[st] = true
		}
	}

	idx.totalDocs = len(idx.docTokens)
	var sumLen int
	for _, l := range idx.docLens {
		sumLen += l
	}
	if idx.totalDocs > 0 {
		idx.avgDocLen = float64(sumLen) / float64(idx.totalDocs)
	}
}

// Score computes the BM25 score of a document given query text.
func (idx *BM25Index) Score(docID string, queryText string) float32 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	docStems, ok := idx.docTokens[docID]
	if !ok || len(docStems) == 0 || idx.totalDocs == 0 {
		return 0.0
	}

	queryStems := idx.extractMorphemeStems(queryText)
	if len(queryStems) == 0 {
		return 0.0
	}

	// Calculate term frequencies in the document
	tf := make(map[string]int)
	for _, s := range docStems {
		tf[s]++
	}

	docLen := float64(idx.docLens[docID])
	var totalScore float64

	for _, qStem := range queryStems {
		freq := float64(tf[qStem])
		if freq == 0 {
			continue
		}

		docFreq := float64(idx.df[qStem])
		// Robertson-Spärck Jones IDF
		idf := math.Log(1.0 + (float64(idx.totalDocs)-docFreq+0.5)/(docFreq+0.5))
		if idf < 0 {
			idf = 0.01
		}

		num := freq * (idx.k1 + 1.0)
		denom := freq + idx.k1*(1.0-idx.b+idx.b*(docLen/idx.avgDocLen))
		totalScore += idf * (num / denom)
	}

	return float32(totalScore)
}
