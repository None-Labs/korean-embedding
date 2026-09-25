package rag

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
)

// ChunkerConfig configures sentence-boundary-preserving Korean document chunking.
type ChunkerConfig struct {
	MaxRunesPerChunk int `json:"max_runes_per_chunk"` // Maximum character length per chunk (default: 350)
	MinRunesPerChunk int `json:"min_runes_per_chunk"` // Minimum character length to form a chunk (default: 50)
	OverlapSentences int `json:"overlap_sentences"`   // Number of overlapping sentences across chunk boundaries (default: 1)
}

// DefaultChunkerConfig returns recommended chunking thresholds for Korean RAG.
func DefaultChunkerConfig() ChunkerConfig {
	return ChunkerConfig{
		MaxRunesPerChunk: 350,
		MinRunesPerChunk: 40,
		OverlapSentences: 1,
	}
}

// KoreanSemanticChunker splits text into chunks strictly along Korean sentence boundaries.
type KoreanSemanticChunker struct {
	cfg       ChunkerConfig
	segmenter *morphology.Segmenter
}

// NewKoreanSemanticChunker creates an initialized semantic chunker.
func NewKoreanSemanticChunker(cfg ChunkerConfig) *KoreanSemanticChunker {
	if cfg.MaxRunesPerChunk <= 0 {
		cfg.MaxRunesPerChunk = 350
	}
	if cfg.MinRunesPerChunk <= 0 {
		cfg.MinRunesPerChunk = 40
	}
	if cfg.OverlapSentences < 0 {
		cfg.OverlapSentences = 1
	}

	return &KoreanSemanticChunker{
		cfg:       cfg,
		segmenter: morphology.DefaultSegmenter(),
	}
}

// ChunkDocument segments a Document into semantically intact chunks.
func (c *KoreanSemanticChunker) ChunkDocument(doc Document) []Chunk {
	paragraphs := c.segmenter.AnalyzeStructure(doc.Content)
	var allSentences []string

	for _, p := range paragraphs {
		for _, s := range p.Sentences {
			trimmed := strings.TrimSpace(s.Text)
			if trimmed != "" {
				allSentences = append(allSentences, trimmed)
			}
		}
	}

	if len(allSentences) == 0 {
		// Fallback: split by line
		lines := strings.Split(doc.Content, "\n")
		for _, l := range lines {
			trimmed := strings.TrimSpace(l)
			if trimmed != "" {
				allSentences = append(allSentences, trimmed)
			}
		}
	}

	if len(allSentences) == 0 {
		return nil
	}

	var chunks []Chunk
	chunkIdx := 1
	n := len(allSentences)

	for i := 0; i < n; {
		var currentSentences []string
		currentRuneCount := 0
		j := i

		for j < n {
			sent := allSentences[j]
			sentLen := utf8.RuneCountInString(sent)

			if len(currentSentences) > 0 && (currentRuneCount+sentLen > c.cfg.MaxRunesPerChunk) {
				break
			}

			currentSentences = append(currentSentences, sent)
			currentRuneCount += sentLen
			j++
		}

		chunkText := strings.Join(currentSentences, " ")
		chunkID := fmt.Sprintf("%s_chk_%03d", doc.ID, chunkIdx)

		chunks = append(chunks, Chunk{
			ID:         chunkID,
			DocID:      doc.ID,
			DocTitle:   doc.Title,
			ChunkIndex: chunkIdx,
			Text:       chunkText,
		})
		chunkIdx++

		// Advance index with sentence overlap
		advance := len(currentSentences) - c.cfg.OverlapSentences
		if advance <= 0 {
			advance = 1
		}
		i += advance
	}

	return chunks
}
