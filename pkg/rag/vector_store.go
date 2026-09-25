package rag

import (
	"encoding/gob"
	"fmt"
	"os"
	"sort"
	"sync"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// VectorStore defines the interface for RAG vector storage and multi-stage retrieval.
type VectorStore interface {
	AddChunks(chunks []Chunk) error
	Count() int
	Search(queryVec []float32, querySubspaces map[embedder.SubspaceType][]float32, opts SearchOptions) ([]SearchResult, error)
	Save(path string) error
	Load(path string) error
}

// InMemoryVectorStore provides a thread-safe, pure Go in-memory vector store with MRL 2-stage search and BM25 hybrid ranking.
type InMemoryVectorStore struct {
	mu     sync.RWMutex
	chunks []Chunk
	slicer *embedder.MRLSlicer
	bm25   *BM25Index
}

// NewInMemoryVectorStore creates an empty in-memory vector store.
func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{
		chunks: make([]Chunk, 0),
		slicer: embedder.NewMRLSlicer(),
		bm25:   NewBM25Index(),
	}
}

// AddChunks appends chunks to the store and updates the Korean BM25 lexical index.
func (vs *InMemoryVectorStore) AddChunks(chunks []Chunk) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	for _, chk := range chunks {
		if len(chk.Embedding) != embedder.TotalDim {
			return fmt.Errorf("chunk %s embedding dimension must be %d, got %d",
				chk.ID, embedder.TotalDim, len(chk.Embedding))
		}
		vs.chunks = append(vs.chunks, chk)
		vs.bm25.AddDocument(chk.ID, chk.Text)
	}
	return nil
}

// Count returns the total number of indexed chunks.
func (vs *InMemoryVectorStore) Count() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.chunks)
}

type coarseCandidate struct {
	index int
	sim   float32
}

// Search executes two-stage hybrid retrieval:
// Stage 1: Fast MRL 512-d coarse candidate selection.
// Stage 2: Dense Subspace (Literal + Intent + Social) and Korean Morpheme BM25 fine re-ranking.
func (vs *InMemoryVectorStore) Search(
	queryVec []float32,
	querySubspaces map[embedder.SubspaceType][]float32,
	opts SearchOptions,
) ([]SearchResult, error) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	n := len(vs.chunks)
	if n == 0 {
		return nil, nil
	}

	if opts.TopK <= 0 {
		opts.TopK = 3
	}
	if opts.CoarseTopK <= 0 {
		opts.CoarseTopK = 20
	}
	if opts.MRLDim <= 0 {
		opts.MRLDim = 512
	}
	if opts.LiteralWeight <= 0 && opts.IntentWeight <= 0 && opts.SocialWeight <= 0 {
		opts.LiteralWeight = 0.60
		opts.IntentWeight = 0.25
		opts.SocialWeight = 0.15
	}
	if opts.DenseWeight <= 0 && opts.BM25Weight <= 0 {
		opts.DenseWeight = 0.60
		opts.BM25Weight = 0.40
	}

	// 1. Stage 1: Coarse candidate filtering using MRL slicing (e.g. 512-d)
	queryMRL, err := vs.slicer.Slice(queryVec, opts.MRLDim)
	if err != nil {
		return nil, fmt.Errorf("failed to slice query vector: %w", err)
	}

	candidates := make([]coarseCandidate, 0, n)
	for i, chk := range vs.chunks {
		// Metadata filtering
		if opts.FilterStance != "" && chk.Metadata.SocialStance != opts.FilterStance {
			continue
		}

		chkMRL, err := vs.slicer.Slice(chk.Embedding, opts.MRLDim)
		if err != nil {
			continue
		}

		sim := embedder.CosineSimilarity(queryMRL, chkMRL)
		// If queryText is provided, also boost candidate if BM25 has direct keyword match
		if opts.QueryText != "" {
			bm25Raw := vs.bm25.Score(chk.ID, opts.QueryText)
			sim += bm25Raw * 0.1
		}

		candidates = append(candidates, coarseCandidate{index: i, sim: sim})
	}

	// Sort candidates by coarse similarity descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].sim > candidates[j].sim
	})

	coarseLimit := opts.CoarseTopK
	if coarseLimit > len(candidates) {
		coarseLimit = len(candidates)
	}
	topCandidates := candidates[:coarseLimit]

	// 2. Stage 2: Fine Subspace and BM25 Hybrid Re-ranking
	qLit := querySubspaces[embedder.SubspaceLiteral]
	qInt := querySubspaces[embedder.SubspaceIntent]
	qSoc := querySubspaces[embedder.SubspaceSocial]
	qNua := querySubspaces[embedder.SubspaceNuance]

	// Compute raw BM25 scores to find max for min-max scaling
	rawBM25 := make([]float32, coarseLimit)
	var maxBM25 float32
	if opts.QueryText != "" {
		for i, cand := range topCandidates {
			s := vs.bm25.Score(vs.chunks[cand.index].ID, opts.QueryText)
			rawBM25[i] = s
			if s > maxBM25 {
				maxBM25 = s
			}
		}
	}

	results := make([]SearchResult, 0, coarseLimit)

	for i, cand := range topCandidates {
		chk := vs.chunks[cand.index]

		fullSim := embedder.CosineSimilarity(queryVec, chk.Embedding)
		litSim := embedder.CosineSimilarity(qLit, chk.Subspaces[embedder.SubspaceLiteral])
		intSim := embedder.CosineSimilarity(qInt, chk.Subspaces[embedder.SubspaceIntent])
		socSim := embedder.CosineSimilarity(qSoc, chk.Subspaces[embedder.SubspaceSocial])
		nuaSim := embedder.CosineSimilarity(qNua, chk.Subspaces[embedder.SubspaceNuance])

		// Subspace-weighted dense score
		denseScore := (opts.LiteralWeight * litSim) +
			(opts.IntentWeight * intSim) +
			(opts.SocialWeight * socSim)

		// Normalized BM25 score
		var normBM25 float32
		if maxBM25 > 0 {
			normBM25 = rawBM25[i] / maxBM25
		}

		// Hybrid composite score: alpha * dense + (1 - alpha) * BM25
		var compositeScore float32
		if opts.QueryText != "" && maxBM25 > 0 {
			compositeScore = (opts.DenseWeight * denseScore) + (opts.BM25Weight * normBM25)
		} else {
			compositeScore = denseScore
		}

		if opts.MinScore > 0 && compositeScore < opts.MinScore {
			continue
		}

		results = append(results, SearchResult{
			Chunk:        chk,
			Score:        compositeScore,
			DenseScore:   denseScore,
			BM25Score:    rawBM25[i],
			FullCosSim:   fullSim,
			LiteralScore: litSim,
			IntentScore:  intSim,
			SocialScore:  socSim,
			NuanceScore:  nuaSim,
		})
	}

	// Sort results by composite score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	finalLimit := opts.TopK
	if finalLimit > len(results) {
		finalLimit = len(results)
	}

	finalResults := results[:finalLimit]
	for idx := range finalResults {
		finalResults[idx].Rank = idx + 1
	}

	return finalResults, nil
}

// Save serializes the vector store to a binary file using encoding/gob.
func (vs *InMemoryVectorStore) Save(path string) error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create vector store file %s: %w", path, err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(vs.chunks); err != nil {
		return fmt.Errorf("failed to encode chunks: %w", err)
	}
	return nil
}

// Load restores the vector store from a binary file and re-indexes the BM25 lexical structures.
func (vs *InMemoryVectorStore) Load(path string) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open vector store file %s: %w", path, err)
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var chunks []Chunk
	if err := dec.Decode(&chunks); err != nil {
		return fmt.Errorf("failed to decode chunks: %w", err)
	}

	vs.chunks = chunks
	vs.bm25 = NewBM25Index()
	for _, chk := range chunks {
		vs.bm25.AddDocument(chk.ID, chk.Text)
	}
	return nil
}
