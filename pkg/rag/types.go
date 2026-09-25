package rag

import (
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// Document represents an un-chunked source knowledge document or conversation record.
type Document struct {
	ID        string                 `json:"id"`
	Title     string                 `json:"title"`
	Content   string                 `json:"content"`
	Category  string                 `json:"category"` // e.g., "사내업무규정", "고객센터FAQ", "상담대화록"
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
}

// Chunk represents a semantically coherent, morphologically bounded sub-document segment.
type Chunk struct {
	ID             string                              `json:"id"`
	DocID          string                              `json:"doc_id"`
	DocTitle       string                              `json:"doc_title"`
	ChunkIndex     int                                 `json:"chunk_index"`
	Text           string                              `json:"text"`
	NormalizedText string                              `json:"normalized_text"`
	Embedding      []float32                           `json:"embedding"` // 3024-d Unit L2-norm
	Metadata       embedder.UtteranceMetadata          `json:"metadata"`
	Subspaces      map[embedder.SubspaceType][]float32 `json:"subspaces,omitempty"`
}

// SearchOptions defines retrieval filtering, MRL multi-stage limits, and hybrid subspace weighting.
type SearchOptions struct {
	QueryText      string                `json:"query_text,omitempty"` // Raw query string for sparse lexical BM25 matching
	TopK           int                   `json:"top_k"`                // Final number of chunks to return (default: 3)
	CoarseTopK     int                   `json:"coarse_top_k"`         // Stage 1 MRL candidate pool size (default: 20)
	MRLDim         int                   `json:"mrl_dim"`              // Stage 1 MRL dimension (512 or 1024, default: 512)
	FilterStance   embedder.SocialStance `json:"filter_stance"`        // Optional stance filter (Formal vs Informal)
	FilterCategory string                `json:"filter_category"`      // Optional document category filter
	MinScore       float32               `json:"min_score"`            // Minimum cosine similarity threshold
	LiteralWeight  float32               `json:"literal_weight"`       // Weight for 1536-d Literal subspace (default: 0.60)
	IntentWeight   float32               `json:"intent_weight"`        // Weight for 512-d Intent subspace (default: 0.25)
	SocialWeight   float32               `json:"social_weight"`        // Weight for 512-d Social subspace (default: 0.15)
	DenseWeight    float32               `json:"dense_weight"`         // Hybrid weight for 3024-d dense embedding (default: 0.60)
	BM25Weight     float32               `json:"bm25_weight"`          // Hybrid weight for Korean morpheme BM25 (default: 0.40)
}

// DefaultSearchOptions returns balanced settings for Korean knowledge retrieval.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		TopK:          3,
		CoarseTopK:    20,
		MRLDim:        512,
		MinScore:      0.20,
		LiteralWeight: 0.60,
		IntentWeight:  0.25,
		SocialWeight:  0.15,
		DenseWeight:   0.60,
		BM25Weight:    0.40,
	}
}

// SearchResult holds a retrieved chunk with its total score and subspace breakdown.
type SearchResult struct {
	Rank         int     `json:"rank"`
	Chunk        Chunk   `json:"chunk"`
	Score        float32 `json:"score"`         // Hybrid composite score
	DenseScore   float32 `json:"dense_score"`   // 3024-d dense composite score
	BM25Score    float32 `json:"bm25_score"`    // Korean morpheme BM25 score
	FullCosSim   float32 `json:"full_cos_sim"`  // Raw 3024-d cosine similarity
	LiteralScore float32 `json:"literal_score"` // 1536-d Literal similarity
	IntentScore  float32 `json:"intent_score"`  // 512-d Intent similarity
	SocialScore  float32 `json:"social_score"`  // 512-d Social similarity
	NuanceScore  float32 `json:"nuance_score"`  // 464-d Nuance similarity
}

// RAGResponse encapsulates end-to-end RAG query execution details.
type RAGResponse struct {
	Query           string                     `json:"query"`
	QueryMetadata   embedder.UtteranceMetadata `json:"query_metadata"`
	RetrievedChunks []SearchResult             `json:"retrieved_chunks"`
	ContextText     string                     `json:"context_text"`
	AugmentedPrompt string                     `json:"augmented_prompt"`
	GeneratedAnswer string                     `json:"generated_answer,omitempty"`
	RetrievalTime   time.Duration              `json:"retrieval_time"`
	TotalLatency    time.Duration              `json:"total_latency"`
}
