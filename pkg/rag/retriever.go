package rag

import (
	"fmt"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

// SubspaceRetriever manages query embedding, intent-driven weight adaptation, and vector retrieval.
type SubspaceRetriever struct {
	emb   *embedder.Embedder
	store VectorStore
}

// NewSubspaceRetriever creates an initialized retriever.
func NewSubspaceRetriever(emb *embedder.Embedder, store VectorStore) *SubspaceRetriever {
	return &SubspaceRetriever{
		emb:   emb,
		store: store,
	}
}

// Retrieve embeds the query, dynamically adapts subspace weights according to the query's speech act,
// and retrieves the top matching knowledge chunks.
func (r *SubspaceRetriever) Retrieve(query string, opts SearchOptions) ([]SearchResult, *embedder.UtteranceMetadata, error) {
	embRes, err := r.emb.Embed(query)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Dynamic intent-driven weight adaptation
	adaptedOpts := opts
	adaptedOpts.QueryText = query
	if adaptedOpts.LiteralWeight == 0 && adaptedOpts.IntentWeight == 0 && adaptedOpts.SocialWeight == 0 {
		adaptedOpts = DefaultSearchOptions()
		adaptedOpts.QueryText = query
	}

	// Adapt weights based on speech act of the user query
	switch embRes.Metadata.SpeechAct {
	case transformer.SpeechActQuestion:
		// Questions demand high factual/literal fidelity
		adaptedOpts.LiteralWeight = 0.70
		adaptedOpts.IntentWeight = 0.20
		adaptedOpts.SocialWeight = 0.10
	case transformer.SpeechActRequest:
		// Requests demand matching actionable procedural guidance
		adaptedOpts.LiteralWeight = 0.50
		adaptedOpts.IntentWeight = 0.35
		adaptedOpts.SocialWeight = 0.15
	case transformer.SpeechActApology, transformer.SpeechActAcceptance:
		// Relationship/affective queries demand higher social and stance alignment
		adaptedOpts.LiteralWeight = 0.40
		adaptedOpts.IntentWeight = 0.30
		adaptedOpts.SocialWeight = 0.30
	}

	results, err := r.store.Search(embRes.Vector, embRes.Subspaces, adaptedOpts)
	if err != nil {
		return nil, nil, fmt.Errorf("retrieval failed: %w", err)
	}

	return results, &embRes.Metadata, nil
}
