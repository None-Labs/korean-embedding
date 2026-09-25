package rag

import (
	"fmt"
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// PipelineConfig configures the RAG system components.
type PipelineConfig struct {
	EmbedderCfg embedder.EmbedderConfig `json:"embedder_cfg"`
	ChunkerCfg  ChunkerConfig           `json:"chunker_cfg"`
	SearchOpts  SearchOptions           `json:"search_opts"`
}

// DefaultPipelineConfig returns recommended settings for the Korean RAG system.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		EmbedderCfg: embedder.DefaultEmbedderConfig(),
		ChunkerCfg:  DefaultChunkerConfig(),
		SearchOpts:  DefaultSearchOptions(),
	}
}

// Pipeline orchestrates document indexing, MRL two-stage retrieval, and prompt generation.
type Pipeline struct {
	cfg       PipelineConfig
	emb       *embedder.Embedder
	chunker   *KoreanSemanticChunker
	store     *InMemoryVectorStore
	retriever *SubspaceRetriever
	builder   *PromptBuilder
	generator *PureGoExtractiveGenerator
	llmClient LLMClient
}

// NewPipeline creates an initialized RAG Pipeline.
func NewPipeline(cfg PipelineConfig) (*Pipeline, error) {
	emb, err := embedder.NewEmbedder(cfg.EmbedderCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder for RAG: %w", err)
	}

	chunker := NewKoreanSemanticChunker(cfg.ChunkerCfg)
	store := NewInMemoryVectorStore()
	retriever := NewSubspaceRetriever(emb, store)
	builder := NewPromptBuilder()
	generator := NewPureGoExtractiveGenerator()

	return &Pipeline{
		cfg:       cfg,
		emb:       emb,
		chunker:   chunker,
		store:     store,
		retriever: retriever,
		builder:   builder,
		generator: generator,
	}, nil
}

// SetLLMClient connects an external LLM generation client (e.g. Gemini/Vertex AI).
func (p *Pipeline) SetLLMClient(client LLMClient) {
	p.llmClient = client
}

// VectorStore returns the underlying vector store.
func (p *Pipeline) VectorStore() *InMemoryVectorStore {
	return p.store
}

// IndexDocuments chunks, embeds, and indexes a collection of knowledge documents.
func (p *Pipeline) IndexDocuments(docs []Document) (int, error) {
	var allChunks []Chunk

	for _, doc := range docs {
		chunks := p.chunker.ChunkDocument(doc)
		for i := range chunks {
			// Embed chunk with 3024-d engine
			res, err := p.emb.Embed(chunks[i].Text)
			if err != nil {
				return 0, fmt.Errorf("failed to embed chunk %s: %w", chunks[i].ID, err)
			}
			chunks[i].NormalizedText = res.Metadata.CleanedText
			chunks[i].Embedding = res.Vector
			chunks[i].Metadata = res.Metadata
			chunks[i].Subspaces = res.Subspaces
			allChunks = append(allChunks, chunks[i])
		}
	}

	if err := p.store.AddChunks(allChunks); err != nil {
		return 0, fmt.Errorf("failed to add chunks to store: %w", err)
	}

	return len(allChunks), nil
}

// Query executes end-to-end RAG retrieval, context compilation, and answer synthesis.
func (p *Pipeline) Query(userQuery string, opts *SearchOptions) (*RAGResponse, error) {
	startTime := time.Now()

	searchOpts := p.cfg.SearchOpts
	if opts != nil {
		searchOpts = *opts
	}

	// 1. Retrieve top matching knowledge chunks
	retrievalStart := time.Now()
	results, queryMeta, err := p.retriever.Retrieve(userQuery, searchOpts)
	if err != nil {
		return nil, fmt.Errorf("query retrieval failed: %w", err)
	}
	retrievalTime := time.Since(retrievalStart)

	// 2. Build grounding context and augmented prompt
	augmentedPrompt, contextText := p.builder.BuildAugmentedPrompt(userQuery, queryMeta, results)

	// 3. Generate answer
	var answer string
	if p.llmClient != nil {
		ans, err := p.llmClient.Generate(augmentedPrompt)
		if err != nil {
			// Fallback to local extractive generator on API error
			answer = p.generator.GenerateSynthesizedAnswer(userQuery, results)
		} else {
			answer = ans
		}
	} else {
		answer = p.generator.GenerateSynthesizedAnswer(userQuery, results)
	}

	totalLatency := time.Since(startTime)

	return &RAGResponse{
		Query:           userQuery,
		QueryMetadata:   *queryMeta,
		RetrievedChunks: results,
		ContextText:     contextText,
		AugmentedPrompt: augmentedPrompt,
		GeneratedAnswer: answer,
		RetrievalTime:   retrievalTime,
		TotalLatency:    totalLatency,
	}, nil
}

// Save persists the indexed knowledge chunks to disk.
func (p *Pipeline) Save(path string) error {
	return p.store.Save(path)
}

// Load restores indexed knowledge chunks from disk.
func (p *Pipeline) Load(path string) error {
	return p.store.Load(path)
}
