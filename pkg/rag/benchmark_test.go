package rag

import (
	"fmt"
	"testing"
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

func BenchmarkRAGRetrieval100Chunks(b *testing.B) {
	emb, err := embedder.NewEmbedder(embedder.DefaultEmbedderConfig())
	if err != nil {
		b.Fatalf("failed to create embedder: %v", err)
	}

	store := NewInMemoryVectorStore()

	// Populate 100 simulated chunks
	baseChunk := Chunk{
		DocID:    "doc_base",
		DocTitle: "사내 공통 업무 규정",
		Text:     "사내 업무 처리 절차 및 결재 승인에 관한 표준 지침서입니다.",
	}
	res, err := emb.Embed(baseChunk.Text)
	if err != nil {
		b.Fatalf("embed error: %v", err)
	}

	var chunks []Chunk
	for i := 0; i < 100; i++ {
		chk := baseChunk
		chk.ID = fmt.Sprintf("chunk_%03d", i)
		chk.ChunkIndex = i + 1
		chk.Embedding = res.Vector
		chk.Metadata = res.Metadata
		chk.Subspaces = res.Subspaces
		chunks = append(chunks, chk)
	}

	if err := store.AddChunks(chunks); err != nil {
		b.Fatalf("AddChunks error: %v", err)
	}

	retriever := NewSubspaceRetriever(emb, store)
	query := "업무 결재 절차 알려주세요."
	opts := DefaultSearchOptions()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := retriever.Retrieve(query, opts)
		if err != nil {
			b.Fatalf("Retrieve error: %v", err)
		}
	}
}

func BenchmarkSemanticChunker(b *testing.B) {
	chunker := NewKoreanSemanticChunker(DefaultChunkerConfig())
	doc := Document{
		ID:        "bench_doc",
		Title:     "대규모 규정 문서",
		Content:   "제1조 회의실을 예약합니다. 제2조 법인카드를 사용합니다. 제3조 재택근무를 승인합니다. 제4조 경조휴가를 지급합니다. 제5조 보안 지침을 준수합니다. 제6조 출퇴근 시간을 기록합니다.",
		CreatedAt: time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = chunker.ChunkDocument(doc)
	}
}
