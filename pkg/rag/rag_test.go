package rag

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

func TestKoreanSemanticChunker(t *testing.T) {
	cfg := ChunkerConfig{
		MaxRunesPerChunk: 100,
		MinRunesPerChunk: 20,
		OverlapSentences: 1,
	}
	chunker := NewKoreanSemanticChunker(cfg)

	doc := Document{
		ID:    "doc_001",
		Title: "사내 연차 휴가 및 복지 규정",
		Content: `제1조 (목적) 본 규정은 임직원의 연차 유급휴가 및 복리후생에 관한 제반 사항을 규정함을 목적으로 합니다.
제2조 (연차 발생) 입사 1년 미만 직원은 1개월 개근 시 1일의 유급휴가가 발생합니다. 1년 이상 근속 시 연간 15일의 연차가 부여됩니다.
제3조 (신청 절차) 휴가 신청은 최소 3일 전 사내 인트라넷 전자결재 시스템을 통해 부서장의 승인을 득해야 합니다.
제4조 (경조 휴가) 본인 결혼 시 5일, 부모상 시 5일의 유급 경조휴가가 지급됩니다.`,
		Category:  "사내규정",
		CreatedAt: time.Now(),
	}

	chunks := chunker.ChunkDocument(doc)
	if len(chunks) == 0 {
		t.Fatal("expected chunks to be created, got 0")
	}

	for _, chk := range chunks {
		if chk.DocID != doc.ID {
			t.Errorf("expected docID %s, got %s", doc.ID, chk.DocID)
		}
		if chk.Text == "" {
			t.Errorf("chunk %s text is empty", chk.ID)
		}
	}
}

func TestVectorStoreAndMRLRetrieval(t *testing.T) {
	emb, err := embedder.NewEmbedder(embedder.DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	store := NewInMemoryVectorStore()

	texts := []struct {
		title  string
		text   string
		stance embedder.SocialStance
	}{
		{
			title:  "회의실 예약 가이드",
			text:   "대회의실 및 소회의실 예약은 사내 캘린더를 통해 1주일 전까지 신청 가능합니다.",
			stance: embedder.SocialStanceFormal,
		},
		{
			title:  "원격 근무 지침",
			text:   "주 2회 재택근무가 가능하며, 코어타임인 오전 10시부터 오후 4시까지는 슬랙에 접속해 있어야 합니다.",
			stance: embedder.SocialStanceFormal,
		},
		{
			title:  "사내 동호회 지원 안내",
			text:   "5인 이상 구성된 취미 동호회는 매월 활동비 10만원을 법인카드로 지원받을 수 있습니다.",
			stance: embedder.SocialStanceFormal,
		},
	}

	var chunks []Chunk
	for i, item := range texts {
		res, err := emb.Embed(item.text)
		if err != nil {
			t.Fatalf("embed error: %v", err)
		}
		chunks = append(chunks, Chunk{
			ID:             res.Metadata.CleanedText,
			DocID:          item.title,
			DocTitle:       item.title,
			ChunkIndex:     i + 1,
			Text:           item.text,
			NormalizedText: res.Metadata.CleanedText,
			Embedding:      res.Vector,
			Metadata:       res.Metadata,
			Subspaces:      res.Subspaces,
		})
	}

	if err := store.AddChunks(chunks); err != nil {
		t.Fatalf("failed to add chunks: %v", err)
	}

	if store.Count() != 3 {
		t.Errorf("expected 3 chunks in store, got %d", store.Count())
	}

	// Query: 회의실 사용하고 싶은데 어떻게 예약하나요?
	qRes, err := emb.Embed("회의실 사용하고 싶은데 어떻게 예약하나요?")
	if err != nil {
		t.Fatalf("query embed failed: %v", err)
	}

	searchOpts := DefaultSearchOptions()
	searchOpts.TopK = 1
	searchOpts.QueryText = "회의실 사용하고 싶은데 어떻게 예약하나요?"

	results, err := store.Search(qRes.Vector, qRes.Subspaces, searchOpts)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least 1 result, got 0")
	}

	topDoc := results[0].Chunk.DocTitle
	if topDoc != "회의실 예약 가이드" {
		t.Errorf("expected top doc '회의실 예약 가이드', got '%s' (score: %.4f)", topDoc, results[0].Score)
	}
}

func TestRAGPipelineEndToEnd(t *testing.T) {
	cfg := DefaultPipelineConfig()
	pipeline, err := NewPipeline(cfg)
	if err != nil {
		t.Fatalf("failed to create RAG Pipeline: %v", err)
	}

	docs := []Document{
		{
			ID:    "policy_01",
			Title: "법인카드 사용 규정",
			Content: `제1조 법인카드는 업무와 직접 관련된 식대 및 교통비 지출에만 사용할 수 있습니다.
개인적인 용도의 결제는 엄격히 금지되며 적발 시 환수 조치됩니다.
제2조 건당 50만원 이상의 지출은 집행 전 부서장의 사전 품의 승인을 받아야 합니다.`,
			Category:  "재무규정",
			CreatedAt: time.Now(),
		},
		{
			ID:    "policy_02",
			Title: "장비 지급 및 보안 수칙",
			Content: `제1조 입사 시 최신형 노트북과 모니터가 1인당 1대씩 기본 지급됩니다.
제2조 사내 모든 장비에는 인가된 보안 백신 및 DRM 솔루션이 반드시 설치되어 있어야 합니다.
외부 반출 시에는 보안 담당자의 반출 승인이 필요합니다.`,
			Category:  "IT보안",
			CreatedAt: time.Now(),
		},
	}

	count, err := pipeline.IndexDocuments(docs)
	if err != nil {
		t.Fatalf("IndexDocuments error: %v", err)
	}
	if count == 0 {
		t.Fatal("expected indexed chunks, got 0")
	}

	// Test Query 1: 법인카드로 50만원 넘게 결제하려면 어떻게 해야 하나요?
	resp, err := pipeline.Query("법인카드로 50만원 넘게 결제하려면 어떻게 해야 하나요?", nil)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if len(resp.RetrievedChunks) == 0 {
		t.Fatal("expected retrieved chunks, got 0")
	}

	topDoc := resp.RetrievedChunks[0].Chunk.DocTitle
	if topDoc != "법인카드 사용 규정" {
		t.Errorf("expected top doc '법인카드 사용 규정', got '%s'", topDoc)
	}

	if !strings.Contains(resp.AugmentedPrompt, "50만원") {
		t.Errorf("expected augmented prompt to contain retrieved context, got: %s", resp.AugmentedPrompt)
	}

	if resp.GeneratedAnswer == "" {
		t.Errorf("expected non-empty generated answer")
	}

	// Test Save and Load
	tmpDir := t.TempDir()
	savePath := filepath.Join(tmpDir, "rag_index.gob")

	if err := pipeline.Save(savePath); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	newPipeline, err := NewPipeline(cfg)
	if err != nil {
		t.Fatalf("NewPipeline error: %v", err)
	}

	if err := newPipeline.Load(savePath); err != nil {
		t.Fatalf("Load error: %v", err)
	}

	if newPipeline.VectorStore().Count() != count {
		t.Errorf("expected loaded count %d, got %d", count, newPipeline.VectorStore().Count())
	}
}
