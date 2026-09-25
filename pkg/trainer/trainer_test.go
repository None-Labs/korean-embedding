package trainer

import (
	"testing"
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

func TestSimCSELoss(t *testing.T) {
	zA := [][]float32{
		embedder.L2Normalize([]float32{1.0, 0.0, 0.0, 0.0}),
		embedder.L2Normalize([]float32{0.0, 1.0, 0.0, 0.0}),
	}
	// zB is close to zA (positive pairs)
	zB := [][]float32{
		embedder.L2Normalize([]float32{0.9, 0.1, 0.0, 0.0}),
		embedder.L2Normalize([]float32{0.1, 0.9, 0.0, 0.0}),
	}

	loss, err := ComputeSimCSELoss(zA, zB, 0.05)
	if err != nil {
		t.Fatalf("ComputeSimCSELoss error: %v", err)
	}
	if loss <= 0.0 {
		t.Errorf("expected positive loss, got %f", loss)
	}
}

func TestIntentContrastiveLoss(t *testing.T) {
	// 3 intent vectors: 0 and 1 have same intent, 2 has different intent
	v0 := embedder.L2Normalize([]float32{1.0, 0.2, 0.0})
	v1 := embedder.L2Normalize([]float32{0.9, 0.3, 0.0})
	v2 := embedder.L2Normalize([]float32{0.0, 0.0, 1.0})

	vectors := [][]float32{v0, v1, v2}
	acts := []transformer.SpeechAct{
		transformer.SpeechActQuestion,
		transformer.SpeechActQuestion,
		transformer.SpeechActStatement,
	}

	loss := ComputeIntentContrastiveLoss(vectors, acts, 0.3)
	if loss < 0.0 {
		t.Errorf("expected non-negative loss, got %f", loss)
	}
}

func TestSocialTripletMarginLoss(t *testing.T) {
	vFormal1 := embedder.L2Normalize([]float32{1.0, 0.1, 0.0})
	vFormal2 := embedder.L2Normalize([]float32{0.95, 0.2, 0.0})
	vInformal := embedder.L2Normalize([]float32{0.0, 0.0, 1.0})

	vectors := [][]float32{vFormal1, vFormal2, vInformal}
	stances := []embedder.SocialStance{
		embedder.SocialStanceFormal,
		embedder.SocialStanceFormal,
		embedder.SocialStanceInformalFriend,
	}

	loss := ComputeSocialTripletMarginLoss(vectors, stances, 0.4)
	if loss < 0.0 {
		t.Errorf("expected non-negative triplet loss, got %f", loss)
	}
}

func TestTrainerStepAndTrain(t *testing.T) {
	emb, err := embedder.NewEmbedder(embedder.DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	cfg := DefaultTrainerConfig()
	cfg.BatchSize = 2
	cfg.AlertInterval = 10 * time.Minute

	tr := NewTrainer(emb, cfg)

	corpus := []string{
		"오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.",
		"금일 안건 회의는 15시 회의실 진행 예정입니다.",
		"야 이번 주말에 롤 ㄱㄱ? 개웃기네 ㅋㅋ",
		"자기야 오늘 저녁에 맛있는 거 먹으러 가용~",
	}

	metrics, err := tr.TrainStep(corpus[:2])
	if err != nil {
		t.Fatalf("TrainStep failed: %v", err)
	}
	if metrics.TotalLoss <= 0 {
		t.Errorf("expected positive total loss, got %f", metrics.TotalLoss)
	}

	allMetrics, err := tr.Train(corpus, 1)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if len(allMetrics) != 2 {
		t.Errorf("expected 2 step metrics, got %d", len(allMetrics))
	}
}
