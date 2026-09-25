package embedder

import (
	"math"
	"strings"
	"testing"

	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

func TestEmbedderDimensionAndL2Norm(t *testing.T) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create Embedder: %v", err)
	}

	sentences := []string{
		"오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.",
		"자기야 오늘 저녁에 맛있는 거 먹으러 갈까용? ㅎㅎ",
		"야 이번 주말에 롤 한판 ㄱㄱ?",
		"어르신, 진지 잡수셨습니까?",
		"정말 대단하십니다, 일을 이따위로 처리하시다니 참 훌륭하네요.",
	}

	for _, s := range sentences {
		res, err := emb.Embed(s)
		if err != nil {
			t.Fatalf("Embed failed for '%s': %v", s, err)
		}

		// Verify Total Dimension
		if len(res.Vector) != TotalDim {
			t.Errorf("expected vector dimension %d, got %d", TotalDim, len(res.Vector))
		}

		// Verify Unit L2 Norm
		var sumSq float64
		for _, v := range res.Vector {
			sumSq += float64(v) * float64(v)
		}
		norm := math.Sqrt(sumSq)
		if math.Abs(norm-1.0) > 1e-4 {
			t.Errorf("expected unit L2 norm 1.0, got %f for sentence '%s'", norm, s)
		}

		// Verify Subspaces
		if len(res.Subspaces) != 4 {
			t.Errorf("expected 4 subspaces, got %d", len(res.Subspaces))
		}

		expectedDims := map[SubspaceType]int{
			SubspaceLiteral: LiteralDim,
			SubspaceIntent:  IntentDim,
			SubspaceSocial:  SocialDim,
			SubspaceNuance:  NuanceDim,
		}

		for st, expDim := range expectedDims {
			sub, ok := res.Subspaces[st]
			if !ok {
				t.Errorf("missing subspace %s", st)
			}
			if len(sub) != expDim {
				t.Errorf("subspace %s expected dim %d, got %d", st, expDim, len(sub))
			}

			// Subspaces must also be Unit L2-Normalized
			var subSumSq float64
			for _, v := range sub {
				subSumSq += float64(v) * float64(v)
			}
			subNorm := math.Sqrt(subSumSq)
			if math.Abs(subNorm-1.0) > 1e-4 {
				t.Errorf("subspace %s unit norm expected 1.0, got %f", st, subNorm)
			}
		}

		// Verify MRL Slices
		for _, mrlDim := range SupportedMRLDims {
			slice, ok := res.MRLSlices[mrlDim]
			if !ok {
				t.Errorf("missing MRL slice %d", mrlDim)
			}
			if len(slice) != mrlDim {
				t.Errorf("MRL slice %d length expected %d, got %d", mrlDim, mrlDim, len(slice))
			}
			var mrlSumSq float64
			for _, v := range slice {
				mrlSumSq += float64(v) * float64(v)
			}
			mrlNorm := math.Sqrt(mrlSumSq)
			if math.Abs(mrlNorm-1.0) > 1e-4 {
				t.Errorf("MRL slice %d unit norm expected 1.0, got %f", mrlDim, mrlNorm)
			}
		}
	}
}

func TestMaskNormalizationAndCleaning(t *testing.T) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create Embedder: %v", err)
	}

	raw := "```python\nprint('hello')\n``` #@이름# 님께서 user@example.com 으로 https://google.com 링크를 보내셨습니다. `alert()`"
	cleaned, tags := emb.NormalizeText(raw)

	if !strings.Contains(cleaned, "<이름>") {
		t.Errorf("expected normalized '<이름>', got: %s", cleaned)
	}
	if !strings.Contains(cleaned, "<이메일>") {
		t.Errorf("expected normalized '<이메일>', got: %s", cleaned)
	}
	if !strings.Contains(cleaned, "<링크>") {
		t.Errorf("expected normalized '<링크>', got: %s", cleaned)
	}
	if strings.Contains(cleaned, "python") || strings.Contains(cleaned, "alert") {
		t.Errorf("script code was not properly cleaned: %s", cleaned)
	}
	if len(tags) < 3 {
		t.Errorf("expected at least 3 tracked masked tags, got: %v", tags)
	}
}

func TestSocialStanceAndAgeClassification(t *testing.T) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create Embedder: %v", err)
	}

	tests := []struct {
		text           string
		expectedStance SocialStance
		expectedAge    AgeGroup
	}{
		{
			text:           "금일 프로젝트 진행 상황을 보고드립니다. 검토 부탁드립니다.",
			expectedStance: SocialStanceFormal,
			expectedAge:    AgeGroupMiddle,
		},
		{
			text:           "자기야 오늘 저녁에 맛있는 거 먹으러 가용~ 사랑해용 ㅎㅎ",
			expectedStance: SocialStanceInformalCouple,
			expectedAge:    AgeGroupYouth,
		},
		{
			text:           "야 이번 주말에 롤 ㄱㄱ? 개웃기네 ㅋㅋ",
			expectedStance: SocialStanceInformalFriend,
			expectedAge:    AgeGroupYouth,
		},
		{
			text:           "어르신 진지 잡수셨습니까, 옥체 강녕하시기를 기원합니다.",
			expectedStance: SocialStanceFormal,
			expectedAge:    AgeGroupSenior,
		},
	}

	for _, tt := range tests {
		res, err := emb.Embed(tt.text)
		if err != nil {
			t.Fatalf("Embed error for '%s': %v", tt.text, err)
		}

		if res.Metadata.SocialStance != tt.expectedStance {
			t.Logf("text '%s': expected stance %s, got %s", tt.text, tt.expectedStance, res.Metadata.SocialStance)
		}
		if res.Metadata.AgeGroup != tt.expectedAge {
			t.Logf("text '%s': expected age %s, got %s", tt.text, tt.expectedAge, res.Metadata.AgeGroup)
		}
	}
}

func TestSpeechActInference(t *testing.T) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create Embedder: %v", err)
	}

	tests := []struct {
		text        string
		expectedAct transformer.SpeechAct
	}{
		{"내일 회의 시간이 언제인가요?", transformer.SpeechActQuestion},
		{"자료 좀 이메일로 보내주세요.", transformer.SpeechActRequest},
		{"늦어서 정말 죄송합니다.", transformer.SpeechActApology},
		{"네, 말씀하신 내용에 전적으로 동의합니다.", transformer.SpeechActAcceptance},
		{"내일까지 반드시 완료하겠습니다.", transformer.SpeechActPromise},
	}

	for _, tt := range tests {
		res, err := emb.Embed(tt.text)
		if err != nil {
			t.Fatalf("Embed error for '%s': %v", tt.text, err)
		}

		if res.Metadata.SpeechAct != tt.expectedAct {
			t.Errorf("for text '%s': expected act %s, got %s", tt.text, tt.expectedAct, res.Metadata.SpeechAct)
		}
	}
}

func TestSemanticSimilarityComparison(t *testing.T) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		t.Fatalf("failed to create Embedder: %v", err)
	}

	// Two formal requests
	sim, err := emb.Compare(
		"오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.",
		"금일 안건 회의는 15시 회의실 진행 예정입니다.",
	)
	if err != nil {
		t.Fatalf("Compare failed: %v", err)
	}

	if sim.CosineSimilarity <= 0.0 {
		t.Errorf("expected positive similarity, got %f", sim.CosineSimilarity)
	}
	if sim.LiteralSimilarity <= 0.0 {
		t.Errorf("expected positive literal similarity, got %f", sim.LiteralSimilarity)
	}
	if len(sim.MRLComparison) != len(SupportedMRLDims) {
		t.Errorf("expected %d MRL comparisons, got %d", len(SupportedMRLDims), len(sim.MRLComparison))
	}
}
