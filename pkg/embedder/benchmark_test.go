package embedder

import (
	"testing"
)

func BenchmarkEmbedSingle(b *testing.B) {
	emb, err := NewEmbedder(DefaultEmbedderConfig())
	if err != nil {
		b.Fatalf("failed to create embedder: %v", err)
	}

	text := "오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다."
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := emb.Embed(text)
		if err != nil {
			b.Fatalf("Embed error: %v", err)
		}
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	v1 := make([]float32, TotalDim)
	v2 := make([]float32, TotalDim)
	for i := range v1 {
		v1[i] = 1.0 / float32(TotalDim)
		v2[i] = 1.0 / float32(TotalDim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CosineSimilarity(v1, v2)
	}
}

func BenchmarkMRLSlicing(b *testing.B) {
	v := make([]float32, TotalDim)
	for i := range v {
		v[i] = 1.0 / float32(TotalDim)
	}
	slicer := NewMRLSlicer()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = slicer.Slice(v, 512)
		_, _ = slicer.Slice(v, 1024)
		_, _ = slicer.Slice(v, 1536)
	}
}
