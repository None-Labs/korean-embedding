package trainer

import (
	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

// ComputeIntentContrastiveLoss computes supervised contrastive loss on the 512-d Intent subspace.
// Sentences with identical SpeechAct are pulled closer; differing SpeechActs are pushed beyond margin.
func ComputeIntentContrastiveLoss(
	intentVectors [][]float32,
	acts []transformer.SpeechAct,
	margin float32,
) float32 {
	n := len(intentVectors)
	if n < 2 || len(acts) != n {
		return 0.0
	}

	if margin <= 0 {
		margin = 0.3
	}

	var posLoss float64
	var posCount int
	var negLoss float64
	var negCount int

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sim := embedder.CosineSimilarity(intentVectors[i], intentVectors[j])

			if acts[i] == acts[j] {
				// Same speech act: maximize similarity -> minimize (1 - sim)
				diff := 1.0 - float64(sim)
				if diff < 0 {
					diff = 0
				}
				posLoss += diff
				posCount++
			} else {
				// Different speech act: push similarity below margin
				diff := float64(sim - margin)
				if diff > 0 {
					negLoss += diff
				}
				negCount++
			}
		}
	}

	var avgPos, avgNeg float64
	if posCount > 0 {
		avgPos = posLoss / float64(posCount)
	}
	if negCount > 0 {
		avgNeg = negLoss / float64(negCount)
	}

	return float32(avgPos + avgNeg)
}
