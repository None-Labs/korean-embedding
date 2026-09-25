package trainer

import (
	"fmt"
	"math"
	"math/rand"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// ApplyDropout creates a perturbed representation by zeroing random dimensions with probability p.
func ApplyDropout(vec []float32, rate float32) []float32 {
	if rate <= 0.0 {
		return vec
	}
	out := make([]float32, len(vec))
	scale := 1.0 / (1.0 - rate)
	for i, v := range vec {
		if rand.Float32() >= rate {
			out[i] = v * scale
		}
	}
	return embedder.L2Normalize(out)
}

// ComputeSimCSELoss calculates unsupervised SimCSE InfoNCE loss over a batch.
// zA and zB are two forward pass representations of the same batch sentences under independent dropout.
// Temperature tau scales the dot products (typically 0.05).
func ComputeSimCSELoss(zA, zB [][]float32, tau float32) (float32, error) {
	n := len(zA)
	if n == 0 || len(zB) != n {
		return 0, fmt.Errorf("batch size mismatch or empty: zA=%d, zB=%d", n, len(zB))
	}
	if tau <= 0 {
		tau = 0.05
	}

	invTau := 1.0 / tau
	var totalLoss float64

	for i := 0; i < n; i++ {
		// Positive pair similarity
		posSim := float64(embedder.CosineSimilarity(zA[i], zB[i])) * float64(invTau)

		// Denominator: sum of similarities over all zB_j in batch
		// Numerical stability: subtract max score
		sims := make([]float64, n)
		maxSim := -math.MaxFloat64
		for j := 0; j < n; j++ {
			s := float64(embedder.CosineSimilarity(zA[i], zB[j])) * float64(invTau)
			sims[j] = s
			if s > maxSim {
				maxSim = s
			}
		}

		var sumExp float64
		for j := 0; j < n; j++ {
			sumExp += math.Exp(sims[j] - maxSim)
		}

		logSumExp := maxSim + math.Log(sumExp)
		// InfoNCE: -log( exp(posSim) / sum_j exp(sim_ij) ) = logSumExp - posSim
		loss_i := logSumExp - posSim
		if loss_i < 0 {
			loss_i = 0
		}
		totalLoss += loss_i
	}

	return float32(totalLoss / float64(n)), nil
}
