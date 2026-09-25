package trainer

import (
	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// Triplet represents an (anchor, positive, negative) index triplet.
type Triplet struct {
	Anchor   int
	Positive int
	Negative int
}

// MineSocialTriplets extracts valid triplets from a batch based on SocialStance categories.
func MineSocialTriplets(stances []embedder.SocialStance) []Triplet {
	n := len(stances)
	var triplets []Triplet

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j || stances[i] != stances[j] {
				continue
			}
			// (i, j) is an anchor-positive pair
			for k := 0; k < n; k++ {
				if stances[i] != stances[k] {
					triplets = append(triplets, Triplet{
						Anchor:   i,
						Positive: j,
						Negative: k,
					})
				}
			}
		}
	}
	return triplets
}

// ComputeSocialTripletMarginLoss computes Triplet Margin Loss over the 512-d Social Stance subspace.
func ComputeSocialTripletMarginLoss(
	socialVectors [][]float32,
	stances []embedder.SocialStance,
	margin float32,
) float32 {
	if len(socialVectors) < 3 || len(stances) != len(socialVectors) {
		return 0.0
	}

	if margin <= 0 {
		margin = 0.4
	}

	triplets := MineSocialTriplets(stances)
	if len(triplets) == 0 {
		return 0.0
	}

	var totalLoss float64
	for _, tr := range triplets {
		simAP := embedder.CosineSimilarity(socialVectors[tr.Anchor], socialVectors[tr.Positive])
		simAN := embedder.CosineSimilarity(socialVectors[tr.Anchor], socialVectors[tr.Negative])

		distAP := 1.0 - float64(simAP)
		distAN := 1.0 - float64(simAN)

		loss := distAP - distAN + float64(margin)
		if loss > 0 {
			totalLoss += loss
		}
	}

	return float32(totalLoss / float64(len(triplets)))
}
