package embedder

import (
	"math"

	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
)

// PoolingMethod denotes the token sequence pooling strategy.
type PoolingMethod string

const (
	PoolingAttentionWeighted PoolingMethod = "attention"
	PoolingMean              PoolingMethod = "mean"
	PoolingMax               PoolingMethod = "max"
)

// AttentionPooling computes an attention-weighted aggregate over token hidden representations,
// taking Korean morphological priors and syntactic importance into account.
type AttentionPooling struct {
	HiddenDim int
	WAttn     []float32 // (HiddenDim x HiddenDim)
	VAttn     []float32 // (HiddenDim)
	BAttn     []float32 // (HiddenDim)
}

// NewAttentionPooling creates an initialized attention pooling module.
func NewAttentionPooling(hiddenDim int) *AttentionPooling {
	return &AttentionPooling{
		HiddenDim: hiddenDim,
		WAttn:     initXavier(hiddenDim, hiddenDim),
		VAttn:     initXavier(hiddenDim, 1),
		BAttn:     make([]float32, hiddenDim),
	}
}

// Pool performs attention-weighted pooling over a sequence of token vectors of shape (seqLen x hiddenDim).
// morphTags and isRoot can be nil or provide token-level POS tags and syntactic root flags.
// mask: slice of length seqLen (1 for valid token, 0 for padding).
func (ap *AttentionPooling) Pool(
	hiddenSeq []float32,
	seqLen int,
	mask []float32,
	morphTags []morphology.POSTag,
	isRoot []bool,
) []float32 {
	d := ap.HiddenDim
	if seqLen == 0 || len(hiddenSeq) < seqLen*d {
		return make([]float32, d)
	}

	scores := make([]float64, seqLen)
	maxScore := -math.MaxFloat64

	// 1. Calculate raw energy scores e_i = v^T tanh(W h_i + b) + morphological bias
	for i := 0; i < seqLen; i++ {
		if mask != nil && i < len(mask) && mask[i] == 0.0 {
			scores[i] = -1e9
			continue
		}

		tokVec := hiddenSeq[i*d : (i+1)*d]
		// Project: u = tanh(W tokVec + b)
		var e float64
		for j := 0; j < d; j++ {
			var u_j float32
			for k := 0; k < d; k++ {
				u_j += tokVec[k] * ap.WAttn[k*d+j]
			}
			u_j += ap.BAttn[j]
			// tanh activation
			tanh_u := float32(math.Tanh(float64(u_j)))
			e += float64(tanh_u * ap.VAttn[j])
		}

		// Morphological & Syntactic semantic bonus
		if morphTags != nil && i < len(morphTags) {
			tag := morphTags[i]
			if tag.IsSubstantive() || tag.IsPredicate() {
				// Substantive (NNG, NNP, NP) or Predicate (VV, VA, VX) carries core proposition
				e += 1.2
			} else if tag == morphology.TagMAG || tag == morphology.TagXR {
				e += 0.8
			} else if tag.IsParticle() || tag.IsEnding() {
				e += 0.2
			}
		}

		// Syntactic Root predicate bonus
		if isRoot != nil && i < len(isRoot) && isRoot[i] {
			e += 1.8
		}

		scores[i] = e
		if e > maxScore {
			maxScore = e
		}
	}

	// 2. Softmax over valid tokens
	var sumExp float64
	expScores := make([]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		if mask != nil && i < len(mask) && mask[i] == 0.0 {
			expScores[i] = 0.0
			continue
		}
		expScores[i] = math.Exp(scores[i] - maxScore)
		sumExp += expScores[i]
	}

	if sumExp <= 0 {
		sumExp = 1.0
	}

	// 3. Attention-weighted summation
	pooled := make([]float32, d)
	for i := 0; i < seqLen; i++ {
		if expScores[i] == 0.0 {
			continue
		}
		alpha := float32(expScores[i] / sumExp)
		tokVec := hiddenSeq[i*d : (i+1)*d]
		for j := 0; j < d; j++ {
			pooled[j] += alpha * tokVec[j]
		}
	}

	return pooled
}

// MeanPooling computes average vector over valid sequence positions.
func MeanPooling(hiddenSeq []float32, seqLen, hiddenDim int, mask []float32) []float32 {
	if seqLen == 0 || hiddenDim == 0 {
		return make([]float32, hiddenDim)
	}

	pooled := make([]float32, hiddenDim)
	var count float32

	for i := 0; i < seqLen; i++ {
		if mask != nil && i < len(mask) && mask[i] == 0.0 {
			continue
		}
		tokVec := hiddenSeq[i*hiddenDim : (i+1)*hiddenDim]
		for j := 0; j < hiddenDim; j++ {
			pooled[j] += tokVec[j]
		}
		count += 1.0
	}

	if count > 0 {
		invCount := 1.0 / count
		for j := 0; j < hiddenDim; j++ {
			pooled[j] *= invCount
		}
	}
	return pooled
}

// MaxPooling computes dimension-wise maximum across valid sequence positions.
func MaxPooling(hiddenSeq []float32, seqLen, hiddenDim int, mask []float32) []float32 {
	if seqLen == 0 || hiddenDim == 0 {
		return make([]float32, hiddenDim)
	}

	pooled := make([]float32, hiddenDim)
	for j := range pooled {
		pooled[j] = -math.MaxFloat32
	}

	hasValid := false
	for i := 0; i < seqLen; i++ {
		if mask != nil && i < len(mask) && mask[i] == 0.0 {
			continue
		}
		hasValid = true
		tokVec := hiddenSeq[i*hiddenDim : (i+1)*hiddenDim]
		for j := 0; j < hiddenDim; j++ {
			if tokVec[j] > pooled[j] {
				pooled[j] = tokVec[j]
			}
		}
	}

	if !hasValid {
		return make([]float32, hiddenDim)
	}
	return pooled
}
