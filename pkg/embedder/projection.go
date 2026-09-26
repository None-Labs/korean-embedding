package embedder

import (
	"math"
	"math/rand"

	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

// SubspaceProjection maps latent hidden representation h_sent to the 3024-dimensional
// multi-aspect semantic space, conditioned on communicative metadata.
type SubspaceProjection struct {
	HiddenDim int

	// Level 1: Literal Semantic Projection (HiddenDim -> 1536)
	WLiteral []float32
	BLiteral []float32

	// Level 2: Speaker Intent Projection (HiddenDim + 8 -> 512)
	WIntent []float32
	BIntent []float32

	// Level 3: Social Stance & Age Hierarchy Projection (HiddenDim + 12 -> 512)
	WSocial []float32
	BSocial []float32

	// Level 4: Pragmatic Nuance & Irony Projection (HiddenDim + 6 -> 464)
	WNuance []float32
	BNuance []float32
}

// NewSubspaceProjection initializes projection matrices with Xavier/Glorot uniform weights.
func NewSubspaceProjection(hiddenDim int) *SubspaceProjection {
	p := &SubspaceProjection{
		HiddenDim: hiddenDim,
		WLiteral:  initXavier(hiddenDim, LiteralDim),
		BLiteral:  make([]float32, LiteralDim),

		WIntent: initXavier(hiddenDim+8, IntentDim),
		BIntent: make([]float32, IntentDim),

		WSocial: initXavier(hiddenDim+12, SocialDim),
		BSocial: make([]float32, SocialDim),

		WNuance: initXavier(hiddenDim+6, NuanceDim),
		BNuance: make([]float32, NuanceDim),
	}
	return p
}

// initXavier creates a weight matrix of size (inDim x outDim) initialized with Xavier scale.
func initXavier(inDim, outDim int) []float32 {
	scale := float32(math.Sqrt(6.0 / float64(inDim+outDim)))
	w := make([]float32, inDim*outDim)
	rng := rand.New(rand.NewSource(int64(inDim*1000 + outDim)))
	for i := range w {
		w[i] = (rng.Float32()*2.0 - 1.0) * scale
	}
	return w
}

// Project performs multi-aspect subspace projection and outputs a unit L2-normalized 3024-d vector.
func (p *SubspaceProjection) Project(hSent []float32, meta *UtteranceMetadata) []float32 {
	if len(hSent) < p.HiddenDim {
		// Pad with zeros if shorter
		padded := make([]float32, p.HiddenDim)
		copy(padded, hSent)
		hSent = padded
	} else if len(hSent) > p.HiddenDim {
		hSent = hSent[:p.HiddenDim]
	}

	// 1. Literal Subspace (1536-d)
	zLit := matVecMul(p.WLiteral, hSent, p.HiddenDim, LiteralDim, p.BLiteral)

	// 2. Speaker Intent Subspace (512-d)
	// Augment hSent with 8-class speech act prior
	hIntent := make([]float32, p.HiddenDim+8)
	copy(hIntent, hSent)
	intentOffset := p.HiddenDim
	if meta != nil {
		for i, sa := range transformer.AllSpeechActs {
			if sa == meta.SpeechAct {
				prob := meta.SpeechActProb
				if prob <= 0 {
					prob = 1.0
				}
				hIntent[intentOffset+i] = prob
			}
		}
	}
	zInt := matVecMul(p.WIntent, hIntent, p.HiddenDim+8, IntentDim, p.BIntent)

	// 3. Social Stance & Hierarchy Subspace (512-d)
	// Augment with social stance, speech level, politeness, formality, age group
	hSocial := make([]float32, p.HiddenDim+12)
	copy(hSocial, hSent)
	socOffset := p.HiddenDim
	if meta != nil {
		hSocial[socOffset+0] = float32(meta.Politeness)
		hSocial[socOffset+1] = float32(meta.Formality)
		hSocial[socOffset+2] = float32(meta.AegyoScore)

		// Social stance one-hot (4 classes)
		switch meta.SocialStance {
		case SocialStanceFormal:
			hSocial[socOffset+3] = 1.0
		case SocialStanceInformalCouple:
			hSocial[socOffset+4] = 1.0
		case SocialStanceInformalFriend:
			hSocial[socOffset+5] = 1.0
		case SocialStanceInformalFamily:
			hSocial[socOffset+6] = 1.0
		}

		// Age group one-hot (3 classes)
		switch meta.AgeGroup {
		case AgeGroupYouth:
			hSocial[socOffset+7] = 1.0
		case AgeGroupMiddle:
			hSocial[socOffset+8] = 1.0
		case AgeGroupSenior:
			hSocial[socOffset+9] = 1.0
		}

		// Speech level indicators (Hasipsio / Haeyo vs Hae)
		if meta.SpeechLevel == morphology.SpeechLevelHasipsio {
			hSocial[socOffset+10] = 1.0
		} else if meta.SpeechLevel == morphology.SpeechLevelHae || meta.SpeechLevel == morphology.SpeechLevelHaera {
			hSocial[socOffset+11] = 1.0
		}
	}
	zSoc := matVecMul(p.WSocial, hSocial, p.HiddenDim+12, SocialDim, p.BSocial)

	// 4. Pragmatic Nuance & Irony Subspace (464-d)
	// Augment with sarcasm, attitude polarity, and irony signals
	hNuance := make([]float32, p.HiddenDim+6)
	copy(hNuance, hSent)
	nuaOffset := p.HiddenDim
	if meta != nil {
		hNuance[nuaOffset+0] = float32(meta.SarcasmScore)
		if meta.SarcasmDetected {
			hNuance[nuaOffset+1] = 1.0
		}
		hNuance[nuaOffset+2] = float32(meta.AttitudePolarity)
		// Discrepancy signal: high politeness with negative attitude
		if meta.Politeness > 0.7 && meta.AttitudePolarity < -0.2 {
			hNuance[nuaOffset+3] = 1.0
		}
		// Sarcastic true intent
		if meta.TrueIntent == "비꼼/반어" {
			hNuance[nuaOffset+4] = 1.0
		} else if meta.TrueIntent == "호의/긍정" {
			hNuance[nuaOffset+5] = 1.0
		}
	}
	zNua := matVecMul(p.WNuance, hNuance, p.HiddenDim+6, NuanceDim, p.BNuance)

	// Concatenate into 3024-d raw vector
	out := make([]float32, TotalDim)
	copy(out[LiteralOffset:], zLit)
	copy(out[IntentOffset:], zInt)
	copy(out[SocialOffset:], zSoc)
	copy(out[NuanceOffset:], zNua)

	// Unit L2-Normalize
	return L2Normalize(out)
}

// matVecMul calculates: y = W^T x + b where W is (inDim x outDim) stored in row-major order.
func matVecMul(W []float32, x []float32, inDim, outDim int, b []float32) []float32 {
	y := make([]float32, outDim)
	for j := 0; j < outDim; j++ {
		var sum float32
		if b != nil && len(b) > j {
			sum = b[j]
		}
		for i := 0; i < inDim; i++ {
			sum += x[i] * W[i*outDim+j]
		}
		y[j] = sum
	}
	return y
}

// L2Normalize returns a unit-norm vector.
func L2Normalize(vec []float32) []float32 {
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm < 1e-9 {
		norm = 1.0
	}

	res := make([]float32, len(vec))
	invNorm := 1.0 / norm
	for i, v := range vec {
		res[i] = v * invNorm
	}
	return res
}

// ExtractSubspace extracts a single subspace from a 3024-d vector and optionally unit-normalizes it.
func ExtractSubspace(vec []float32, st SubspaceType, normalize bool) ([]float32, error) {
	raw, err := SubspaceSlice(vec, st)
	if err != nil {
		return nil, err
	}
	sub := make([]float32, len(raw))
	copy(sub, raw)
	if normalize {
		return L2Normalize(sub), nil
	}
	return sub, nil
}

// CosineSimilarity computes cosine similarity between two unit-normalized vectors.
func CosineSimilarity(v1, v2 []float32) float32 {
	if len(v1) != len(v2) || len(v1) == 0 {
		return 0.0
	}
	var dot float32
	for i := range v1 {
		dot += v1[i] * v2[i]
	}
	return dot
}
