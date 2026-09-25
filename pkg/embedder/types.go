package embedder

import (
	"fmt"

	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
	"gitlab.com/none-labs/korean-tokenizer/pkg/syntax"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

// Subspace constants defining the 3024-dimensional semantic layout.
const (
	TotalDim   = 3024
	LiteralDim = 1536
	IntentDim  = 512
	SocialDim  = 512
	NuanceDim  = 464

	LiteralOffset = 0
	IntentOffset  = LiteralOffset + LiteralDim
	SocialOffset  = IntentOffset + IntentDim
	NuanceOffset  = SocialOffset + SocialDim
)

// SubspaceType identifies one of the four multi-aspect semantic subspaces.
type SubspaceType string

const (
	SubspaceLiteral SubspaceType = "literal" // Level 1: 1536-d Morphological & Syntactic Literal Proposition
	SubspaceIntent  SubspaceType = "intent"  // Level 2: 512-d 8-Class Speaker Speech Act & Directive Intention
	SubspaceSocial  SubspaceType = "social"  // Level 3: 512-d Social Stance (Formal/Informal) & Age Hierarchy
	SubspaceNuance  SubspaceType = "nuance"  // Level 4: 464-d Pragmatic Nuance, Irony & Sarcastic Attitude
)

// SubspaceInfo holds metadata for a single subspace partition.
type SubspaceInfo struct {
	Type        SubspaceType `json:"type"`
	Offset      int          `json:"offset"`
	Dim         int          `json:"dimension"`
	Description string       `json:"description"`
}

// AllSubspaces lists all 4 semantic subspace specifications in layout order.
var AllSubspaces = []SubspaceInfo{
	{
		Type:        SubspaceLiteral,
		Offset:      LiteralOffset,
		Dim:         LiteralDim,
		Description: "Morpho-syntactic literal semantics and objective proposition (1536-d)",
	},
	{
		Type:        SubspaceIntent,
		Offset:      IntentOffset,
		Dim:         IntentDim,
		Description: "Speaker communicative intent and 8-speech-act projection (512-d)",
	},
	{
		Type:        SubspaceSocial,
		Offset:      SocialOffset,
		Dim:         SocialDim,
		Description: "Social stance (Formal vs Informal) and age-group hierarchy (512-d)",
	},
	{
		Type:        SubspaceNuance,
		Offset:      NuanceOffset,
		Dim:         NuanceDim,
		Description: "Pragmatic nuance, irony/sarcasm, and attitude discrepancy (464-d)",
	},
}

// SocialStance represents communicative relationship framing.
type SocialStance string

const (
	// SocialStanceFormal: 조직 간 대화, 공적 보고, 비즈니스 협의, 고객 응대 (하십시오체/해요체, 객관적 서술)
	SocialStanceFormal SocialStance = "FORMAL_ORGANIZATIONAL"
	// SocialStanceInformalCouple: 연인 간 대화 (친밀도 극대, 애정/감정 표출, 애교체/축약형)
	SocialStanceInformalCouple SocialStance = "INFORMAL_COUPLE"
	// SocialStanceInformalFriend: 친구 간 대화 (해체, 격식 배제, 은어/슬랭)
	SocialStanceInformalFriend SocialStance = "INFORMAL_FRIEND"
	// SocialStanceInformalFamily: 가족 간 대화 (존칭과 친밀함의 혼합, 세대 간 어휘)
	SocialStanceInformalFamily SocialStance = "INFORMAL_FAMILY"
)

// Description returns a human-readable explanation for the social stance.
func (s SocialStance) Description() string {
	switch s {
	case SocialStanceFormal:
		return "공적 대화 (조직/업무/비즈니스/고객응대 - Formal & Organizational)"
	case SocialStanceInformalCouple:
		return "사적 대화 (연인 간 대화 - Romantic & Affectionate)"
	case SocialStanceInformalFriend:
		return "사적 대화 (친구 간 대화 - Casual Peer & Slang)"
	case SocialStanceInformalFamily:
		return "사적 대화 (가족 간 대화 - Kinship & Mixed Politeness)"
	default:
		return "미정의 대화 관계 (Undefined Stance)"
	}
}

// AgeGroup represents the speaker's generational speech style subspace.
type AgeGroup string

const (
	// AgeGroupYouth: 10대~20대 청년층 (신조어, 자모 축약, 빠른 어미 변화)
	AgeGroupYouth AgeGroup = "YOUTH_10_20"
	// AgeGroupMiddle: 30대~40대 중장년층 (직장/실용 중심 문체)
	AgeGroupMiddle AgeGroup = "MIDDLE_30_40"
	// AgeGroupSenior: 50대 이상 장년/노년층 (전통적 경어법, 완곡한 간접 화법)
	AgeGroupSenior AgeGroup = "SENIOR_50_PLUS"
)

// Description returns a human-readable explanation for the age group.
func (ag AgeGroup) Description() string {
	switch ag {
	case AgeGroupYouth:
		return "10대~20대 청년층 (Youth - 신조어, 자모 축약, SNS 메신저체)"
	case AgeGroupMiddle:
		return "30대~40대 중장년층 (Middle Age - 직장/실용 중심 문체)"
	case AgeGroupSenior:
		return "50대 이상 장년/노년층 (Senior - 전통적 경어법, 완곡 어법)"
	default:
		return "연령대 판별 불가 (Neutral Age)"
	}
}

// UtteranceMetadata encapsulates all grammatical, syntactic, tone, and pragmatic features.
type UtteranceMetadata struct {
	OriginalText     string                 `json:"original_text"`
	CleanedText      string                 `json:"cleaned_text"`
	MaskedTags       []string               `json:"masked_tags,omitempty"`
	SpeechAct        transformer.SpeechAct  `json:"speech_act"`
	SpeechActDesc    string                 `json:"speech_act_desc"`
	SpeechActProb    float32                `json:"speech_act_prob"`
	SpeechLevel      morphology.SpeechLevel `json:"speech_level"`
	SpeechLevelDesc  string                 `json:"speech_level_desc"`
	SocialStance     SocialStance           `json:"social_stance"`
	SocialStanceDesc string                 `json:"social_stance_desc"`
	AgeGroup         AgeGroup               `json:"age_group"`
	AgeGroupDesc     string                 `json:"age_group_desc"`
	Politeness       float64                `json:"politeness_score"` // 0.0 (반말) ~ 1.0 (극존칭)
	Formality        float64                `json:"formality_score"`  // 0.0 (비격식) ~ 1.0 (격식)
	AegyoScore       float64                `json:"aegyo_score"`      // 0.0 ~ 1.0 (애교/친근)
	SarcasmDetected  bool                   `json:"sarcasm_detected"`
	SarcasmScore     float64                `json:"sarcasm_score"`     // 0.0 ~ 1.0 (비꼼 확률)
	AttitudePolarity float64                `json:"attitude_polarity"` // -1.0 (냉소/부정) ~ +1.0 (호의/긍정)
	TrueIntent       string                 `json:"true_intent"`
	KeyEnding        string                 `json:"key_ending,omitempty"`
	Morphemes        []morphology.Morpheme  `json:"morphemes,omitempty"`
	SyntaxNodes      []syntax.SyntaxNode    `json:"syntax_nodes,omitempty"`
}

// EmbeddingResult represents the output of the 3024-d embedding pipeline.
type EmbeddingResult struct {
	Dimension int                        `json:"dimension"`
	Vector    []float32                  `json:"vector"` // 3024-d Unit L2-normalized vector
	Metadata  UtteranceMetadata          `json:"metadata"`
	Subspaces map[SubspaceType][]float32 `json:"subspaces,omitempty"`  // L2-normalized subspace vectors
	MRLSlices map[int][]float32          `json:"mrl_slices,omitempty"` // Sliced & normalized at 512, 1024, 1536, 3024
}

// SimilarityResult holds detailed semantic comparison between two utterances across all subspaces.
type SimilarityResult struct {
	Text1             string          `json:"text1"`
	Text2             string          `json:"text2"`
	CosineSimilarity  float32         `json:"cosine_similarity"`  // Full 3024-d cosine similarity
	LiteralSimilarity float32         `json:"literal_similarity"` // 1536-d Literal subspace similarity
	IntentSimilarity  float32         `json:"intent_similarity"`  // 512-d Intent subspace similarity
	SocialSimilarity  float32         `json:"social_similarity"`  // 512-d Social stance similarity
	NuanceSimilarity  float32         `json:"nuance_similarity"`  // 464-d Nuance similarity
	MRLComparison     map[int]float32 `json:"mrl_comparison"`     // Similarities at 512, 1024, 1536, 3024 dims
	IntentMatch       bool            `json:"intent_match"`
	SocialStanceMatch bool            `json:"social_stance_match"`
	IronyDivergence   float32         `json:"irony_divergence"`
}

// SubspaceSlice returns the un-normalized sub-slice of the 3024-d vector for the specified subspace.
func SubspaceSlice(vec []float32, st SubspaceType) ([]float32, error) {
	if len(vec) != TotalDim {
		return nil, fmt.Errorf("vector length must be %d, got %d", TotalDim, len(vec))
	}
	switch st {
	case SubspaceLiteral:
		return vec[LiteralOffset : LiteralOffset+LiteralDim], nil
	case SubspaceIntent:
		return vec[IntentOffset : IntentOffset+IntentDim], nil
	case SubspaceSocial:
		return vec[SocialOffset : SocialOffset+SocialDim], nil
	case SubspaceNuance:
		return vec[NuanceOffset : NuanceOffset+NuanceDim], nil
	default:
		return nil, fmt.Errorf("unknown subspace type: %s", st)
	}
}
