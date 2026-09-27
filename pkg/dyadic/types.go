package dyadic

import (
	"time"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// DynamicPattern identifies the 4 dyadic emotional dynamics patterns defined in Section 4.3.
type DynamicPattern string

const (
	PatternExplosion   DynamicPattern = "ACUTE_EXPLOSION"    // 급성 감정 폭발 (Acute Emotional Explosion)
	PatternResonance   DynamicPattern = "RAPPORT_RESONANCE"  // 정서적 공명 및 이완 (Rapport Resonance & Attenuation)
	PatternFreezing    DynamicPattern = "LATENT_FREEZING"    // 잠복형 정서 동결 (Latent Emotional Freezing)
	PatternOscillation DynamicPattern = "AMBIVALENT_OSCILLATION" // 양가성 정서 진동 (Ambivalent Emotional Oscillation)
	PatternBaseline    DynamicPattern = "BASELINE_CONVERSATION" // 일상적 정보/친밀 대화
)

// KoreanName returns a human-readable Korean name for the dynamic pattern.
func (p DynamicPattern) KoreanName() string {
	switch p {
	case PatternExplosion:
		return "급성 감정 폭발 (Acute Explosion)"
	case PatternResonance:
		return "정서적 공명/이완 (Rapport Resonance)"
	case PatternFreezing:
		return "잠복형 정서 동결 (Latent Freezing)"
	case PatternOscillation:
		return "양가성 정서 진동 (Ambivalent Oscillation)"
	case PatternBaseline:
		return "일상 기저 대화 (Baseline Routine)"
	default:
		return "미정의 패턴 (Undefined)"
	}
}

// ChatMessage represents a raw line in the KakaoTalk chat export.
type ChatMessage struct {
	Timestamp time.Time `json:"timestamp"`
	Speaker   string    `json:"speaker"`
	Text      string    `json:"text"`
	IsMedia   bool      `json:"is_media"`
}

// ParagraphTurn represents a unified paragraph of contiguous utterances from one speaker.
type ParagraphTurn struct {
	ID             int                      `json:"id"`
	Speaker        string                   `json:"speaker"`
	StartTime      time.Time                `json:"start_time"`
	EndTime        time.Time                `json:"end_time"`
	RawMessages    []string                 `json:"raw_messages"`
	MergedText     string                   `json:"merged_text"`
	CleanedText    string                   `json:"cleaned_text"`
	CharCount      int                      `json:"char_count"`
	MessageCount   int                      `json:"message_count"`
	MediaOnly      bool                     `json:"media_only"`
	Embedding      *embedder.EmbeddingResult `json:"embedding,omitempty"`
}

// DyadicTurnPair represents a 1-on-1 interaction stimulus-response sequence.
type DyadicTurnPair struct {
	PairIndex          int             `json:"pair_index"`
	ClientTurnID       int             `json:"client_turn_id"`
	PartnerTurnID      int             `json:"partner_turn_id"`
	Timestamp          time.Time       `json:"timestamp"`
	PartnerText        string          `json:"partner_text"`
	ClientText         string          `json:"client_text"`
	TurnLatencySec     float64         `json:"turn_latency_sec"`     // Δt_turn
	DyadicDivergence   float64         `json:"dyadic_divergence"`   // 1 - CosSim(z_client, z_partner)
	IntraDivergence    float64         `json:"intra_divergence"`    // 1 - CosSim(z_client, z_prev_client)
	RapportScore       float64         `json:"rapport_score"`       // CosSim(z_social_client, z_social_partner)
	NuanceDiscrepancy  float64         `json:"nuance_discrepancy"`  // Pragmatic nuance shift
	AttitudePolarity   float64         `json:"attitude_polarity"`   // Client polarity (-1.0 ~ +1.0)
	PolarityShift      float64         `json:"polarity_shift"`      // |Polarity_k - Polarity_{k-1}|
	SarcasmScore       float64         `json:"sarcasm_score"`
	SpeechAct          string          `json:"speech_act"`
	SocialStance       string          `json:"social_stance"`
	EVI                float64         `json:"evi"`                 // Dyadic Emotional Volatility Index
	IsMediaStimulus    bool            `json:"is_media_stimulus"`
	IsMediaResponse    bool            `json:"is_media_response"`
	Pattern            DynamicPattern  `json:"pattern"`
	PatternReason      string          `json:"pattern_reason"`
}

// VolatilePeak highlights a significant emotional volatility event for clinical review.
type VolatilePeak struct {
	Rank           int             `json:"rank"`
	PairIndex      int             `json:"pair_index"`
	Timestamp      time.Time       `json:"timestamp"`
	EVI            float64         `json:"evi"`
	Pattern        DynamicPattern  `json:"pattern"`
	PartnerPrompt  string          `json:"partner_prompt"`
	ClientReaction string          `json:"client_reaction"`
	TurnLatencySec float64         `json:"turn_latency_sec"`
	DyadicDiv      float64         `json:"dyadic_divergence"`
	IntraDiv       float64         `json:"intra_divergence"`
	NuanceShift    float64         `json:"nuance_shift"`
	Polarity       float64         `json:"polarity"`
	Sarcasm        float64         `json:"sarcasm"`
	IsMediaStim    bool            `json:"is_media_stimulus"`
	ClinicalNote   string          `json:"clinical_note"`
}

// DyadicAnalysisReport synthesizes the full longitudinal chat analysis.
type DyadicAnalysisReport struct {
	TotalRawMessages      int                       `json:"total_raw_messages"`
	TotalParagraphTurns   int                       `json:"total_paragraph_turns"`
	ClientParagraphTurns  int                       `json:"client_paragraph_turns"`
	PartnerParagraphTurns int                       `json:"partner_paragraph_turns"`
	TotalDyadicPairs      int                       `json:"total_dyadic_pairs"`
	StartDate             time.Time                 `json:"start_date"`
	EndDate               time.Time                 `json:"end_date"`
	DurationDays          float64                   `json:"duration_days"`
	MeanEVI               float64                   `json:"mean_evi"`
	MedianEVI             float64                   `json:"median_evi"`
	MaxEVI                float64                   `json:"max_evi"`
	EVIStandardDev        float64                   `json:"evi_std_dev"`
	PatternCounts         map[DynamicPattern]int    `json:"pattern_counts"`
	TopVolatilePeaks      []VolatilePeak            `json:"top_volatile_peaks"`
	TopTextPeaks          []VolatilePeak            `json:"top_text_peaks"`
	MonthlyTrends         []MonthlyTrend            `json:"monthly_trends"`
	InteractionPairs      []DyadicTurnPair          `json:"interaction_pairs,omitempty"`
}

// MonthlyTrend tracks the longitudinal evolution of emotional volatility per calendar month.
type MonthlyTrend struct {
	YearMonth    string  `json:"year_month"`
	PairCount    int     `json:"pair_count"`
	MeanEVI      float64 `json:"mean_evi"`
	MaxEVI       float64 `json:"max_evi"`
	Explosions   int     `json:"explosions"`
	Freezings    int     `json:"freezings"`
	Resonances   int     `json:"resonances"`
	Oscillations int     `json:"oscillations"`
}

// AnalysisConfig controls paragraph chunking and EVI formula weighting.
type AnalysisConfig struct {
	ClientSpeakerName   string  // e.g. "미르"
	PartnerSpeakerName  string  // e.g. "🎈"
	MaxTurnGapMinutes   float64 // e.g. 30.0 (Split into new paragraph if same speaker gap > 30m)
	WeightDyadicDiv     float64 // α (default: 0.35)
	WeightIntraDiv      float64 // β (default: 0.35)
	WeightNuanceShift   float64 // γ (default: 0.20)
	WeightTurnLatency   float64 // δ (default: 0.10)
}

// DefaultAnalysisConfig returns calibrated hyperparameters based on Section 4.3.
func DefaultAnalysisConfig() AnalysisConfig {
	return AnalysisConfig{
		ClientSpeakerName:  "미르",
		PartnerSpeakerName: "🎈",
		MaxTurnGapMinutes:  30.0,
		WeightDyadicDiv:    0.35,
		WeightIntraDiv:     0.35,
		WeightNuanceShift:  0.20,
		WeightTurnLatency:  0.10,
	}
}
