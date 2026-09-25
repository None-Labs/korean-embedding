package embedder

import (
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"

	"gitlab.com/none-labs/korean-tokenizer/pkg/bpe"
	"gitlab.com/none-labs/korean-tokenizer/pkg/morphology"
	"gitlab.com/none-labs/korean-tokenizer/pkg/syntax"
	"gitlab.com/none-labs/korean-tokenizer/pkg/transformer"
)

// Regex patterns for mask normalization and script cleaning
var (
	reNameMask   = regexp.MustCompile(`(?:#@이름#|#이름#|#이름|<이름>)`)
	reEmailMask  = regexp.MustCompile(`(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|<이메일>)`)
	reURLMask    = regexp.MustCompile(`(?:https?://[^\s]+|<링크>)`)
	reCodeBlock  = regexp.MustCompile("(?s)```[a-zA-Z]*\n?.*?```")
	reInlineCode = regexp.MustCompile("`[^`]+`")
	reHTMLTag    = regexp.MustCompile(`<[^>]+>`)
)

var sejongTagIndices = map[morphology.POSTag]int{
	morphology.TagNNG: 0,
	morphology.TagNNP: 1,
	morphology.TagNNB: 2,
	morphology.TagNP:  3,
	morphology.TagNR:  4,
	morphology.TagVV:  5,
	morphology.TagVA:  6,
	morphology.TagVX:  7,
	morphology.TagVCP: 8,
	morphology.TagVCN: 9,
	morphology.TagMM:  10,
	morphology.TagMAG: 11,
	morphology.TagMAJ: 12,
	morphology.TagIC:  13,
	morphology.TagJKS: 14,
	morphology.TagJKC: 15,
	morphology.TagJKG: 16,
	morphology.TagJKO: 17,
	morphology.TagJKB: 18,
	morphology.TagJKV: 19,
	morphology.TagJKQ: 20,
	morphology.TagJX:  21,
	morphology.TagJC:  22,
	morphology.TagEP:  23,
	morphology.TagEF:  24,
	morphology.TagEC:  25,
	morphology.TagETN: 26,
	morphology.TagETM: 27,
	morphology.TagXPN: 28,
	morphology.TagXSN: 29,
	morphology.TagXSV: 30,
	morphology.TagXSA: 31,
	morphology.TagXR:  32,
	morphology.TagSF:  33,
	morphology.TagSP:  34,
	morphology.TagSS:  35,
	morphology.TagSE:  36,
	morphology.TagSO:  37,
	morphology.TagSW:  38,
	morphology.TagSL:  39,
	morphology.TagSH:  40,
	morphology.TagSN:  41,
}

// EmbedderConfig holds configuration for the 3024-d Embedding Engine.
type EmbedderConfig struct {
	HiddenDim        int     `json:"hidden_dim"`
	NumHeads         int     `json:"num_heads"`
	NumLayers        int     `json:"num_layers"`
	VocabSize        int     `json:"vocab_size"`
	DropoutRate      float32 `json:"dropout_rate"`
	BPETokenizerPath string  `json:"bpe_tokenizer_path,omitempty"`
}

// DefaultEmbedderConfig returns optimized configuration for CPU inference and training.
func DefaultEmbedderConfig() EmbedderConfig {
	return EmbedderConfig{
		HiddenDim:   128,
		NumHeads:    4,
		NumLayers:   2,
		VocabSize:   32000,
		DropoutRate: 0.1,
	}
}

// Embedder orchestrates the end-to-end 3024-d Korean Utterance Semantic Embedding pipeline.
type Embedder struct {
	cfg          EmbedderConfig
	segmenter    *morphology.Segmenter
	toneAnalyzer *morphology.ToneAnalyzer
	syntaxParser *syntax.DependencyParser
	bpeModel     *bpe.Model
	transformer  *transformer.HierarchicalTransformer
	pooler       *AttentionPooling
	projection   *SubspaceProjection
	slicer       *MRLSlicer
	mu           sync.RWMutex
}

// NewEmbedder creates an initialized 3024-d Korean Utterance Embedding Engine.
func NewEmbedder(cfg EmbedderConfig) (*Embedder, error) {
	if cfg.HiddenDim <= 0 {
		cfg.HiddenDim = 128
	}
	if cfg.NumHeads <= 0 {
		cfg.NumHeads = 4
	}
	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 2
	}
	if cfg.VocabSize <= 0 {
		cfg.VocabSize = 32000
	}

	tfCfg := transformer.Config{
		VocabSize:       cfg.VocabSize,
		NumPOSTags:      46,
		NumDepRels:      8,
		HiddenDim:       cfg.HiddenDim,
		NumHeads:        cfg.NumHeads,
		NumLayers:       cfg.NumLayers,
		DiscourseLayers: 1,
		LearningRate:    0.001,
		WeightDecay:     0.01,
	}

	emb := &Embedder{
		cfg:          cfg,
		segmenter:    morphology.DefaultSegmenter(),
		toneAnalyzer: morphology.DefaultToneAnalyzer(),
		syntaxParser: syntax.NewParser(),
		transformer:  transformer.NewHierarchicalTransformer(tfCfg),
		pooler:       NewAttentionPooling(cfg.HiddenDim),
		projection:   NewSubspaceProjection(cfg.HiddenDim),
		slicer:       NewMRLSlicer(),
	}

	if cfg.BPETokenizerPath != "" {
		if _, err := os.Stat(cfg.BPETokenizerPath); err == nil {
			if m, err := bpe.Load(cfg.BPETokenizerPath); err == nil {
				emb.bpeModel = m
			}
		}
	}

	return emb, nil
}

// Transformer returns the underlying transformer model.
func (e *Embedder) Transformer() *transformer.HierarchicalTransformer {
	return e.transformer
}

// Projection returns the underlying subspace projection layer.
func (e *Embedder) Projection() *SubspaceProjection {
	return e.projection
}

// NormalizeText cleans non-Korean code/markup while strictly standardizing special entity masks.
func (e *Embedder) NormalizeText(raw string) (string, []string) {
	var maskedTags []string

	// 1. Remove Markdown/Python code blocks and HTML tags
	text := reCodeBlock.ReplaceAllString(raw, " ")
	text = reInlineCode.ReplaceAllString(text, " ")

	// 2. Normalize personal masks
	if reNameMask.MatchString(text) {
		maskedTags = append(maskedTags, "<이름>")
		text = reNameMask.ReplaceAllString(text, " <이름> ")
	}

	// 3. Normalize email masks
	if reEmailMask.MatchString(text) {
		maskedTags = append(maskedTags, "<이메일>")
		text = reEmailMask.ReplaceAllString(text, " <이메일> ")
	}

	// 4. Normalize URL masks
	if reURLMask.MatchString(text) {
		maskedTags = append(maskedTags, "<링크>")
		text = reURLMask.ReplaceAllString(text, " <링크> ")
	}

	// 5. Clean remaining HTML tags (keep standard masking tags intact)
	text = reHTMLTag.ReplaceAllStringFunc(text, func(m string) string {
		if m == "<이름>" || m == "<이메일>" || m == "<링크>" {
			return m
		}
		return " "
	})

	// 6. Normalize multiple whitespaces
	fields := strings.Fields(text)
	normalized := strings.Join(fields, " ")
	return normalized, maskedTags
}

// InferSocialStance analyzes formality, politeness, and vocabulary to classify interpersonal framing.
func (e *Embedder) InferSocialStance(meta *UtteranceMetadata) SocialStance {
	text := meta.CleanedText

	// 1. Formal & Organizational:
	// Business reporting terms, high formality score (> 0.65), Hasipsio-che
	businessTerms := []string{
		"보고드립니다", "검토", "진행하겠습니다", "안건", "공지", "협조", "요청드립니다",
		"회의", "결재", "부서", "팀장님", "과장님", "대표님", "담당자", "안녕하십니까", "감사합니다",
	}
	hasBusinessTerm := false
	for _, term := range businessTerms {
		if strings.Contains(text, term) {
			hasBusinessTerm = true
			break
		}
	}
	if hasBusinessTerm || meta.SpeechLevel == morphology.SpeechLevelHasipsio || meta.Formality > 0.65 {
		return SocialStanceFormal
	}

	// 2. Informal Romantic/Couple:
	// High aegyo score, affectionate pet names, romantic vocabulary
	coupleTerms := []string{
		"자기야", "여보", "사랑해", "보고싶어", "쪽", "하트", "남친", "여친", "내사랑",
		"애기야", "뽀뽀", "데이트", "칼퇴하고 봐", "이따 봐요",
	}
	for _, term := range coupleTerms {
		if strings.Contains(text, term) {
			return SocialStanceInformalCouple
		}
	}
	if meta.AegyoScore >= 0.7 || meta.SpeechLevel == morphology.SpeechLevelAegyo {
		return SocialStanceInformalCouple
	}

	// 3. Informal Family:
	// Family kinship titles
	familyTerms := []string{
		"엄마", "아빠", "어머니", "아버지", "할머니", "할아버지", "삼촌", "이모", "고모",
		"누나", "형", "오빠", "동생", "진지", "식사하셨어요", "다녀왔습니다",
	}
	for _, term := range familyTerms {
		if strings.Contains(text, term) {
			return SocialStanceInformalFamily
		}
	}

	// 4. Informal Friend (Default casual talk):
	// Hae-che, youth slang, casual particles
	return SocialStanceInformalFriend
}

// InferAgeGroup analyzes generational markers, contractions, and stylistic traits.
func (e *Embedder) InferAgeGroup(meta *UtteranceMetadata) AgeGroup {
	text := meta.CleanedText

	// 1. Youth (10-20대):
	// Consonant contractions, Gen-Z / Alpha slang
	youthMarkers := []string{
		"ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅇㅇ", "ㄱㅅ", "ㄷㄷ", "ㅈㅅ", "ㄹㅇ", "갓생", "점메추",
		"스불재", "알잘딱", "킹받네", "오운완", "뇌절", "어쩔", "존맛", "개웃겨", "레알",
		"대박", "미쳤다", "쩐다", "쌉가능", "인정",
	}
	for _, marker := range youthMarkers {
		if strings.Contains(text, marker) {
			return AgeGroupYouth
		}
	}
	if meta.SpeechLevel == morphology.SpeechLevelAegyo || meta.AegyoScore > 0.6 {
		return AgeGroupYouth
	}

	// 2. Senior (50대+):
	// Traditional honorifics, archaic speech levels, senior vocabulary
	seniorMarkers := []string{
		"연세", "춘추", "진지", "댁", "어르신", "옥체", "강녕", "평안",
		"~하오", "~하게", "~구려", "~옵소서", "귀하", "혜량", "만수무강",
	}
	for _, marker := range seniorMarkers {
		if strings.Contains(text, marker) {
			return AgeGroupSenior
		}
	}

	// 3. Middle (30-40대):
	// Standard workplace / practical style
	return AgeGroupMiddle
}

// Embed encodes a single Korean utterance into the 3024-dimensional continuous semantic space.
func (e *Embedder) Embed(text string) (*EmbeddingResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	cleaned, maskedTags := e.NormalizeText(text)
	if strings.TrimSpace(cleaned) == "" {
		cleaned = text
	}

	// 1. Morphological and Tone Analysis
	docTone := e.toneAnalyzer.AnalyzeText(cleaned)
	morphemes := e.segmenter.Analyze(cleaned)

	// Primary sentence tone
	var sentTone morphology.SentenceTone
	if len(docTone.Sentences) > 0 {
		sentTone = docTone.Sentences[0]
	} else {
		sentTone = morphology.SentenceTone{
			SpeechLevel: morphology.SpeechLevelNeutral,
			Modality:    morphology.ModalityDeclarative,
		}
	}

	// 2. Syntactic Dependency Parsing
	var syntaxNodes []syntax.SyntaxNode
	structure := e.segmenter.AnalyzeStructure(cleaned)
	if len(structure) > 0 && len(structure[0].Sentences) > 0 {
		syntaxNodes = e.syntaxParser.ParseSentence(structure[0].Sentences[0])
	}

	// 3. Determine Speech Act from tone modality and endings
	speechAct, actProb := determineSpeechAct(sentTone, cleaned)

	// 4. Construct UtteranceMetadata
	meta := UtteranceMetadata{
		OriginalText:     text,
		CleanedText:      cleaned,
		MaskedTags:       maskedTags,
		SpeechAct:        speechAct,
		SpeechActDesc:    string(speechAct),
		SpeechActProb:    actProb,
		SpeechLevel:      sentTone.SpeechLevel,
		SpeechLevelDesc:  sentTone.SpeechLevel.Description(),
		Politeness:       sentTone.Politeness,
		Formality:        sentTone.Formality,
		AegyoScore:       sentTone.AegyoScore,
		SarcasmDetected:  sentTone.SarcasmDetected,
		SarcasmScore:     sentTone.SarcasmScore,
		AttitudePolarity: sentTone.AttitudePolarity,
		TrueIntent:       sentTone.TrueIntent,
		KeyEnding:        sentTone.KeyEnding,
		Morphemes:        morphemes,
		SyntaxNodes:      syntaxNodes,
	}

	// Infer Social Stance and Age Group
	meta.SocialStance = e.InferSocialStance(&meta)
	meta.SocialStanceDesc = meta.SocialStance.Description()
	meta.AgeGroup = e.InferAgeGroup(&meta)
	meta.AgeGroupDesc = meta.AgeGroup.Description()

	// 5. Tokenize into subwords and construct SentenceInput
	tokenIDs, posIDs, depIDs, isRootList, posTags := e.buildTokenFeatures(cleaned, morphemes, syntaxNodes)

	// 6. Transformer Level 3: Feature Fusion & Intra-Sentence Attention
	d := e.cfg.HiddenDim
	seqLen := len(tokenIDs)
	var hSent []float32

	if seqLen > 0 {
		// Feature fusion
		X := make([]float32, seqLen*d)
		for i := 0; i < seqLen; i++ {
			offset := i * d
			tok := tokenIDs[i] % e.cfg.VocabSize
			pos := posIDs[i] % 46
			dep := depIDs[i] % 8

			tokOffset := tok * d
			posOffset := pos * d
			depOffset := dep * d

			for j := 0; j < d; j++ {
				X[offset+j] = e.transformer.WSubword[tokOffset+j] + e.transformer.WPOS[posOffset+j] + e.transformer.WDep[depOffset+j]
			}
		}

		// Intra-Sentence Layers with RoPE
		for _, layer := range e.transformer.IntraLayers {
			X = layer.Forward(X, seqLen, true)
		}

		// 7. Attention-Weighted Pooling with Korean morphological & syntactic root emphasis
		mask := make([]float32, seqLen)
		for i := range mask {
			mask[i] = 1.0
		}
		hSent = e.pooler.Pool(X, seqLen, mask, posTags, isRootList)
	} else {
		hSent = make([]float32, d)
	}

	// 8. Level 4: 3024-d Multi-Aspect Semantic Projection
	z3024 := e.projection.Project(hSent, &meta)

	// 9. Extract Subspaces (L2 normalized)
	subspaces := make(map[SubspaceType][]float32, 4)
	for _, st := range []SubspaceType{SubspaceLiteral, SubspaceIntent, SubspaceSocial, SubspaceNuance} {
		if sub, err := ExtractSubspace(z3024, st, true); err == nil {
			subspaces[st] = sub
		}
	}

	// 10. Extract MRL Slices (512, 1024, 1536, 3024)
	mrlSlices, _ := e.slicer.SliceAll(z3024)

	return &EmbeddingResult{
		Dimension: TotalDim,
		Vector:    z3024,
		Metadata:  meta,
		Subspaces: subspaces,
		MRLSlices: mrlSlices,
	}, nil
}

// EmbedBatch processes multiple utterances in batch.
func (e *Embedder) EmbedBatch(texts []string) ([]*EmbeddingResult, error) {
	results := make([]*EmbeddingResult, len(texts))
	for i, t := range texts {
		res, err := e.Embed(t)
		if err != nil {
			return nil, fmt.Errorf("error embedding text at index %d: %w", i, err)
		}
		results[i] = res
	}
	return results, nil
}

// Compare computes overall cosine similarity and multi-aspect subspace breakdown between two utterances.
func (e *Embedder) Compare(text1, text2 string) (*SimilarityResult, error) {
	res1, err := e.Embed(text1)
	if err != nil {
		return nil, err
	}
	res2, err := e.Embed(text2)
	if err != nil {
		return nil, err
	}

	cosSim := CosineSimilarity(res1.Vector, res2.Vector)
	litSim := CosineSimilarity(res1.Subspaces[SubspaceLiteral], res2.Subspaces[SubspaceLiteral])
	intSim := CosineSimilarity(res1.Subspaces[SubspaceIntent], res2.Subspaces[SubspaceIntent])
	socSim := CosineSimilarity(res1.Subspaces[SubspaceSocial], res2.Subspaces[SubspaceSocial])
	nuaSim := CosineSimilarity(res1.Subspaces[SubspaceNuance], res2.Subspaces[SubspaceNuance])

	mrlComp, _ := e.slicer.CompareMRL(res1.Vector, res2.Vector)

	ironyDiv := float32(math.Abs(float64(res1.Metadata.SarcasmScore - res2.Metadata.SarcasmScore)))

	return &SimilarityResult{
		Text1:             text1,
		Text2:             text2,
		CosineSimilarity:  cosSim,
		LiteralSimilarity: litSim,
		IntentSimilarity:  intSim,
		SocialSimilarity:  socSim,
		NuanceSimilarity:  nuaSim,
		MRLComparison:     mrlComp,
		IntentMatch:       res1.Metadata.SpeechAct == res2.Metadata.SpeechAct,
		SocialStanceMatch: res1.Metadata.SocialStance == res2.Metadata.SocialStance,
		IronyDivergence:   ironyDiv,
	}, nil
}

// buildTokenFeatures generates token IDs, POS IDs, Dep relation IDs, and root node flags.
func (e *Embedder) buildTokenFeatures(
	text string,
	morphemes []morphology.Morpheme,
	syntaxNodes []syntax.SyntaxNode,
) ([]int, []int, []int, []bool, []morphology.POSTag) {
	if len(morphemes) == 0 {
		runes := []rune(text)
		tokens := make([]int, len(runes))
		for i, r := range runes {
			tokens[i] = int(r) % e.cfg.VocabSize
		}
		return tokens, make([]int, len(tokens)), make([]int, len(tokens)), make([]bool, len(tokens)), make([]morphology.POSTag, len(tokens))
	}

	tokens := make([]int, len(morphemes))
	posIDs := make([]int, len(morphemes))
	depIDs := make([]int, len(morphemes))
	isRoot := make([]bool, len(morphemes))
	posTags := make([]morphology.POSTag, len(morphemes))

	// Map syntax head/root
	hasRootNode := false
	var rootWord string
	for _, sn := range syntaxNodes {
		if sn.Head == 0 || sn.DepRel == syntax.DepRelRoot {
			rootWord = sn.Text
			hasRootNode = true
			break
		}
	}

	for i, m := range morphemes {
		posTags[i] = m.Tag
		// Deterministic token hash or BPE ID
		if e.bpeModel != nil {
			ids, _ := e.bpeModel.Encode(m.Surface)
			if len(ids) > 0 {
				tokens[i] = ids[0] % e.cfg.VocabSize
			} else {
				tokens[i] = hashString(m.Surface) % e.cfg.VocabSize
			}
		} else {
			tokens[i] = hashString(m.Surface) % e.cfg.VocabSize
		}

		if idx, ok := sejongTagIndices[m.Tag]; ok {
			posIDs[i] = idx
		} else {
			posIDs[i] = 42 // Unknown
		}

		if hasRootNode && strings.Contains(rootWord, m.Surface) {
			isRoot[i] = true
			depIDs[i] = 0 // Root
		} else if m.Tag.IsParticle() {
			depIDs[i] = 1 // Dependent
		} else {
			depIDs[i] = 2 // Content
		}
	}

	return tokens, posIDs, depIDs, isRoot, posTags
}

func hashString(s string) int {
	var h uint32 = 2166136261
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return int(h & 0x7FFFFFFF)
}

func determineSpeechAct(tone morphology.SentenceTone, text string) (transformer.SpeechAct, float32) {
	// 1. Apology (High priority)
	if strings.Contains(text, "죄송") || strings.Contains(text, "사과") || strings.Contains(text, "미안") ||
		strings.Contains(text, "실례") || strings.Contains(text, "유감") {
		return transformer.SpeechActApology, 0.98
	}

	// 2. Acceptance (Agreement / Confirmation)
	if strings.Contains(text, "알겠습니다") || strings.Contains(text, "동의") || strings.Contains(text, "수용") ||
		strings.Contains(text, "좋습니다") || strings.Contains(text, "그러죠") || strings.HasPrefix(text, "네") {
		return transformer.SpeechActAcceptance, 0.92
	}

	// 3. Question (Inquiry / Request for info)
	if tone.Modality == morphology.ModalityInterrogative || strings.Contains(text, "?") {
		return transformer.SpeechActQuestion, 0.95
	}

	// 4. Imperative / Request (Action directive / favor)
	if tone.Modality == morphology.ModalityImperative || strings.Contains(text, "주세요") ||
		strings.Contains(text, "부탁") || strings.Contains(text, "바랍니다") || strings.Contains(text, "해줘") ||
		strings.Contains(text, "해주십시오") || strings.Contains(text, "하십시오") {
		return transformer.SpeechActRequest, 0.93
	}

	// 5. Threat / Warning
	if strings.Contains(text, "경고") || strings.Contains(text, "가만두지 않") || strings.Contains(text, "조심해") {
		return transformer.SpeechActThreat, 0.94
	}

	// 6. Promise (Future commitment)
	if strings.Contains(text, "약속") || strings.Contains(text, "겠습니다") || strings.Contains(text, "꼭 하") ||
		strings.Contains(text, "보장합니다") || strings.Contains(text, "할게요") || strings.Contains(text, "할게") {
		return transformer.SpeechActPromise, 0.89
	}

	// 7. Exclamation (Emotion burst)
	if tone.Modality == morphology.ModalityExclamatory || strings.Contains(text, "!") || strings.Contains(text, "와아") {
		return transformer.SpeechActExclamation, 0.90
	}

	// 8. Default Statement
	return transformer.SpeechActStatement, 0.85
}
