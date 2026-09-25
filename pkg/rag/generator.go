package rag

import (
	"fmt"
	"strings"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// LLMClient defines the interface for generating answers given an augmented RAG prompt.
type LLMClient interface {
	Generate(prompt string) (string, error)
}

// PromptBuilder constructs Korean-optimized grounding prompts incorporating speaker context.
type PromptBuilder struct{}

// NewPromptBuilder creates a new prompt builder.
func NewPromptBuilder() *PromptBuilder {
	return &PromptBuilder{}
}

// BuildContext compiles retrieved search results into a structured grounding context string.
func (pb *PromptBuilder) BuildContext(results []SearchResult) string {
	if len(results) == 0 {
		return "참고할 관련 문서가 검색되지 않았습니다."
	}

	var sb strings.Builder
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("[참고 자료 %d] 출처: %s (일치율: %.1f%%, 관계: %s)\n",
			i+1, r.Chunk.DocTitle, r.Score*100, r.Chunk.Metadata.SocialStance))
		sb.WriteString(fmt.Sprintf("%s\n\n", strings.TrimSpace(r.Chunk.Text)))
	}
	return strings.TrimSpace(sb.String())
}

// BuildAugmentedPrompt crafts the complete LLM instruction prompt.
func (pb *PromptBuilder) BuildAugmentedPrompt(
	query string,
	meta *embedder.UtteranceMetadata,
	results []SearchResult,
) (string, string) {
	contextText := pb.BuildContext(results)

	var sb strings.Builder
	sb.WriteString("당신은 질문자의 의도와 대화 맥락을 정확하게 파악하는 전문 한국어 어시스턴트입니다.\n\n")

	// Inject speaker metadata guidance
	if meta != nil {
		sb.WriteString("### [사용자 발화 특성 및 태도 분석]\n")
		sb.WriteString(fmt.Sprintf("- 질문 의도(화행): %s\n", meta.SpeechAct))
		sb.WriteString(fmt.Sprintf("- 발화 격식/어조: %s (%s, 존칭점수: %.2f)\n",
			meta.SpeechLevel, meta.SpeechLevelDesc, meta.Politeness))
		sb.WriteString(fmt.Sprintf("- 대화 관계 분류: %s\n\n", meta.SocialStanceDesc))
	}

	sb.WriteString("### [검색된 신뢰 참고 문서]\n")
	sb.WriteString(contextText)
	sb.WriteString("\n\n")

	sb.WriteString("### [답변 지침 및 가이드라인]\n")
	sb.WriteString("1. 반드시 제시된 [검색된 신뢰 참고 문서]의 사실에만 기반하여 정직하고 정확하게 답변하십시오.\n")
	sb.WriteString("2. 참고 문서에 명시되지 않은 정보는 절대 허위 날조(할루시네이션)하지 말고 사실대로 알 수 없다고 안내하십시오.\n")
	sb.WriteString("3. 사용자의 발화 어조와 관계에 맞추어 정중하고 신뢰할 수 있는 한국어로 답변을 작성하십시오.\n")
	sb.WriteString("4. 답변 시 근거가 된 출처 자료의 제목을 자연스럽게 언급하십시오.\n\n")

	sb.WriteString(fmt.Sprintf("사용자 질문: %s\n답변:\n", query))

	return sb.String(), contextText
}

// PureGoExtractiveGenerator generates grounded extractive responses without external API calls.
type PureGoExtractiveGenerator struct{}

// NewPureGoExtractiveGenerator creates a zero-cost local generator.
func NewPureGoExtractiveGenerator() *PureGoExtractiveGenerator {
	return &PureGoExtractiveGenerator{}
}

// GenerateSynthesizedAnswer generates an extractive summary answer from top retrieved chunks.
func (g *PureGoExtractiveGenerator) GenerateSynthesizedAnswer(
	query string,
	results []SearchResult,
) string {
	if len(results) == 0 {
		return "죄송합니다. 문의하신 내용과 관련된 정보를 사내 지식 기반에서 찾을 수 없습니다."
	}

	top := results[0]
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("'%s' 문서에 명시된 내용에 따라 안내해 드립니다:\n\n", top.Chunk.DocTitle))
	sb.WriteString(fmt.Sprintf("%s\n\n", strings.TrimSpace(top.Chunk.Text)))

	if len(results) > 1 {
		sb.WriteString(fmt.Sprintf("(추가 관련 참고: '%s')", results[1].Chunk.DocTitle))
	}

	return sb.String()
}
