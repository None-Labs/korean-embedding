package dyadic

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// SaveJSONReport exports the dyadic analysis report to a JSON file.
func SaveJSONReport(report *DyadicAnalysisReport, outPath string) error {
	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("failed to create JSON file %s: %w", outPath, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

// PrintConsoleSummary prints a clean summary to the terminal.
func PrintConsoleSummary(report *DyadicAnalysisReport) {
	fmt.Println("================================================================================")
	fmt.Println("   1대1 대화(Dyadic Interaction) 기반 문단 단위 시계열 감정기폭 분석 보고서")
	fmt.Println("================================================================================")
	fmt.Printf("• 분석 기간         : %s ~ %s (총 %.1f일)\n",
		report.StartDate.Format("2006-01-02"), report.EndDate.Format("2006-01-02"), report.DurationDays)
	fmt.Printf("• 전체 원천 메시지  : %d 건\n", report.TotalRawMessages)
	fmt.Printf("• 분절된 문단(턴)   : 총 %d 개 (내담자[미르]: %d, 상대방[🎈]: %d)\n",
		report.TotalParagraphTurns, report.ClientParagraphTurns, report.PartnerParagraphTurns)
	fmt.Printf("• 1대1 상호작용 쌍 : %d 개 (자극-반응 문단 페어)\n", report.TotalDyadicPairs)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                     감정기폭 지수 (EVI) 종합 통계")
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("• 평균 EVI (Mean)   : %.4f\n", report.MeanEVI)
	fmt.Printf("• 중위 EVI (Median) : %.4f\n", report.MedianEVI)
	fmt.Printf("• 최대 EVI (Max)    : %.4f\n", report.MaxEVI)
	fmt.Printf("• 표준편차 (StdDev) : %.4f\n", report.EVIStandardDev)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                 4대 시계열 기폭 동역학 패턴 (Dynamics) 분포")
	fmt.Println("--------------------------------------------------------------------------------")
	for _, p := range []DynamicPattern{PatternExplosion, PatternResonance, PatternFreezing, PatternOscillation, PatternBaseline} {
		cnt := report.PatternCounts[p]
		pct := 0.0
		if report.TotalDyadicPairs > 0 {
			pct = float64(cnt) / float64(report.TotalDyadicPairs) * 100.0
		}
		fmt.Printf("• %-35s : %4d 건 (%5.1f%%)\n", p.KoreanName(), cnt, pct)
	}
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                   월별 감정기폭 추이 및 패턴 빈도 (Monthly)")
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("%-8s | %-6s | %-8s | %-8s | %-6s | %-6s | %-6s | %-6s\n",
		"년-월", "턴 수", "평균 EVI", "최대 EVI", "폭발", "동결", "공명", "진동")
	fmt.Println("--------------------------------------------------------------------------------")
	for _, m := range report.MonthlyTrends {
		fmt.Printf("%-8s | %6d | %8.4f | %8.4f | %6d | %6d | %6d | %6d\n",
			m.YearMonth, m.PairCount, m.MeanEVI, m.MaxEVI, m.Explosions, m.Freezings, m.Resonances, m.Oscillations)
	}
	fmt.Println("================================================================================")
	fmt.Println("       순수 언어 대화 상호작용 감정기폭 스파이크 (Top 5 Pure Text Peaks)")
	fmt.Println("================================================================================")
	topLimit := 5
	if len(report.TopTextPeaks) < topLimit {
		topLimit = len(report.TopTextPeaks)
	}
	for i := 0; i < topLimit; i++ {
		peak := report.TopTextPeaks[i]
		fmt.Printf("[%d위] EVI: %.4f | 패턴: %s | 일시: %s\n",
			peak.Rank, peak.EVI, peak.Pattern.KoreanName(), peak.Timestamp.Format("2006-01-02 15:04"))
		fmt.Printf("  • 상대방(자극): %s\n", peak.PartnerPrompt)
		fmt.Printf("  • 내담자(반응): %s\n", peak.ClientReaction)
		fmt.Printf("  • 메트릭스: 턴지연=%.0f초 | 자극괴리=%.2f | 내부변위=%.2f | 극성=%.2f | 비꼼=%.2f\n",
			peak.TurnLatencySec, peak.DyadicDiv, peak.IntraDiv, peak.Polarity, peak.Sarcasm)
		fmt.Printf("  • 임상 주해: %s\n\n", peak.ClinicalNote)
	}
	fmt.Println("================================================================================")
}

// GenerateMarkdownReport outputs an in-depth clinical analysis markdown document.
func GenerateMarkdownReport(report *DyadicAnalysisReport, outPath string) error {
	var sb strings.Builder

	sb.WriteString("# 카카오톡 1대1 대화 기반 문단 단위 시계열 감정기폭(Dyadic EVI) 정밀 분석 보고서\n\n")
	sb.WriteString("> **분석 대상 로그**: `/home/limmireu1214/_projects/KakaoTalkChats.txt`  \n")
	sb.WriteString(fmt.Sprintf("> **분석 기간**: %s ~ %s (총 %.1f일)  \n",
		report.StartDate.Format("2006년 01월 02일"), report.EndDate.Format("2006년 01월 02일"), report.DurationDays))
	sb.WriteString("> **참여 화자**: 상대방(`🎈`) ↔ 내담자(`미르`)  \n")
	sb.WriteString("> **분석 프레임워크**: `korean-embedding` 3024차원 시맨틱 공간 및 Section 4.3 문단 기폭 역학 수식  \n\n")

	sb.WriteString("## 1. 데이터 집계 및 파싱 메트릭스\n\n")
	sb.WriteString("| 항목 | 수치 | 비고 |\n")
	sb.WriteString("| :--- | :--- | :--- |\n")
	sb.WriteString(fmt.Sprintf("| **전체 원천 카카오톡 메시지** | %d 건 | 미디어/텍스트 전체 라인 |\n", report.TotalRawMessages))
	sb.WriteString(fmt.Sprintf("| **분절된 총 문단 턴 (Paragraph)** | %d 개 | 연속 발화 병합(30분 갭 분절) |\n", report.TotalParagraphTurns))
	sb.WriteString(fmt.Sprintf("| • 내담자(`미르`) 문단 수 | %d 개 (%.1f%%) | 반응 및 자가 정서 표현 |\n",
		report.ClientParagraphTurns, float64(report.ClientParagraphTurns)/float64(report.TotalParagraphTurns)*100.0))
	sb.WriteString(fmt.Sprintf("| • 상대방(`🎈`) 문단 수 | %d 개 (%.1f%%) | 자극 및 환경 피드백 |\n",
		report.PartnerParagraphTurns, float64(report.PartnerParagraphTurns)/float64(report.TotalParagraphTurns)*100.0))
	sb.WriteString(fmt.Sprintf("| **1대1 상호작용 쌍 (Dyadic Pairs)** | %d 개 | 자극 문단($P_C$) ➔ 반응 문단($P_K$) |\n\n", report.TotalDyadicPairs))

	sb.WriteString("## 2. 감정기폭 지수 ($\text{EVI}$) 통계 및 동역학 분포\n\n")
	sb.WriteString("### 2.1 EVI 지표 요약\n")
	sb.WriteString(fmt.Sprintf("- **평균 EVI (Mean)**: `%.4f`\n", report.MeanEVI))
	sb.WriteString(fmt.Sprintf("- **중위 EVI (Median)**: `%.4f`\n", report.MedianEVI))
	sb.WriteString(fmt.Sprintf("- **최대 EVI (Max Peak)**: `%.4f`\n", report.MaxEVI))
	sb.WriteString(fmt.Sprintf("- **표준편차 (Std Dev)**: `%.4f`\n\n", report.EVIStandardDev))

	sb.WriteString("### 2.2 4대 시계열 기폭 동역학 패턴 (Dynamics Patterns) 분포\n\n")
	sb.WriteString("| 패턴 코드 | 패턴 명칭 | 발생 빈도 | 점유율 (%) | 핵심 임상 기제 |\n")
	sb.WriteString("| :--- | :--- | :---: | :---: | :--- |\n")
	for _, p := range []DynamicPattern{PatternExplosion, PatternResonance, PatternFreezing, PatternOscillation, PatternBaseline} {
		cnt := report.PatternCounts[p]
		pct := 0.0
		if report.TotalDyadicPairs > 0 {
			pct = float64(cnt) / float64(report.TotalDyadicPairs) * 100.0
		}
		desc := ""
		switch p {
		case PatternExplosion:
			desc = "즉각적 응답(<120s), 자극-반응 불일치, 격앙/부정 어조 및 감정 표출"
		case PatternResonance:
			desc = "높은 사회적 라포르, 자극 수용 안정성, 정서적 완화 및 웃음"
		case PatternFreezing:
			desc = "장시간 침묵(>30m), 극단적 단답, 방어적 감정 차단"
		case PatternOscillation:
			desc = "감정 극성 급변 또는 냉소/비꼼, 웃음-고통의 양가적 갈등"
		case PatternBaseline:
			desc = "통상적인 일상 정보 교류 및 친밀 대화"
		}
		sb.WriteString(fmt.Sprintf("| `%s` | %s | **%d** 건 | **%.1f%%** | %s |\n", p, p.KoreanName(), cnt, pct, desc))
	}
	sb.WriteString("\n")

	sb.WriteString("## 3. 월별 종단 시계열 감정기폭 및 동역학 추이\n\n")
	sb.WriteString("| 년-월 (Period) | 턴 페어 수 | 평균 EVI | 최대 EVI | 폭발 (Explosion) | 동결 (Freezing) | 공명 (Resonance) | 진동 (Oscillation) |\n")
	sb.WriteString("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
	for _, m := range report.MonthlyTrends {
		sb.WriteString(fmt.Sprintf("| **%s** | %d | `%.4f` | `%.4f` | %d | %d | %d | %d |\n",
			m.YearMonth, m.PairCount, m.MeanEVI, m.MaxEVI, m.Explosions, m.Freezings, m.Resonances, m.Oscillations))
	}
	sb.WriteString("\n")

	sb.WriteString("## 4. 순수 언어 대화 상호작용 감정기폭 스파이크 (Top Pure Text-to-Text Peaks)\n\n")
	sb.WriteString("> 미디어/사진 전송을 배제하고, 상대방과 내담자가 실제로 나눈 언어 대화 턴 간의 가장 극심한 감정 기폭 및 격앙 반응 사건들을 분석합니다.\n\n")
	for i, peak := range report.TopTextPeaks {
		sb.WriteString(fmt.Sprintf("### [%d위] EVI: `%.4f` — %s\n\n", i+1, peak.EVI, peak.Pattern.KoreanName()))
		sb.WriteString(fmt.Sprintf("- **발생 일시**: %s (턴 지연: `%.0f초`)\n", peak.Timestamp.Format("2006년 01월 02일 15시 04분"), peak.TurnLatencySec))
		sb.WriteString(fmt.Sprintf("- **상호작용 자극 괴리도 ($D_{\\text{dyadic}}$)**: `%.4f` | **내부 정서 변위 ($D_{\\text{intra}}$)**: `%.4f`\n", peak.DyadicDiv, peak.IntraDiv))
		sb.WriteString(fmt.Sprintf("- **태도 극성 (Polarity)**: `%.2f` | **비꼼/냉소 점수 (Sarcasm)**: `%.2f`\n\n", peak.Polarity, peak.Sarcasm))
		sb.WriteString("```text\n")
		sb.WriteString(fmt.Sprintf("[상대방 🎈 (자극)]:\n%s\n\n", peak.PartnerPrompt))
		sb.WriteString(fmt.Sprintf("[내담자 미르 (반응)]:\n%s\n", peak.ClientReaction))
		sb.WriteString("```\n")
		sb.WriteString(fmt.Sprintf("> **임상 역학 주해**: %s\n\n", peak.ClinicalNote))
		sb.WriteString("---\n\n")
	}

	return os.WriteFile(outPath, []byte(sb.String()), 0644)
}
