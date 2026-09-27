package dyadic

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

// AnalyzeDyadicChats processes paragraph turns, computes 3024-d embeddings, dyadic metrics, and generates a comprehensive report.
func AnalyzeDyadicChats(paragraphs []*ParagraphTurn, emb *embedder.Embedder, cfg AnalysisConfig) (*DyadicAnalysisReport, error) {
	if len(paragraphs) == 0 {
		return nil, fmt.Errorf("no paragraphs provided for dyadic analysis")
	}

	report := &DyadicAnalysisReport{
		TotalParagraphTurns: len(paragraphs),
		PatternCounts:       make(map[DynamicPattern]int),
		StartDate:           paragraphs[0].StartTime,
		EndDate:             paragraphs[len(paragraphs)-1].EndTime,
	}
	report.DurationDays = report.EndDate.Sub(report.StartDate).Hours() / 24.0

	// 1. Compute 3024-d embeddings for each paragraph
	clientParaCount := 0
	partnerParaCount := 0

	for _, p := range paragraphs {
		if p.Speaker == cfg.ClientSpeakerName {
			clientParaCount++
		} else if p.Speaker == cfg.PartnerSpeakerName {
			partnerParaCount++
		}

		embedTargetText := p.CleanedText
		if p.MediaOnly || strings.TrimSpace(embedTargetText) == "" {
			embedTargetText = "[사진/동영상/이모티콘 미디어 전송]"
		}

		res, err := emb.Embed(embedTargetText)
		if err != nil {
			return nil, fmt.Errorf("failed to embed turn %d (%s): %w", p.ID, p.Speaker, err)
		}
		p.Embedding = res
	}

	report.ClientParagraphTurns = clientParaCount
	report.PartnerParagraphTurns = partnerParaCount

	// 2. Identify Dyadic 1-on-1 Interaction Pairs (Partner Stimulus -> Client Response)
	var pairs []DyadicTurnPair
	var prevClientPara *ParagraphTurn
	var lastPartnerPara *ParagraphTurn

	pairIdx := 0

	for _, p := range paragraphs {
		if p.Speaker == cfg.PartnerSpeakerName {
			lastPartnerPara = p
		} else if p.Speaker == cfg.ClientSpeakerName {
			if lastPartnerPara != nil && !p.MediaOnly {
				pairIdx++

				stimulus := lastPartnerPara
				response := p

				// Latency
				latencySec := response.StartTime.Sub(stimulus.EndTime).Seconds()
				if latencySec < 0 {
					latencySec = 0
				}

				// Dyadic Divergence: 1 - CosSim(z_client, z_partner)
				cosDyadic := float64(embedder.CosineSimilarity(response.Embedding.Vector, stimulus.Embedding.Vector))
				dyadicDiv := math.Max(0.0, 1.0-cosDyadic)

				// Intra Divergence: 1 - CosSim(z_client, z_prev_client)
				intraDiv := 0.0
				nuanceShift := 0.0
				polShift := 0.0
				if prevClientPara != nil && prevClientPara.Embedding != nil {
					cosIntra := float64(embedder.CosineSimilarity(response.Embedding.Vector, prevClientPara.Embedding.Vector))
					intraDiv = math.Max(0.0, 1.0-cosIntra)

					polShift = math.Abs(response.Embedding.Metadata.AttitudePolarity - prevClientPara.Embedding.Metadata.AttitudePolarity)
					aegyoShift := math.Abs(response.Embedding.Metadata.AegyoScore - prevClientPara.Embedding.Metadata.AegyoScore)
					nuanceShift = (polShift + aegyoShift + response.Embedding.Metadata.SarcasmScore) / 3.0
				} else {
					intraDiv = dyadicDiv
					nuanceShift = response.Embedding.Metadata.SarcasmScore
				}

				// Rapport Score: Social Subspace similarity
				rapportScore := 0.5
				if subResp, ok := response.Embedding.Subspaces[embedder.SubspaceSocial]; ok {
					if subStim, ok2 := stimulus.Embedding.Subspaces[embedder.SubspaceSocial]; ok2 {
						rapportScore = float64(embedder.CosineSimilarity(subResp, subStim))
					}
				}

				// Turn Latency Weight (Faster response under emotional surge yields higher weight)
				latencyWeight := 1.0 / (1.0 + math.Log(1.0+latencySec))

				// EVI Formulation: Section 4.3
				evi := cfg.WeightDyadicDiv*dyadicDiv +
					cfg.WeightIntraDiv*intraDiv +
					cfg.WeightNuanceShift*nuanceShift +
					cfg.WeightTurnLatency*latencyWeight

				if evi > 1.0 {
					evi = 1.0
				}
				if evi < 0.0 {
					evi = 0.0
				}

				// Dynamic Pattern Classification
				pattern, reason := classifyPattern(dyadicDiv, intraDiv, rapportScore, latencySec, evi, polShift, stimulus, response)

				report.PatternCounts[pattern]++

				pair := DyadicTurnPair{
					PairIndex:         pairIdx,
					ClientTurnID:      response.ID,
					PartnerTurnID:     stimulus.ID,
					Timestamp:         response.StartTime,
					PartnerText:       stimulus.MergedText,
					ClientText:        response.MergedText,
					TurnLatencySec:    latencySec,
					DyadicDivergence:  dyadicDiv,
					IntraDivergence:   intraDiv,
					RapportScore:      rapportScore,
					NuanceDiscrepancy: nuanceShift,
					AttitudePolarity:  response.Embedding.Metadata.AttitudePolarity,
					PolarityShift:     polShift,
					SarcasmScore:      response.Embedding.Metadata.SarcasmScore,
					SpeechAct:         string(response.Embedding.Metadata.SpeechAct),
					SocialStance:      string(response.Embedding.Metadata.SocialStance),
					EVI:               evi,
					IsMediaStimulus:   stimulus.MediaOnly,
					IsMediaResponse:   response.MediaOnly,
					Pattern:           pattern,
					PatternReason:     reason,
				}

				pairs = append(pairs, pair)
			}
			prevClientPara = p
		}
	}

	report.TotalDyadicPairs = len(pairs)
	report.InteractionPairs = pairs

	if len(pairs) == 0 {
		return report, nil
	}

	// 3. EVI Statistics (Mean, Median, Max, StdDev)
	var evis []float64
	sumEVI := 0.0
	maxEVI := 0.0

	for _, pair := range pairs {
		evis = append(evis, pair.EVI)
		sumEVI += pair.EVI
		if pair.EVI > maxEVI {
			maxEVI = pair.EVI
		}
	}

	report.MeanEVI = sumEVI / float64(len(pairs))
	report.MaxEVI = maxEVI

	sort.Float64s(evis)
	if len(evis)%2 == 0 {
		report.MedianEVI = (evis[len(evis)/2-1] + evis[len(evis)/2]) / 2.0
	} else {
		report.MedianEVI = evis[len(evis)/2]
	}

	var varianceSum float64
	for _, v := range evis {
		diff := v - report.MeanEVI
		varianceSum += diff * diff
	}
	report.EVIStandardDev = math.Sqrt(varianceSum / float64(len(evis)))

	// 4. Extract Top Volatile Peaks (Top 20 overall EVI spikes)
	sortedPairs := make([]DyadicTurnPair, len(pairs))
	copy(sortedPairs, pairs)
	sort.Slice(sortedPairs, func(i, j int) bool {
		return sortedPairs[i].EVI > sortedPairs[j].EVI
	})

	topN := 20
	if len(sortedPairs) < topN {
		topN = len(sortedPairs)
	}

	for i := 0; i < topN; i++ {
		p := sortedPairs[i]
		peak := VolatilePeak{
			Rank:           i + 1,
			PairIndex:      p.PairIndex,
			Timestamp:      p.Timestamp,
			EVI:            p.EVI,
			Pattern:        p.Pattern,
			PartnerPrompt:  truncateText(p.PartnerText, 120),
			ClientReaction: truncateText(p.ClientText, 160),
			TurnLatencySec: p.TurnLatencySec,
			DyadicDiv:      p.DyadicDivergence,
			IntraDiv:       p.IntraDivergence,
			NuanceShift:    p.NuanceDiscrepancy,
			Polarity:       p.AttitudePolarity,
			Sarcasm:        p.SarcasmScore,
			IsMediaStim:    p.IsMediaStimulus,
			ClinicalNote:   p.PatternReason,
		}
		report.TopVolatilePeaks = append(report.TopVolatilePeaks, peak)
	}

	// 4-1. Extract Top Pure Text-to-Text Peaks (genuine dialogue emotional spikes)
	var textPairs []DyadicTurnPair
	for _, p := range pairs {
		if !p.IsMediaStimulus && !p.IsMediaResponse &&
			len([]rune(strings.TrimSpace(p.PartnerText))) >= 5 &&
			len([]rune(strings.TrimSpace(p.ClientText))) >= 5 {
			textPairs = append(textPairs, p)
		}
	}

	sort.Slice(textPairs, func(i, j int) bool {
		return textPairs[i].EVI > textPairs[j].EVI
	})

	topTextN := 15
	if len(textPairs) < topTextN {
		topTextN = len(textPairs)
	}

	for i := 0; i < topTextN; i++ {
		p := textPairs[i]
		peak := VolatilePeak{
			Rank:           i + 1,
			PairIndex:      p.PairIndex,
			Timestamp:      p.Timestamp,
			EVI:            p.EVI,
			Pattern:        p.Pattern,
			PartnerPrompt:  truncateText(p.PartnerText, 120),
			ClientReaction: truncateText(p.ClientText, 160),
			TurnLatencySec: p.TurnLatencySec,
			DyadicDiv:      p.DyadicDivergence,
			IntraDiv:       p.IntraDivergence,
			NuanceShift:    p.NuanceDiscrepancy,
			Polarity:       p.AttitudePolarity,
			Sarcasm:        p.SarcasmScore,
			IsMediaStim:    false,
			ClinicalNote:   p.PatternReason,
		}
		report.TopTextPeaks = append(report.TopTextPeaks, peak)
	}

	// 5. Compute Monthly Trends
	monthMap := make(map[string]*MonthlyTrend)
	var monthKeys []string

	for _, p := range pairs {
		ym := p.Timestamp.Format("2006-01")
		t, exists := monthMap[ym]
		if !exists {
			t = &MonthlyTrend{YearMonth: ym}
			monthMap[ym] = t
			monthKeys = append(monthKeys, ym)
		}

		t.PairCount++
		t.MeanEVI += p.EVI
		if p.EVI > t.MaxEVI {
			t.MaxEVI = p.EVI
		}

		switch p.Pattern {
		case PatternExplosion:
			t.Explosions++
		case PatternFreezing:
			t.Freezings++
		case PatternResonance:
			t.Resonances++
		case PatternOscillation:
			t.Oscillations++
		}
	}

	sort.Strings(monthKeys)
	for _, ym := range monthKeys {
		t := monthMap[ym]
		if t.PairCount > 0 {
			t.MeanEVI = t.MeanEVI / float64(t.PairCount)
		}
		report.MonthlyTrends = append(report.MonthlyTrends, *t)
	}

	return report, nil
}

// classifyPattern assigns one of the 4 clinical volatility dynamic patterns with linguistic grounding.
func classifyPattern(dyadicDiv, intraDiv, rapportScore, latencySec, evi, polShift float64, stim, resp *ParagraphTurn) (DynamicPattern, string) {
	clientText := resp.CleanedText
	polarity := resp.Embedding.Metadata.AttitudePolarity
	sarcasm := resp.Embedding.Metadata.SarcasmScore

	isLaughterOnly := (strings.Contains(clientText, "ㅋㅋㅋ") || strings.Contains(clientText, "ㅎㅎ") || strings.Contains(clientText, "ㅋㅎ")) &&
		len([]rune(clientText)) <= 25 && polarity >= 0.0
	isLinkSharing := strings.Contains(clientText, "http") || strings.Contains(clientText, "coupang")

	// 1. Pattern D: Ambivalent Oscillation (양가성 정서 진동)
	// Polarity 급변 or 비꼼/냉소 or (웃음과 부정적 어휘의 양가적 공존)
	hasAmbivalentKeywords := (strings.Contains(clientText, "ㅋㅋ") || strings.Contains(clientText, "ㅎㅎ")) &&
		(strings.Contains(clientText, "짜증") || strings.Contains(clientText, "에휴") || strings.Contains(clientText, "귀찮") || strings.Contains(clientText, "시려") || strings.Contains(clientText, "힘들"))
	if hasAmbivalentKeywords || polShift >= 0.35 || sarcasm >= 0.25 {
		return PatternOscillation, fmt.Sprintf("정서 극성 반전(Δ=%.2f) 또는 양가적 감정 충돌(Sarcasm=%.2f, 웃음-고통 공존)", polShift, sarcasm)
	}

	// 2. Pattern A: Acute Explosion (급성 감정 폭발)
	// Rapid response, high divergence, high EVI, and actual negative/confrontational tone
	isNegativeOrAgitated := polarity < -0.05 ||
		strings.Contains(clientText, "뭐라") || strings.Contains(clientText, "에휴") ||
		strings.Contains(clientText, "징그럽") || strings.Contains(clientText, "왜") ||
		strings.Contains(clientText, "아니") || strings.Contains(clientText, "귀찮")
	if latencySec <= 120.0 && dyadicDiv >= 0.45 && evi >= 0.42 && !isLaughterOnly && !isLinkSharing && isNegativeOrAgitated {
		return PatternExplosion, fmt.Sprintf("즉각적 반응(%.0f초) 및 격앙/부정 어조(극성=%.2f)에 기인한 급성 정서 폭발", latencySec, polarity)
	}

	// 3. Pattern C: Latent Freezing (잠복형 정서 동결)
	// Prolonged silence (> 1800s / 30m) followed by short/evasive response (<= 15 runes)
	if latencySec >= 1800.0 && resp.CharCount <= 16 && !isLaughterOnly {
		return PatternFreezing, fmt.Sprintf("장시간 침묵(%.1f분) 후 회피성 단답(%d자)에 의한 방어적 정서 동결", latencySec/60.0, resp.CharCount)
	}

	// 4. Pattern B: Rapport Resonance (정서적 공명 및 이완)
	if (rapportScore >= 0.60 && dyadicDiv <= 0.40 && evi <= 0.35) || isLaughterOnly {
		return PatternResonance, fmt.Sprintf("상호 라포르 동조(%.2f) 및 친밀/공감 완화(Div=%.2f)", rapportScore, dyadicDiv)
	}

	// Default: Baseline Conversation
	return PatternBaseline, "일상적 친밀 대화 및 일반 정보 교환"
}

func truncateText(s string, maxChars int) string {
	runes := []rune(s)
	if len(runes) <= maxChars {
		return s
	}
	return string(runes[:maxChars]) + "..."
}
