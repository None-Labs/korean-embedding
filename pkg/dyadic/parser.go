package dyadic

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var (
	reMessageLine = regexp.MustCompile(`^([0-9]{4})년\s*([0-9]{1,2})월\s*([0-9]{1,2})일\s*(오전|오후)\s*([0-9]{1,2}):([0-9]{2}),\s*([^:]+)\s*:\s*(.*)$`)
	reDateHeader  = regexp.MustCompile(`^[0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일\s*(오전|오후)\s*[0-9]{1,2}:[0-9]{2}$`)
	reMediaOnly   = regexp.MustCompile(`^(?:사진(?:\s*[0-9]+장)?|동영상|이모티콘|보이스톡\s*해요|메시지가\s*삭제되었습니다\.|삭제된\s*메시지입니다\.)$`)
	kstZone       = time.FixedZone("KST", 9*3600)
)

// ParseKakaoTalkFile reads the KakaoTalk chat export and groups messages into continuous ParagraphTurns.
func ParseKakaoTalkFile(filePath string, cfg AnalysisConfig) ([]*ParagraphTurn, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open chat file %s: %w", filePath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Buffer size up to 1MB per line for long messages
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	var rawMessages []ChatMessage
	var curMsg *ChatMessage

	for scanner.Scan() {
		line := scanner.Text()
		line = strings.TrimPrefix(line, "\ufeff") // Strip UTF-8 BOM if present

		// Skip header lines
		if strings.HasPrefix(line, "저장한 날짜 :") || strings.Contains(line, "님과 카카오톡 대화") {
			continue
		}

		// Match single timestamp separator
		if reDateHeader.MatchString(strings.TrimSpace(line)) {
			continue
		}

		// Check if line starts a new message
		matches := reMessageLine.FindStringSubmatch(line)
		if len(matches) == 9 {
			if curMsg != nil {
				rawMessages = append(rawMessages, *curMsg)
			}

			year, _ := strconv.Atoi(matches[1])
			month, _ := strconv.Atoi(matches[2])
			day, _ := strconv.Atoi(matches[3])
			ampm := matches[4]
			hour, _ := strconv.Atoi(matches[5])
			minute, _ := strconv.Atoi(matches[6])
			speaker := strings.TrimSpace(matches[7])
			content := matches[8]

			// Convert 12-hour AM/PM to 24-hour KST
			if ampm == "오전" {
				if hour == 12 {
					hour = 0
				}
			} else { // 오후
				if hour < 12 {
					hour += 12
				}
			}

			msgTime := time.Date(year, time.Month(month), day, hour, minute, 0, 0, kstZone)
			isMedia := reMediaOnly.MatchString(strings.TrimSpace(content))

			curMsg = &ChatMessage{
				Timestamp: msgTime,
				Speaker:   speaker,
				Text:      content,
				IsMedia:   isMedia,
			}
		} else {
			// Multi-line continuation of current message
			if curMsg != nil {
				curMsg.Text += "\n" + line
				if curMsg.IsMedia && !reMediaOnly.MatchString(strings.TrimSpace(curMsg.Text)) {
					curMsg.IsMedia = false
				}
			}
		}
	}

	if curMsg != nil {
		rawMessages = append(rawMessages, *curMsg)
	}

	if err := scanner.Err(); err != nil {
		return nil, 0, fmt.Errorf("error reading chat file: %w", err)
	}

	totalRawMessages := len(rawMessages)
	if totalRawMessages == 0 {
		return nil, 0, fmt.Errorf("no valid chat messages found in %s", filePath)
	}

	// 2. Aggregate into Paragraph Turns
	maxGapDuration := time.Duration(cfg.MaxTurnGapMinutes * float64(time.Minute))
	var paragraphs []*ParagraphTurn
	var currentPara *ParagraphTurn

	turnCounter := 0
	for _, msg := range rawMessages {
		if currentPara == nil {
			turnCounter++
			currentPara = &ParagraphTurn{
				ID:           turnCounter,
				Speaker:      msg.Speaker,
				StartTime:    msg.Timestamp,
				EndTime:      msg.Timestamp,
				RawMessages:  []string{msg.Text},
				MessageCount: 1,
			}
			continue
		}

		timeDiff := msg.Timestamp.Sub(currentPara.EndTime)
		sameSpeaker := (msg.Speaker == currentPara.Speaker)
		withinGap := (timeDiff <= maxGapDuration)

		if sameSpeaker && withinGap {
			// Append to current paragraph turn
			currentPara.RawMessages = append(currentPara.RawMessages, msg.Text)
			currentPara.EndTime = msg.Timestamp
			currentPara.MessageCount++
		} else {
			// Finalize current paragraph
			finalizeParagraph(currentPara)
			paragraphs = append(paragraphs, currentPara)

			// Start new paragraph turn
			turnCounter++
			currentPara = &ParagraphTurn{
				ID:           turnCounter,
				Speaker:      msg.Speaker,
				StartTime:    msg.Timestamp,
				EndTime:      msg.Timestamp,
				RawMessages:  []string{msg.Text},
				MessageCount: 1,
			}
		}
	}

	if currentPara != nil {
		finalizeParagraph(currentPara)
		paragraphs = append(paragraphs, currentPara)
	}

	return paragraphs, totalRawMessages, nil
}

// finalizeParagraph merges raw messages, cleans text, and determines media-only status.
func finalizeParagraph(p *ParagraphTurn) {
	var nonMediaTexts []string
	mediaCount := 0

	for _, text := range p.RawMessages {
		trimmed := strings.TrimSpace(text)
		if reMediaOnly.MatchString(trimmed) {
			mediaCount++
		} else if trimmed != "" {
			nonMediaTexts = append(nonMediaTexts, trimmed)
		}
	}

	if len(nonMediaTexts) == 0 {
		p.MediaOnly = true
		p.MergedText = strings.Join(p.RawMessages, " ")
		p.CleanedText = "[미디어 발화]"
	} else {
		p.MediaOnly = false
		p.MergedText = strings.Join(nonMediaTexts, "\n")
		p.CleanedText = strings.Join(nonMediaTexts, " ")
	}

	p.CharCount = len([]rune(p.MergedText))
}
