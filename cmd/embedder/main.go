package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"gitlab.com/none-labs/korean-embedding/pkg/embedder"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	subArgs := os.Args[2:]

	switch command {
	case "embed":
		runEmbed(subArgs)
	case "similarity", "sim":
		runSimilarity(subArgs)
	case "cluster":
		runCluster(subArgs)
	case "version":
		fmt.Println("Korean Utterance 3024-d Embedding Engine v1.0.0 (Pure Go / Limmireu Pipeline)")
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`Korean Utterance Semantic Embedding Engine (3024-d) CLI

Usage:
  embedder <command> [arguments]

Available Commands:
  embed       Generate 3024-d unit embedding and speaker metadata for an utterance
  similarity  Compare two utterances across 4 semantic subspaces and MRL scales
  cluster     Cluster utterances by social stance, age hierarchy, and communicative intent
  version     Display engine version

Run 'embedder <command> -h' for more details on a specific command.`)
}

func runEmbed(args []string) {
	fs := flag.NewFlagSet("embed", flag.ExitOnError)
	text := fs.String("text", "", "Korean utterance text to embed (required)")
	dim := fs.Int("dim", 3024, "Output dimension (supports 512, 1024, 1536, 3024)")
	jsonOutput := fs.Bool("json", false, "Output results in JSON format")
	showSubspaces := fs.Bool("subspaces", false, "Display individual subspace norms and vector previews")
	modelPath := fs.String("model", "", "Path to optional trained BPE tokenizer JSON")

	_ = fs.Parse(args)

	if *text == "" {
		fmt.Fprintln(os.Stderr, "Error: --text is required.")
		fs.Usage()
		os.Exit(1)
	}

	cfg := embedder.DefaultEmbedderConfig()
	if *modelPath != "" {
		cfg.BPETokenizerPath = *modelPath
	}

	emb, err := embedder.NewEmbedder(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing embedder: %v\n", err)
		os.Exit(1)
	}

	res, err := emb.Embed(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error embedding utterance: %v\n", err)
		os.Exit(1)
	}

	// Slice vector if dim < 3024
	vec := res.Vector
	if *dim < embedder.TotalDim {
		slicer := embedder.NewMRLSlicer()
		sliced, err := slicer.Slice(res.Vector, *dim)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error slicing dimension to %d: %v\n", *dim, err)
			os.Exit(1)
		}
		vec = sliced
	}

	if *jsonOutput {
		outMap := map[string]interface{}{
			"dimension": len(vec),
			"metadata":  res.Metadata,
			"vector":    vec,
		}
		if *showSubspaces {
			outMap["subspaces"] = res.Subspaces
		}
		outMap["mrl_slices_preview"] = map[string]interface{}{
			"512_d_norm":  normStr(res.MRLSlices[512]),
			"1024_d_norm": normStr(res.MRLSlices[1024]),
			"1536_d_norm": normStr(res.MRLSlices[1536]),
			"3024_d_norm": normStr(res.MRLSlices[3024]),
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(outMap)
		return
	}

	// CLI Human-Readable Output
	fmt.Println("================================================================================")
	fmt.Println("          3024-Dimensional Korean Utterance Semantic Embedding")
	fmt.Println("================================================================================")
	fmt.Printf("• Original Utterance : %s\n", res.Metadata.OriginalText)
	fmt.Printf("• Normalized Text    : %s\n", res.Metadata.CleanedText)
	if len(res.Metadata.MaskedTags) > 0 {
		fmt.Printf("• Masked Entities    : %v\n", res.Metadata.MaskedTags)
	}
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                    Speaker Intent & Social Metadata")
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("• Speech Act (Level 2)  : %s (Confidence: %.2f%%)\n", res.Metadata.SpeechAct, res.Metadata.SpeechActProb*100)
	fmt.Printf("• Speech Level (어조)   : %s (%s)\n", res.Metadata.SpeechLevel, res.Metadata.SpeechLevelDesc)
	fmt.Printf("• Social Stance (관계)  : %s\n", res.Metadata.SocialStanceDesc)
	fmt.Printf("• Age Group Subspace    : %s\n", res.Metadata.AgeGroupDesc)
	fmt.Printf("• Politeness / Formal   : Politeness=%.2f, Formality=%.2f, Aegyo=%.2f\n",
		res.Metadata.Politeness, res.Metadata.Formality, res.Metadata.AegyoScore)
	fmt.Printf("• Pragmatic Nuance      : Sarcasm=%.2f (Detected=%t), Polarity=%.2f, Intent='%s'\n",
		res.Metadata.SarcasmScore, res.Metadata.SarcasmDetected, res.Metadata.AttitudePolarity, res.Metadata.TrueIntent)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("• Output Dimension      : %d (Unit L2-Norm: %.6f)\n", len(vec), vectorNorm(vec))
	fmt.Printf("• Vector Preview (Head) : %v\n", previewSlice(vec, 5))
	fmt.Printf("• Vector Preview (Tail) : %v\n", previewTail(vec, 5))

	if *showSubspaces {
		fmt.Println("--------------------------------------------------------------------------------")
		fmt.Println("                      4 Subspace Decomposition")
		fmt.Println("--------------------------------------------------------------------------------")
		for _, info := range embedder.AllSubspaces {
			sub := res.Subspaces[info.Type]
			fmt.Printf("• Subspace [%-7s] (%4d-d): L2-Norm=%.4f, Preview=%v\n",
				info.Type, len(sub), vectorNorm(sub), previewSlice(sub, 3))
		}
	}
	fmt.Println("================================================================================")
}

func runSimilarity(args []string) {
	fs := flag.NewFlagSet("similarity", flag.ExitOnError)
	text1 := fs.String("text1", "", "First utterance text (required)")
	text2 := fs.String("text2", "", "Second utterance text (required)")
	jsonOutput := fs.Bool("json", false, "Output results in JSON format")
	modelPath := fs.String("model", "", "Path to optional trained BPE tokenizer JSON")

	_ = fs.Parse(args)

	if *text1 == "" || *text2 == "" {
		fmt.Fprintln(os.Stderr, "Error: both --text1 and --text2 are required.")
		fs.Usage()
		os.Exit(1)
	}

	cfg := embedder.DefaultEmbedderConfig()
	if *modelPath != "" {
		cfg.BPETokenizerPath = *modelPath
	}

	emb, err := embedder.NewEmbedder(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing embedder: %v\n", err)
		os.Exit(1)
	}

	sim, err := emb.Compare(*text1, *text2)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error comparing utterances: %v\n", err)
		os.Exit(1)
	}

	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(sim)
		return
	}

	fmt.Println("================================================================================")
	fmt.Println("             3024-d Utterance Semantic Similarity & Alignment")
	fmt.Println("================================================================================")
	fmt.Printf("• Utterance 1 : %s\n", sim.Text1)
	fmt.Printf("• Utterance 2 : %s\n", sim.Text2)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("★ Overall 3024-d Cosine Similarity : %.4f (%s)\n",
		sim.CosineSimilarity, similarityLabel(sim.CosineSimilarity))
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("                    Subspace Similarity Breakdown")
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("1. Literal Proposition (1536-d) : %.4f  (어휘 표면 및 객관적 사실 명제 일치율)\n", sim.LiteralSimilarity)
	fmt.Printf("2. Speaker Intent      ( 512-d) : %.4f  (화자 행동 요구 및 화행 일치율: Match=%t)\n", sim.IntentSimilarity, sim.IntentMatch)
	fmt.Printf("3. Social Stance       ( 512-d) : %.4f  (공사 관계 및 사회적 거리감 일치율: Match=%t)\n", sim.SocialSimilarity, sim.SocialStanceMatch)
	fmt.Printf("4. Pragmatic Nuance    ( 464-d) : %.4f  (반어법/어조 괴리도 일치율, Divergence=%.2f)\n", sim.NuanceSimilarity, sim.IronyDivergence)
	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Println("            Matryoshka Representation Learning (MRL) Scaling")
	fmt.Println("--------------------------------------------------------------------------------")
	for _, dim := range embedder.SupportedMRLDims {
		fmt.Printf("• %4d-d Cosine Similarity : %.4f\n", dim, sim.MRLComparison[dim])
	}
	fmt.Println("================================================================================")
}

func runCluster(args []string) {
	fs := flag.NewFlagSet("cluster", flag.ExitOnError)
	k := fs.Int("k", 3, "Number of clusters to form")
	filePath := fs.String("file", "", "Path to text file containing utterances (one per line)")
	jsonOutput := fs.Bool("json", false, "Output results in JSON format")
	modelPath := fs.String("model", "", "Path to optional trained BPE tokenizer JSON")

	_ = fs.Parse(args)

	cfg := embedder.DefaultEmbedderConfig()
	if *modelPath != "" {
		cfg.BPETokenizerPath = *modelPath
	}

	emb, err := embedder.NewEmbedder(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing embedder: %v\n", err)
		os.Exit(1)
	}

	var dataset []string
	if *filePath != "" {
		f, err := os.Open(*filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening file %s: %v\n", *filePath, err)
			os.Exit(1)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				dataset = append(dataset, line)
			}
		}
	} else {
		// Benchmark representative Korean utterances
		dataset = []string{
			// Formal / Organizational (공적 보고, 비즈니스)
			"금일 프로젝트 진행 상황을 보고드립니다. 검토 후 피드백 부탁드립니다.",
			"오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.",
			"요청하신 3분기 결산 보고서 송부드립니다. 확인 바랍니다.",
			"고객사 미팅 안건에 대해 사전 협의 요청드립니다.",

			// Informal Romantic / Couple (연인 간, 애교)
			"자기야 오늘 저녁에 맛있는 거 먹으러 가용~ 보고싶당 ㅎㅎ",
			"내사랑 오늘 하루도 고생 많았어용! 얼른 퇴근하구 봐요 쪽~",
			"오빠 오늘 데이트 어디로 갈까용? ㅎㅎ 기대된당!",

			// Informal Peer / Friend (친구 간, 해체, 슬랭)
			"야 이번 주말에 롤 ㄱㄱ? 개웃기네 ㅋㅋ",
			"오늘 점심 뭐 먹냐? 존맛탱 맛집 찾음 ㄲㄲ",
			"대박 진짜 미쳤다 ㄹㅇ 인정 ㅋㅋㅋ",

			// Senior / Traditional (50대+ 장노년층)
			"어르신 진지 잡수셨습니까, 옥체 강녕하시기를 기원하옵니다.",
			"춘추가 높으신데도 평안하시니 참으로 다행입니다.",
		}
	}

	if len(dataset) < *k {
		*k = len(dataset)
	}

	// 1. Embed all utterances
	results, err := emb.EmbedBatch(dataset)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error embedding dataset: %v\n", err)
		os.Exit(1)
	}

	// 2. Spherical K-Means Clustering on 3024-d unit sphere
	numK := *k
	centroids := make([][]float32, numK)
	for i := 0; i < numK; i++ {
		centroids[i] = make([]float32, embedder.TotalDim)
		copy(centroids[i], results[i].Vector)
	}

	assignments := make([]int, len(results))

	// Run 10 iterations of spherical K-Means
	for iter := 0; iter < 10; iter++ {
		// Assign to nearest centroid
		for i, r := range results {
			bestCluster := 0
			bestSim := float32(-1.0)
			for c := 0; c < numK; c++ {
				sim := embedder.CosineSimilarity(r.Vector, centroids[c])
				if sim > bestSim {
					bestSim = sim
					bestCluster = c
				}
			}
			assignments[i] = bestCluster
		}

		// Recompute centroids
		newCentroids := make([][]float32, numK)
		counts := make([]int, numK)
		for c := 0; c < numK; c++ {
			newCentroids[c] = make([]float32, embedder.TotalDim)
		}

		for i, c := range assignments {
			counts[c]++
			for j := 0; j < embedder.TotalDim; j++ {
				newCentroids[c][j] += results[i].Vector[j]
			}
		}

		for c := 0; c < numK; c++ {
			if counts[c] > 0 {
				centroids[c] = embedder.L2Normalize(newCentroids[c])
			}
		}
	}

	type ClusterInfo struct {
		ClusterID        int      `json:"cluster_id"`
		MemberCount      int      `json:"member_count"`
		DominantStance   string   `json:"dominant_stance"`
		DominantAgeGroup string   `json:"dominant_age_group"`
		Utterances       []string `json:"utterances"`
	}

	clusters := make([]ClusterInfo, numK)
	for c := 0; c < numK; c++ {
		stanceCounts := make(map[embedder.SocialStance]int)
		ageCounts := make(map[embedder.AgeGroup]int)
		var utts []string

		for i, a := range assignments {
			if a == c {
				utts = append(utts, results[i].Metadata.OriginalText)
				stanceCounts[results[i].Metadata.SocialStance]++
				ageCounts[results[i].Metadata.AgeGroup]++
			}
		}

		domStance := ""
		maxStance := 0
		for st, cnt := range stanceCounts {
			if cnt > maxStance {
				maxStance = cnt
				domStance = string(st)
			}
		}

		domAge := ""
		maxAge := 0
		for ag, cnt := range ageCounts {
			if cnt > maxAge {
				maxAge = cnt
				domAge = string(ag)
			}
		}

		clusters[c] = ClusterInfo{
			ClusterID:        c + 1,
			MemberCount:      len(utts),
			DominantStance:   domStance,
			DominantAgeGroup: domAge,
			Utterances:       utts,
		}
	}

	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(clusters)
		return
	}

	fmt.Println("================================================================================")
	fmt.Printf("        Korean Utterance Semantic Clustering (K=%d Spherical Clusters)\n", numK)
	fmt.Println("================================================================================")
	for _, c := range clusters {
		fmt.Printf("\n[Cluster %d] (%d Utterances) - Dominant: Stance=%s, Age=%s\n",
			c.ClusterID, c.MemberCount, c.DominantStance, c.DominantAgeGroup)
		fmt.Println("--------------------------------------------------------------------------------")
		for idx, u := range c.Utterances {
			fmt.Printf("  %d. %s\n", idx+1, u)
		}
	}
	fmt.Println("\n================================================================================")
}

func similarityLabel(sim float32) string {
	switch {
	case sim >= 0.85:
		return "Extremely High Semantic Alignment"
	case sim >= 0.70:
		return "High Semantic & Intent Alignment"
	case sim >= 0.50:
		return "Moderate Contextual Similarity"
	default:
		return "Divergent Communicative Intent or Stance"
	}
}

func vectorNorm(vec []float32) float64 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}

func normStr(vec []float32) string {
	return fmt.Sprintf("%.6f", vectorNorm(vec))
}

func previewSlice(vec []float32, n int) []string {
	if len(vec) < n {
		n = len(vec)
	}
	res := make([]string, n)
	for i := 0; i < n; i++ {
		res[i] = fmt.Sprintf("%.4f", vec[i])
	}
	return res
}

func previewTail(vec []float32, n int) []string {
	if len(vec) < n {
		n = len(vec)
	}
	res := make([]string, n)
	start := len(vec) - n
	for i := 0; i < n; i++ {
		res[i] = fmt.Sprintf("%.4f", vec[start+i])
	}
	return res
}
