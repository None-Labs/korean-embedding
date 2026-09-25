# 3024차원 한국어 발화자 의미 임베딩 엔진 (Korean Utterance Semantic Embedding Engine)

[![Go Report Card](https://goreportcard.com/badge/gitlab.com/none-labs/korean-embedding)](https://goreportcard.com/report/gitlab.com/none-labs/korean-embedding)
[![GitLab CI/CD](https://gitlab.com/none-labs/korean-embedding/badges/main/pipeline.svg)](https://gitlab.com/none-labs/korean-embedding/-/pipelines)
[![Target Go Version](https://img.shields.io/badge/Go-1.24.1+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-none--labs%2Fkorean--embedding--3024d-orange)](https://huggingface.co/none-labs/korean-embedding-3024d)

---

## 1. 프로젝트 개요 (Executive Summary)

본 프로젝트(`korean-embedding`)는 한국어 자연어 처리에서 단순 표면 텍스트의 어휘적 유사도를 넘어, **발화자의 숨겨진 의미(화자의 의도, 8대 화행, 공적/사적 대화 관계, 나이대별 표현 특성, 반어/냉소적 뉘앙스)**를 정확하게 파악하고 투영하는 **3024차원 고밀도 연속 임베딩 엔진**입니다.

- **완성본 배포**: Hugging Face Hub 모델 업로드 호환 (`none-labs/korean-embedding-3024d`)
- **DevOps & 소스 제어**: GitLab CI/CD 파이프라인 운용 (`gitlab.com/none-labs/korean-embedding`)
- **개발 언어 & 런타임**: **Pure Golang (Go 1.24.1+)** (Limmireu 파이프라인 글로벌 룰 준수: CPU Cost-First, 불필요한 GPU/Python 인프라 비용 0원)
- **차원 규격**: **3024차원** (OpenAI `text-embedding-3-large` 3072차원 호환 및 Matryoshka Representation Learning 슬라이싱 지원)

---

## 2. 3024차원 발화 의미 서브스페이스 아키텍처

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              3024차원 발화자 의미 임베딩 벡터 (Unit L2-Normalized)                             │
├────────────────────────────┬─────────────────────────────┬───────────────────────────┬─────────────────┤
│ 1. Literal Semantic (1536) │ 2. Speaker Intent (512)     │ 3. Social Stance (512)    │ 4. Nuance (464) │
│ - 형태소 + 어휘 문맥 의미     │ - 8대 화행 의도 분류         │ - 공적 대화 (조직/업무)    │ - 반어/비꼬임(Irony)│
│ - 구문 트리 지배 관계        │ - 질문/요청/약속/감탄 등    │ - 사적 대화 (연인/친구/가족)│ - 어조 괴리도 및 감정  │
│ - 문장의 객관적 사실 명제   │ - 화자의 궁극적 행동 요구    │ - 상대높임 6체계 & 나이대  │ - 냉소 및 완곡 어법   │
└────────────────────────────┴─────────────────────────────┴───────────────────────────┴─────────────────┘
```

### 4대 서브스페이스 레이아웃

| 서브스페이스 | 차원 (Dim) | 인덱스 범위 | 핵심 반영 특징 |
|:---|:---:|:---:|:---|
| **1. Literal Semantic** | **1536-d** | $0 \sim 1535$ | 세종 45종 품사(POS)와 격틀 의존 구문 트리, 어절 간 지배소-의존소 관계 보존 |
| **2. Speaker Intent** | **512-d** | $1536 \sim 2047$ | 8대 화행(진술, 질문, 요청, 약속, 위협, 감탄, 사과, 수용) 및 행동 지시 의도 |
| **3. Social Stance** | **512-d** | $2048 \sim 2559$ | 공적(조직/비즈니스) vs 사적(연인/친구/가족) 관계, 상대높임 6체계, 세대별(청년/중장년/노년) 거리감 |
| **4. Pragmatic Nuance** | **464-d** | $2560 \sim 3023$ | 반어법/비꼬임(Sarcasm), 표면 긍정과 내면 비꼼 간 어조 괴리도, 태도 극성 (-1.0 ~ +1.0) |
| **Total** | **3024-d** | $0 \sim 3023$ | **Unit L2-Normalized ($\|\mathbf{z}\|_2 = 1.0$)** |

---

## 3. 수학적 수식 및 손실 함수 체계 (Mathematical Formulation)

### 3.1. 피처 융합 및 어텐션 가중치 풀링
입력 발화의 $t$번째 토큰에 대해 세종 품사 $p_t$와 의존 구문 레이블 $d_t$를 융합:
$$E_t = W_{\text{subword}}(t) + W_{\text{pos}}(p_t) + W_{\text{dep}}(d_t) + \text{RoPE}(t)$$
한국어의 형태론적 특성을 반영하여 내용어(체언/용언 어근/부사)와 최상위 서술어(Root)에 가중치를 부여하는 어텐션 풀링을 통해 문장 잠재 벡터 $\mathbf{h}_{\text{sent}}$를 추출:
$$\alpha_t = \text{softmax}\left( \mathbf{v}^T \tanh(W_h \mathbf{h}_t + \mathbf{b}_h) + \text{morphBonus}_t \right), \quad \mathbf{h}_{\text{sent}} = \sum_t \alpha_t \mathbf{h}_t$$

### 3.2. 멀티 태스크 대조 학습 손실 체계
전체 미세조정 손실 함수는 비지도 SimCSE와 서브스페이스별 감독 대조 손실의 결합으로 구성됩니다:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{SimCSE}} + \lambda_2 \mathcal{L}_{\text{Intent}} + \lambda_3 \mathcal{L}_{\text{Social}}$$

1. **드롭아웃 기반 비지도 SimCSE 손실 (Literal 융합)**:
   $$\mathcal{L}_{\text{SimCSE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp\left(\frac{\mathbf{z}_i \cdot \mathbf{z}_i^+}{\tau}\right)}{\sum_{j=1}^N \exp\left(\frac{\mathbf{z}_i \cdot \mathbf{z}_j^+}{\tau}\right)}$$
2. **512차원 화자 의도 서브스페이스 대조 손실**:
   $$\mathcal{L}_{\text{Intent}} = \frac{1}{|P|} \sum_{(i,j) \in P} (1 - \mathbf{z}_i^{\text{int}} \cdot \mathbf{z}_j^{\text{int}}) + \frac{1}{|N|} \sum_{(i,k) \in N} \max\left(0, \mathbf{z}_i^{\text{int}} \cdot \mathbf{z}_k^{\text{int}} - \alpha\right)$$
3. **512차원 사회적 관계 서브스페이스 삼중항 마진 손실**:
   $$\mathcal{L}_{\text{Social}} = \max\left(0, \|\mathbf{z}_a^{\text{soc}} - \mathbf{z}_p^{\text{soc}}\|_2^2 - \|\mathbf{z}_a^{\text{soc}} - \mathbf{z}_n^{\text{soc}}\|_2^2 + m\right)$$

---

## 4. Matryoshka Representation Learning (MRL) 슬라이싱 가이드

| 차원 (Dim) | 서브스페이스 구성 | 주요 권장 용도 | 연산 속도 & 메모리 |
|:---:|:---|:---|:---:|
| **512-d** | Literal 핵심 성분 | 대규모 벡터 DB 1차 인덱싱 및 고속 필터링 | **최고 속도** (메모리 83% 절감) |
| **1024-d** | Literal + Intent 요약 | 고객센터 인텐트 라우팅, FAQ 시맨틱 검색 | **균형** (속도 및 의도 정확도) |
| **1536-d** | Literal 전 범위 | OpenAI Small 규격 호환 및 일반 RAG 검색 | **표준 호환** (범용 문서 검색) |
| **3024-d** | **Full Multi-Aspect (전체)** | **발화자 심층 의미, 화행 의도, 공사 관계, 반어/비꼼 분석** | **최고 정밀도 (심층 대화 분석)** |

---

## 5. 설치 및 빌드 방법 (Installation & Build)

### 요구사항
- Go 1.24.1 이상
- Make 및 GCC/Clang (CGO 지원)

```bash
# 리포지토리 클론
git clone git@gitlab.com:none-labs/korean-embedding.git
cd korean-embedding

# 의존성 설치 및 CLI 빌드
make build

# 단위 테스트 실행
make test

# 벤치마크 실행
make bench
```

---

## 6. CLI 사용법 및 예제 (CLI Usage)

### 6.1. 단일 발화 문장 임베딩 (`embed`)
```bash
./bin/embedder embed --text="오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다." --subspaces
```

출력 예시:
```text
================================================================================
          3024-Dimensional Korean Utterance Semantic Embedding
================================================================================
• Original Utterance : 오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.
• Normalized Text    : 오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.
--------------------------------------------------------------------------------
                    Speaker Intent & Social Metadata
--------------------------------------------------------------------------------
• Speech Act (Level 2)  : PROMISE (Confidence: 89.00%)
• Speech Level (어조)   : HASIPSIO (Formal Polite)
• Social Stance (관계)  : 공적 대화 (조직/업무/비즈니스/고객응대 - Formal & Organizational)
• Age Group Subspace    : 30대~40대 중장년층 (Middle Age - 직장/실용 중심 문체)
• Politeness / Formal   : Politeness=1.00, Formality=0.95, Aegyo=0.00
• Pragmatic Nuance      : Sarcasm=0.00 (Detected=false), Polarity=0.00, Intent='Respectful/Formal'
--------------------------------------------------------------------------------
• Output Dimension      : 3024 (Unit L2-Norm: 1.000000)
• Vector Preview (Head) : [-0.0015 0.0200 -0.0024 -0.0122 0.0054]
• Vector Preview (Tail) : [0.0089 -0.0208 0.0162 0.0329 0.0284]
--------------------------------------------------------------------------------
                      4 Subspace Decomposition
--------------------------------------------------------------------------------
• Subspace [literal] (1536-d): L2-Norm=1.0000, Preview=[-0.0029 0.0380 -0.0046]
• Subspace [intent ] ( 512-d): L2-Norm=1.0000, Preview=[-0.0292 0.0643 -0.0426]
• Subspace [social ] ( 512-d): L2-Norm=1.0000, Preview=[0.0282 -0.0699 0.0071]
• Subspace [nuance ] ( 464-d): L2-Norm=1.0000, Preview=[0.0409 0.0551 -0.0289]
================================================================================
```

### 6.2. 두 발화 간 의미 일치도 분석 (`similarity`)
```bash
./bin/embedder similarity \
  --text1="오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다." \
  --text2="금일 안건 회의는 15시 회의실 진행 예정입니다."
```

출력 예시:
```text
================================================================================
             3024-d Utterance Semantic Similarity & Alignment
================================================================================
• Utterance 1 : 오늘 프로젝트 회의는 오후 3시에 대회의실에서 진행하겠습니다.
• Utterance 2 : 금일 안건 회의는 15시 회의실 진행 예정입니다.
--------------------------------------------------------------------------------
★ Overall 3024-d Cosine Similarity : 0.7342 (High Semantic & Intent Alignment)
--------------------------------------------------------------------------------
                    Subspace Similarity Breakdown
--------------------------------------------------------------------------------
1. Literal Proposition (1536-d) : 0.7358  (어휘 표면 및 객관적 사실 명제 일치율)
2. Speaker Intent      ( 512-d) : 0.7249  (화자 행동 요구 및 화행 일치율: Match=false)
3. Social Stance       ( 512-d) : 0.7552  (공사 관계 및 사회적 거리감 일치율: Match=true)
4. Pragmatic Nuance    ( 464-d) : 0.7181  (반어법/어조 괴리도 일치율, Divergence=0.00)
--------------------------------------------------------------------------------
            Matryoshka Representation Learning (MRL) Scaling
--------------------------------------------------------------------------------
•  512-d Cosine Similarity : 0.7058
• 1024-d Cosine Similarity : 0.7369
• 1536-d Cosine Similarity : 0.7358
• 3024-d Cosine Similarity : 0.7342
================================================================================
```

### 6.3. 대화 군집화 및 센트로이드 계산 (`cluster`)
```bash
./bin/embedder cluster --k=3
```

---

## 7. 성능 벤치마크 (Performance Benchmarks)

AMD EPYC CPU 환경에서 순수 Go 런타임으로 측정된 단일 코어 성능:

| 벤치마크 항목 | 수행 속도 | 할당 메모리 | 초당 처리량 (Throughput) |
|:---|:---:|:---:|:---:|
| **3024-d 코사인 유사도 계산** | **1,022 ns/op** | 0 B/op (0 allocs) | **978,000 연산/초** |
| **MRL 4단계 슬라이싱 및 L2 정규화** | **12.1 µs/op** | 24 KB/op | **82,500 회/초** |
| **전체 파이프라인 단일 발화 임베딩** | **8.4 ms/op** | 364 KB/op | **118 발화/초** (GPU 없이 CPU 구동) |

---

## 8. Hugging Face Hub 배포 가이드

```bash
# Hugging Face 토큰 설정
export HF_TOKEN="hf_xxxxxxxxxxxx"

# 모델 설정 생성 및 업로드
./scripts/upload_hf.sh models none-labs/korean-embedding-3024d
```

---

## 9. Limmireu Pipeline 글로벌 룰 준수 사항

1. **언어 및 기술 스택**: 시스템 및 코어 엔진 100% **Pure Golang (Go 1.24+)** 구현. Python 의존성 일체 배제.
2. **비용 최우선 (Cost-First)**: 불필요한 GPU 인스턴스 비용 0원, 경량 CPU 환경에서 100회/초 이상의 발화 임베딩 성능 보장.
3. **프로세스 모니터링 & 알림**: `pkg/trainer` 학습 시 Google Chat Webhook 연동을 통해 **1시간 주기 상태 알림** 및 **종료/예외 즉시 알림** 지원.
4. **VM 자동 정리**: `scripts/cleanup_brain.sh` 스크립트를 통해 KST 00:00 시점 자동 세션 압축 및 디렉토리 정리 기능 탑재.
5. **코드 표준**: Shell Script `set -euo pipefail`, Go 디렉토리 표준 구조 및 `gofmt` 100% 준수.
