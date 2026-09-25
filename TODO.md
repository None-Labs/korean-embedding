# [구현 명세서 및 계획서] 3024차원 한국어 발화자 의미 임베딩 엔진 (Korean Utterance Semantic Embedding Engine)

---

## 1. 프로젝트 목적 및 추구하는 방향 (Core Purpose & Objectives)

본 프로젝트(`korean-embedding`)는 앞서 구축한 `korean-tokenizer`의 하이브리드 형태소 분석 및 통사 구문 트리, 담화 분석 결과를 바탕으로, 단순 표면 텍스트의 어휘적 유사도를 넘어 **발화자의 숨겨진 의미(화자의 의도, 화행, 공적/사적 대화 관계, 나이대별 표현 특성, 반어/냉소적 뉘앙스)**를 정확하게 파악하고 투영하는 **3024차원 고밀도 연속 임베딩 엔진**을 구축하는 것을 목적으로 합니다.

- **완성본 배포**: Hugging Face Hub 모델 업로드 예정
- **DevOps & 소스 제어**: GitLab CI/CD 파이프라인 운용 (`gitlab.com/none-labs/korean-embedding`)
- **개발 언어 & 런타임**: **Pure Golang (Go 1.24.1+)** (Limmireu 파이프라인 글로벌 룰 준수: CPU Cost-First, 불필요한 GPU/Python 인프라 비용 0원)
- **차원 규격**: **3024차원** (OpenAI `text-embedding-3-large` 3072차원 규격 호환 및 Matryoshka Representation Learning 슬라이싱 지원)

---

## 2. 발화자 의미 파악을 위한 데이터 파이프라인 및 정제 요구사항

### 2.1. 마스킹 및 태그 트래킹 연동
- BigQuery 및 전처리 파이프라인에서 필터링된 특수 마스킹 태그 트래킹:
  - `<이름>`, `#이름`, `#@이름#` 등 다양하게 인입되는 개인식별 마스킹을 표준 `<이름>` 형태로 정규화하여 토크나이저 및 임베딩 입력에 보존
  - 이메일(`<이메일>`), URL 링크(`<링크>`) 태그를 특수 토큰으로 보존하여 문맥 임베딩에 반영
- Python, HTML 등 비한글/스크립트 코드는 완벽히 배제하고 **순수 한국어 발화 및 정규화된 태그**만을 임베딩 벡터화

### 2.2. 발화자 의미 및 클러스터링(Clustering) 3대 축
1. **공적인 대화 (Formal & Organizational Stance)**
   - 조직 간 대화, 공적 보고, 비즈니스 협의, 고객 응대 (하십시오체/해요체, 객관적 서술)
2. **사적인 대화 (Informal & Interpersonal Stance)**
   - 연인 간 대화 (친밀도 극대, 애정/감정 표출, 축약형)
   - 친구 간 대화 (해체, 격식 배제, 은어/슬랭)
   - 가족 간 대화 (존칭과 친밀함의 혼합, 세대 간 어휘)
3. **나이대별 발화 특성 (Age Group Subspace)**
   - 10대~20대 청년층 (신조어, 자모 축약, 빠른 어미 변화)
   - 30대~40대 중장년층 (직장/실용 중심 문체)
   - 50대 이상 장년/노년층 (전통적 경어법, 완곡한 간접 화법)

---

## 3. 3024차원 발화 의미 서브스페이스 아키텍처 (Multi-Aspect Subspace)

발화자의 표면 문자열뿐만 아니라 **화자의 의도와 심리적 태도**를 다각도로 파악하기 위해, 3024차원 임베딩 공간 $\mathbf{z} \in \mathbb{R}^{3024}$를 4대 의미론적 서브스페이스로 분할·결합합니다:

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

### 3.1. Level 1: 형태론 & 통사 융합 기반 어휘 표현 (Literal Subspace, $1536$-d)
- 세종 45종 품사(POS)와 격틀(Case-frame) 의존 구문 트리가 융합된 트랜스포머 인코더 히든 벡터
- 주어-서술어-목적어의 문법적 구조를 보존하여 OOV(미등록어) 왜곡을 방지한 객관적 발화 내용

### 3.2. Level 2: 화자 의도 및 화행 (Speaker Intent Subspace, $512$-d)
- 화자가 발화를 통해 청자에게 전달하고자 하는 8대 화행 투영:
  - `진술(STATEMENT)`: 사실/정보의 단순 전달
  - `질문(QUESTION)`: 정보 요구나 확인 요청
  - `요청(REQUEST)`: 청자의 특정 행동 지시 또는 부탁
  - `약속(PROMISE)`: 화자의 미래 행동 확약
  - `위협(THREAT)`: 경고, 제재, 압박
  - `감탄/감정표출(EXCLAMATION)`: 놀람, 환희, 슬픔 등 감정 발산
  - `사과(APOLOGY)`: 잘못 인정 및 유감 표명
  - `수용(ACCEPTANCE)`: 상대방 의견 동의 및 승낙

### 3.3. Level 3: 사회적 관계 및 나이대 (Social Stance & Hierarchy Subspace, $512$-d)
- 상대높임법 6체계(하십시오체, 해요체, 하게체, 하오체, 해라체, 해체)를 반영한 종결 어미 벡터
- **대화 성격 분류**: 공적인 대화(조직 간) vs 사적인 대화(연인간, 친구간, 가족간)
- 화자의 나이대별 어휘 분포와 사회적 거리감이 코사인 각도로 매핑

### 3.4. Level 4: 화용론적 뉘앙스 및 반어법 (Pragmatic Nuance & Irony Subspace, $464$-d)
- 표면 어휘의 긍정성과 실제 발화 어조/맥락 간의 불일치를 감지하는 비꼬임(Sarcasm) 신호
- 완곡한 거절, 냉소, 반어 표현을 포착하여 겉과 속이 다른 발화자의 실제 심리 표현 벡터화

---

## 4. 전체 시스템 파이프라인 구조

```
[원문 한국어 발화 스트림]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Level 1. 형태소 & 어조 분석 (korean-tokenizer 연동)     │
│  - 세종 45종 품사 Lattice Viterbi 분석 + 마스킹 태그 보존  │
│  - 종결어미 기반 상대높임 6체계 및 어조 극성 추출       │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Level 2. 의존 구문 분석 (Syntax Parser)                │
│  - 어절 단위 주어-목적어-서술어 지배소 의존 트리 구조화 │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Level 3. Feature Fusion & Transformer Encoder          │
│  - E_i = W_subword + W_pos + W_dep + RoPE 위치 임베딩   │
│  - GoMLX / Native Multi-Head Self-Attention            │
│  - Attention-Weighted Pooling (의미론적 주요 어근 집중)  │
└───────────────────────┬────────────────────────────────┘
                        │ 512-d Latent Hidden Vector
                        ▼
┌────────────────────────────────────────────────────────┐
│ Level 4. 3024-d Multi-Aspect Semantic Projection Layer │
│  - Literal Projection (512 -> 1536)                    │
│  - Speaker Intent Projection (512 -> 512)              │
│  - Social Stance Projection (512 -> 512)               │
│  - Pragmatic Nuance Projection (512 -> 464)            │
│  - L2 Unit Normalization                               │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
       [3024-d Unit Utterance Embedding Output + Metadata]
```

---

## 5. 단계별 상세 구현 로드맵 (Implementation TODO)

### Phase 1: 기반 인프라 구축 및 모듈 초기화
- [x] Go 1.24 모듈 생성 (`gitlab.com/none-labs/korean-embedding`)
- [x] `korean-tokenizer` 라이브러리(`pkg/morphology`, `pkg/syntax`, `pkg/bpe`) 의존성 연동
- [x] `Makefile` 및 `.gitlab-ci.yml` CI 파이프라인 (Lint, Test, Build, Security Scan) 구축
- [x] `.gitignore` 설정

### Phase 2: 3024차원 임베딩 코어 엔진 구현 (`pkg/embedder`)
- [x] `types.go`: 3024차원 서브스페이스 레이아웃 및 발화 의미 메타데이터(`UtteranceMetadata`) 정의
- [x] `projection.go`: 4대 서브스페이스(1536 + 512 + 512 + 464 = 3024) 프로젝션 가중치 레이어
- [x] `pooling.go`: 패딩 마스크를 고려한 Attention-Weighted Pooling 및 Mean Pooling
- [x] `mrl.go`: Matryoshka Representation Learning (512, 1024, 1536, 3024차원 유닛 슬라이싱)
- [x] `embedder.go`: 원문 발화 $\to$ 토큰화/형태소/구문 분석 $\to$ 3024차원 벡터 및 메타데이터 단일 추출 파이프라인

### Phase 3: 발화자 의미 학습 및 미세조정 엔진 (`pkg/trainer`)
- [x] `simcse.go`: 드롭아웃 마스크 기반 비지도 대조 학습(SimCSE) 루프
- [x] `contrastive_intent.go`: 동일 화행/의도 문장 쌍에 대한 인텐트 대조 손실(Intent Contrastive Loss)
- [x] `contrastive_social.go`: 공적(조직) vs 사적(연인/친구/가족) 대화 분리를 위한 삼중항 마진 손실(Triplet Margin Loss)
- [x] GoMLX / Native AdamW 최적화기를 통한 CPU 고속 배치 학습 루프

### Phase 4: CLI 인터페이스 및 대화 분석 도구 (`cmd/embedder`)
- [x] `embed`: 단일 발화 문장의 3024차원 임베딩 벡터 및 화자 의미 메타데이터(화행, 어조, 공사 분류) JSON 출력
- [x] `similarity`: 두 발화 문장 간의 코사인 유사도 및 서브스페이스별 의미 일치도 분석
  - 표면 어휘 일치율(Literal) vs 화자 의도 일치율(Intent) vs 사회적 태도 일치율(Social)
- [x] `cluster`: 나이대별 / 공사 대화별 발화 군집화 및 센트로이드 계산
- [x] 단위 테스트(`*_test.go`) 및 벤치마크 작성

### Phase 5: 문서화 및 GitLab 원격 배포
- [x] `README.md` 작성: 수학적 배경, 3024차원 벡터 레이아웃 명세, Hugging Face 배포 가이드
- [x] 전체 빌드 및 테스트 검증 (`make test`)
- [x] GitLab `main` 브랜치 최초 커밋 및 푸시

---

## 6. MRL (Matryoshka Representation Learning) 차원별 활용 가이드

| 차원 (Dim) | 서브스페이스 구성 | 주요 용도 | 연산 효율성 |
|:---|:---|:---|:---|
| **512-d** | Literal 핵심 성분 | 초고속 Vector DB 1차 인덱싱, 대규모 유사 발화 필터링 | 최고 속도 (메모리 절약 극대화) |
| **1024-d** | Literal + Intent 요약 | 고객센터 의도 라우팅, FAQ 시맨틱 검색 | 속도와 의미 정확도의 균형 |
| **1536-d** | Literal 전 범위 | 표준 LLM 임베딩 호환 (OpenAI small 규격 호환) | 일반 텍스트 및 RAG 문서 검색 |
| **3024-d** | **Full Multi-Aspect (전체)** | **발화자 심층 의미, 화행 의도, 공사 관계, 반어/비꼼 분석** | **최고 정밀도 (심층 심리/대화 분석)** |
