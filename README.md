# RAG_Agent_Test

## 디렉토리 구조

```text
RAG_Agent_Test/
├── .env                          # 환경변수 (OPENAI_API_KEY 등, git 제외)
├── .gitignore
├── .python-version               # Python 3.13
├── CLAUDE.md
├── Dockerfile
├── Makefile                      # lint / format / type-check 단축 명령
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── pyrightconfig.json
├── ruff.toml
├── uv.lock
├── configs/                      # Hydra 설정 그룹 (benchmark 실험용)
│   ├── config.yaml               # 기본값 (serialization=kv_pairs, retrieval=top_k, prompt=raw_stuffing, query_representation=question_only)
│   ├── prompt/
│   │   ├── few_shot_envelope.yaml
│   │   ├── labeled_context.yaml
│   │   ├── raw_stuffing.yaml
│   │   └── structured_context.yaml
│   ├── query_representation/
│   │   ├── question_only.yaml        # baseline
│   │   └── question_plus_choices.yaml
│   ├── retrieval/
│   │   ├── hybrid.yaml               # placeholder (미구현)
│   │   ├── mmr.yaml
│   │   ├── score_threshold.yaml
│   │   ├── top_k.yaml
│   │   └── top_k_category_filter.yaml
│   └── serialization/
│       ├── dual.yaml
│       ├── kv_pairs.yaml
│       ├── kv_pairs_no_category.yaml  # Category ablation
│       ├── narrative.yaml
│       ├── raw.yaml
│       └── weighted.yaml
├── data/
│   ├── dev.csv                   # 평가용 질문 데이터 (Law 230개, Criminal Law 29개)
│   └── train.csv                 # 벡터 DB 색인용 학습 데이터 (Law 1834개, Criminal Law 239개)
├── outputs/                      # benchmark 실험 결과 CSV/JSON (자동 생성)
├── src/                          # 서버 & 에이전트 소스 루트
│   ├── __init__.py
│   ├── main.py                   # uvicorn 엔트리포인트
│   ├── agent/                    # RAG 파이프라인 컴포넌트
│   │   ├── chunker.py
│   │   ├── data_loader.py
│   │   ├── embedder/
│   │   │   ├── _registry.py
│   │   │   ├── embedder.py
│   │   │   ├── strategy_dual_representation.py
│   │   │   ├── strategy_field_weighted_kv.py
│   │   │   ├── strategy_kv_pairs.py
│   │   │   ├── strategy_narrativized_lite.py
│   │   │   └── strategy_raw.py
│   │   ├── query_encoder/
│   │   │   ├── __init__.py
│   │   │   └── query_encoder.py  # build_query_text(method, question, choices)
│   │   ├── prompt_builder/
│   │   │   ├── _registry.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── strategy_few_shot_envelope.py
│   │   │   ├── strategy_labeled_context.py
│   │   │   ├── strategy_raw_stuffing.py
│   │   │   └── strategy_structured_context.py
│   │   └── retriever/
│   │       ├── _registry.py
│   │       ├── retriever.py
│   │       ├── strategy_hybrid.py
│   │       ├── strategy_mmr.py
│   │       ├── strategy_score_threshold.py
│   │       └── strategy_top_k.py
│   ├── app/                      # FastAPI 애플리케이션 레이어
│   │   ├── server.py             # create_app 팩토리, lifespan
│   │   ├── controller/
│   │   │   ├── health_controller.py
│   │   │   └── inference_controller.py
│   │   ├── router/
│   │   │   ├── health_router.py  # GET /health
│   │   │   └── inference_router.py  # POST /
│   │   ├── service/
│   │   │   └── openai_service.py
│   │   └── types/
│   │       └── inference_dto.py
│   └── config/
│       └── config.py             # Settings 싱글턴 (환경변수 기반)
└── test/
    ├── benchmark.py              # Hydra 기반 단일 실험 실행
    └── oaat_sweep.py             # One-Axis-At-a-Time 배치 실험 (4축 13개)
```

## 초기 Agent System 구축 및 평가 스크립트 실행 방법

## Agent System 구조

```
[dev.csv 질문 입력]
      ↓
[query 임베딩 (text-embedding-3-small)]
      ↓
[train.csv 벡터 DB에서 코사인 유사도 검색]
      ↓
[Prompt 구성]
      ↓
[gpt-4o-mini 추론]
      ↓
[A / B / C / D 출력]
```

## Inference 서버 실행 방법

## 단계별 사고 및 전략, 조사

### 260406_최초 문제 이해

#### 목표

- 법률 객관식 문제를 입력받아 `A / B / C / D` 중 하나를 반환하는 RAG Agent System 구성
- FastAPI 기반 inference 서버 구성
- 구축한 서버를 호출하는 dev set 평가 코드 작성 포함

#### 요구사항

- Python 3.13, uv, ruff, pyright 적용
- `make lint`, `make format`, `make type-check` 통과
- Docker 빌드 및 실행 가능 (healthy 상태 진입 후 10분 이내 완료 조건)
- README에 문제 이해, 구조, 실행 방법, 평가 방법, dev 성능 정리

#### API 조건

- 입력: `query: str`
- 출력: `answer: "A" | "B" | "C" | "D"`

#### 제약사항

- 생성 모델: `gpt-4o-mini`
- 임베딩 모델: `text-embedding-3-small`
- 제출 압축 파일 200MB 이하
- API Key 제출 금지


---

### 260406_데이터 구조 파악

#### 전체 개요

| 항목 | train.csv | dev.csv |
| --- | --- | --- |
| 행 수 | 2,073 | 259 |
| 열 수 | 8 | 8 |
| 결측치 | 0 | 0 |
| question 중복 | 0 | 0 |
| train-dev 간 question 중복 | - | 0 |

| 컬럼명 | 의미 | 비고 |
| --- | --- | --- |
| `question` | 문제 본문 | 한국어 객관식 문제 |
| `answer` | 정답 번호 | `1, 2, 3, 4` 형식 |
| `A` | 선택지 A | 문자열 |
| `B` | 선택지 B | 문자열 |
| `C` | 선택지 C | 문자열 |
| `D` | 선택지 D | 문자열 |
| `Category` | 법률 분야 | `Law`, `Criminal Law` |
| `Human Accuracy` | 인간 정답률 | `0.0 ~ 1.0` 실수 |

| 항목 | 내용 |
| --- | --- |
| 정답 형식 | `A/B/C/D`가 아니라 `1/2/3/4`로 저장됨 |
| 카테고리 수 | 2개 (`Law`, `Criminal Law`) |
| 텍스트 언어 | 문제/선택지는 한국어, Category는 영어 |
| 결측치 | 없음 |
| 질문 중복 | train, dev 내부 및 상호 간 중복 없음 |

#### train.csv

| answer | 개수 |
| --- | --- |
| 1 | 480 |
| 2 | 509 |
| 3 | 538 |
| 4 | 546 |

| Category | 개수 |
| --- | --- |
| Law | 1,834 |
| Criminal Law | 239 |


#### dev.csv

| answer | 개수 |
| --- | --- |
| 1 | 65 |
| 2 | 63 |
| 3 | 51 |
| 4 | 80 |

| Category | 개수 |
| --- | --- |
| Law | 230 |
| Criminal Law | 29 |

#### Human Accuracy 정보

| 항목 | train.csv | dev.csv |
| --- | --- | --- |
| 평균 | 0.4474 | 0.4568 |
| 중앙값 | 0.5795 | 0.5959 |
| 최소값 | 0.0 | 0.0 |
| 최대값 | 1.0 | 1.0 |

### 260406_에이전트 방식 결정: Tool-calling vs 고정형 RAG

#### 검토한 방식

| 방식 | 특징 | 이 과제 적합성 |
|:---|:---|:---|
| **Tool-calling Agent** | LLM이 스스로 검색 도구를 호출할지 판단 |  출력 형식이 고정돼 있어 불필요한 복잡성 증가 |
| **고정형 RAG (Non-tool)** | 검색 → 컨텍스트 구성 → 생성이 코드 레벨에서 순서 고정 |  A/B/C/D 고정 출력에 최적, 속도·안정성 우위 |

#### 결정

**고정형 RAG 방식 채택.**

이 과제는 입력(법률 객관식 문제)과 출력(A/B/C/D) 형식이 완전히 고정되어 있어, LLM의 자율적 판단이 필요 없다.
파이프라인을 코드 레벨에서 결정론적으로 제어하면 속도, 비용, 안정성 모든 면에서 유리하다.

---

### 260406_데이터셋 분석

#### 데이터셋 구조 파악

`train.csv` 및 `dev.csv`의 각 Row는 아래 필드로 구성된다.

| 필드 | 설명 |
|:---|:---|
| `question` | 법률 객관식 문제 본문 |
| `A` / `B` / `C` / `D` | 선택지 4개 |
| `answer` | 정답 (`A`, `B`, `C`, `D` 중 하나) |
| `Category` | 법률 분야 카테고리 (예: 민법, 형법 등) |
| `Human Accuracy` | 해당 문제에 대한 인간 정답률 (0.0 ~ 1.0) |

---

### 260406_청킹 전략 고민

#### 청킹 단위: Row 단위 고정

각 CSV Row를 하나의 청크(Chunk)로 취급한다.
법률 객관식 문제는 각 Row가 독립적인 의미 단위이므로, row 간 문맥 연속성이 없어 row 단위 분리가 자연스럽다.

#### Embedding Vector 변환 전략 비교

- A. corpus-side serialization, 
- B. chunking / metadata enrichment, 
- C. query / document expansion

| 축 | 전략 | 정의 | 장점 | 단점 | 현재 평가 |
|---|---|---|---|---|---|
| A | Raw CSV | CSV row를 거의 원문 그대로 문자열화하여 임베딩 | 구현 가장 단순, baseline 만들기 쉬움 | 필드 구조가 흐려지고 구분자 노이즈가 남기 쉬움 | baseline용으로만 유지 |
| A | JSON | JSON 구조를 유지한 채 직렬화하여 임베딩 | 필드 경계와 구조 보존이 쉬움 | 괄호·키 이름 비중이 커져 자연어 질의와 거리감이 생길 수 있음 | 구조 보존형 baseline으로 유지 |
| A | KV Pairs | `질문: ... / 선택지A: ...` 형태로 필드명을 붙여 직렬화 | 구조 보존력 높음, 구현 단순, 디버깅 쉬움 | 문장 자연스러움은 낮음 | 1차 유력 후보 |
| A | Narrativized | row를 자연어 문장 형태로 풀어 써서 직렬화 | 질의가 자연어일 때 정렬이 잘 될 가능성 높음 | 토큰 길이 증가, 변환 설계 품질에 민감 | 1차 유력 후보 |
| A | Field-Weighted | 중요한 필드를 앞에 두거나 더 자세히 넣는 가중 직렬화 | 질문·선택지 중심 검색 강화 가능 | 설계가 과하면 오히려 편향 생김 | 1차 유력 후보 |
| A | Dual Representation | 같은 row를 두 가지 표현으로 각각 색인 | 단일 표현의 약점을 보완 가능 | 색인·검색 복잡도 증가 | 2차 유력 후보 |
| B | Header-Seg | 헤더·섹션 정보와 함께 chunk를 나누는 방식 | 부분 검색 시 맥락 손실 감소 | 반복 정보가 늘어날 수 있음 | serialization이 아니라 chunking 축으로 관리 |
| B | Semantic Tag | 과목·주제·영역 태그를 앞이나 메타데이터에 부여 | 카테고리 신호를 보강할 수 있음 | 태그가 부정확하거나 과하면 noise 증가 | enrichment 전략으로 소량만 권장 |
| B | Contextual Preamble | chunk 앞에 짧은 설명 문장을 덧붙여 맥락 보강 | 문맥 없는 chunk의 의미 보완 가능 | 전처리 비용 증가, 잘못 쓰면 장황해짐 | contextual retrieval 실험 후보 |
| C | HyDE | 질의에서 가상 문서를 생성한 뒤 그 임베딩으로 검색 | zero-shot dense retrieval 강화 가능 | 질의마다 추가 LLM 호출 필요 | 후속 query-side 실험안 |
| C | Synthetic Query Expansion | 문서가 답할 법한 예상 질문을 생성해 문서에 붙임 | query-document 어휘 불일치 완화 가능 | 전처리 비용 증가, 생성 품질 영향 큼 | 2차 실험 후보 |
| C | Doc2Query | 문서별 pseudo-query를 생성해 색인 전 문서 확장 | 검색 recall 개선 가능 | 노이즈 query가 붙으면 역효과 가능 | document expansion 계열 대표 후보 |



#### 우선 구현 우선순위


1. KV Pairs
- 가장 단순하고 구조 보존력이 높음
- baseline으로 적합
- 구조화 데이터는 입력 형식에 민감하다는 연구와 부합

2. Narrativized-lite
- 자연어 질의와 정렬 가능성 확인
- 길이 증가와 변환 민감성 때문에 KV 다음

3. Field-Weighted KV
- content order가 성능에 영향을 줄 수 있으므로 저비용으로 시도 가치 큼

4. Dual Representation
- 단일 포맷 실험 후 복수 포맷 결합
- 성능 가능성은 있지만 복잡도 증가

5. Raw row




---


### 260407_로컬FAISSVS Docker Qdrant

| **비교 항목** | **로컬 파이썬 Vector DB (FAISS)** | **Docker 컨테이너 Vector DB (Qdrant, Milvus)** |
| --- | --- | --- |
| **적재/실행 방식** | Python RAM 메모리에 곧바로 적재 후 실행 `(uv sync만으로 완료)` | Docker 데몬 스핀업 후 통신 연결 `(docker-compose up 필요)` |
| **속도 (Latency)** | 파이썬 앱 내에서 C++ 바이너리로 즉각 호출* | 컨테이너 간 통신을 거침 |
| **제출물 용량** | 제출 폴더에 영향을 주지 않으며, 설치 패키지도 30MB 안팎 | 스탠드얼론 이미지의 크기는 500MB~1GB 정도 |

로컬 FAISS(float 16) 채택, train.csv 크기가 그렇게까지 수천 row 수준이라, InMemory VectorDB 채용해도 괜찮겠다 판단.
또한 실제 현업 RAG 프로덕션에서는 서버 비용과 연산 속도를 위해 float16이나 int8 Quantization까지 도입하며 약간의 Recall 하락을 감수하기도 하니, 괜찮다고 봄.

---

### 260407_유사도 검색 고민

| 방법 | 설명 | 장점 | 단점 | 언제 쓰기 좋은가 | 현재 평가 | 근거 |
| --- | --- | --- | --- | --- | --- | --- |
| 고정 Top-k | query에 대해 가장 가까운 k개만 그대로 반환 | 구현 가장 단순, baseline으로 적합 | k가 작으면 recall 부족, 크면 noise 증가 | 첫 실험, 기준선 확보 | 높음 | FAISS의 기본 `search`는 “at most k vectors”를 반환하는 k-NN 검색이다. ([Faiss](https://faiss.ai/cpp_api/struct/structfaiss_1_1Index.html)) |
| Top-k + score threshold | 먼저 top-k를 뽑고, 그중 score 기준을 넘는 것만 유지 | irrelevant 문서 유입 감소, top-k 단점 일부 보완 | metric마다 threshold 의미가 다름. L2는 작을수록 좋고, IP/cosine은 클수록 좋음 | cosine/IP 기반 점수 해석이 안정적일 때 | 매우 높음 | LangChain도 retriever search type으로 `similarity_score_threshold`를 공식 지원한다.  |
| Top-k + MMR | top-k 후보에서 relevance와 diversity를 함께 고려해 다시 고름 | 비슷한 문서만 반복해서 들어오는 문제 완화 | λ 같은 하이퍼파라미터 추가, 계산량 증가 | 유사 문항이 몰려 나오는 경우 | 높음 | . ([LangChain](https://reference.langchain.com/v0.3/python/mongodb/utils/langchain_mongodb.utils.maximal_marginal_relevance.html?utm_source=chatgpt.com)) |
| Hybrid (dense + sparse) | FAISS dense 검색과 BM25/키워드 검색을 결합 | 의미 유사성과 정확한 용어 일치를 함께 반영 | 구현 복잡도 증가 | 법률처럼 전문 용어와 exact match가 중요한 경우 | 높음 |  ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)) |


#### 우선 구축 및 실험 순서

1. 1차 baseline: 고정 Top-k
2. 2차 개선: Top-k + score threshold
3. 3차 개선: Top-k + MMR


---

### 260407_LLM User Prompt 방식 고민

| 방법 | 전달 구조 | 코드상 핵심 구현 | 장점 | 단점 | 언제 쓰기 좋은가 | 현재 평가 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_stuffing | retrieval 결과 원문을 순서대로 그대로 이어 붙임 | `content_dict`를 `"key: value | ..."` 형태로 펼쳐 단순 연결 | 구현 가장 단순, baseline으로 적합 | 문서 경계가 약하고 구조가 납작함 | 첫 실험, 기준선 확보 |
| labeled_context | 원문은 유지하되 각 context 앞에 `[참고자료 #N]` 라벨 부착 | raw_stuffing과 동일한 내용 포맷 + 번호 라벨 추가 | 문서 경계가 명확해짐, 디버깅 쉬움 | 내용 구조 자체는 여전히 약함 | raw 대비 작은 개선 실험 | 가벼운 1차 후보 |
| structured_context | 각 context를 `[참고자료 #N]` 아래 `- 필드명: 값` 형태로 구조화 | `content_dict`를 줄 단위 key-value 블록으로 변환 | 필드 경계가 명확, LLM이 정보 구조를 읽기 쉬움 | 자연어 흐름이 약해지고 데이터 행처럼 보일 수 있음 | 객관식 / 표형 데이터 / 법률 QA처럼 필드가 분명할 때 | 유력 후보 |
| few_shot_envelope | retrieval 결과를 참고자료가 아니라 예시 문제-정답 데모로 변환 | context에서 `question`, `A~D`, `answer`를 꺼내 few-shot 블록 구성 | 출력 형식 안정화, 문제 풀이 패턴 유도 가능 | 예시 품질에 민감, retrieval 결과가 안 맞으면 오히려 혼동 가능 | 문제 유형이 반복적이고 예시 기반 유도가 잘 먹힐 때 | 실험 가치 높음 |

#### 우선 구축 및 실험 순서

1. 원문 stuffing
2. labeled_context
3. structured_context
4. few_shot_envelope

고정 System Prompt로 우선 수행, User Prompt 의 적합성 판단 후에 System Prompt 고정 작업 진행.

---

### 260407_실제 실험(훈련) 진행 순서

#### 목적

- baseline 1회 실행
- 세 축 중 한 축만 변경하는 OAAT(One-At-A-Time) 1차 실험 수행
- 결과를 CSV / JSON으로 자동 저장

#### 실험 방식

| 구분 | 내용 |
| --- | --- |
| 실험 방식 | OAAT (One-At-A-Time) |
| baseline 실행 | 1회 |
| 축별 실험 | serialization / retrieval / prompt / query_representation 중 1개만 변경 |
| 나머지 축 | baseline 고정 |
| 목적 | 각 축이 성능에 미치는 영향 분리 확인 |
| 결과 저장 | `outputs/oaat_sweep/`에 CSV, JSON 저장 |
| 개별 run 저장 | `outputs/oaat_sweep/{timestamp}/` 하위 Hydra run dir 저장 |

| 축 | baseline 값 | 비고 |
| --- | --- | --- |
| serialization | `kv_pairs` | exclude_fields: answer, Human Accuracy |
| retrieval | `top_k` | top_k=5, cosine sim |
| prompt | `raw_stuffing` | exclude_fields: answer, Human Accuracy |
| query_representation | `question_only` | 질문 텍스트만 query 임베딩에 사용 |

#### 실행 결과 파싱 항목

| 항목 | 설명 |
| --- | --- |
| `accuracy_pct` | 최종 정확도 |
| `correct` | 정답 수 |
| `total` | 전체 문항 수 |
| `total_time_s` | 전체 소요 시간 |
| `avg_time_s` | 문항당 평균 시간 |
| `status` | 실행 성공 / 실패 상태 |



#### OAAT 실험 기록표 (총 13개)

> 실행 명령: `python test/oaat_sweep.py --yes`

##### 전체 실험 요약표

| 실험명 | axis | serialization | retrieval | prompt | query_repr | accuracy(%) | correct/total | total_time(s) | avg_time(s/q) | status | 비고 |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| baseline | baseline | kv_pairs | top_k | raw_stuffing | question_only | 51.74 | 134/259 | 329.6 | 1.273 | ok | baseline |
| serialization__raw | serialization | raw | top_k | raw_stuffing | question_only | 51.35 | 133/259 | 377.6 | 1.458 | ok | answer 포함 오염 기준선 |
| serialization__narrative | serialization | narrative | top_k | raw_stuffing | question_only | 50.19 | 130/259 | 384.0 | 1.483 | ok |  |
| serialization__weighted | serialization | weighted | top_k | raw_stuffing | question_only | 50.58 | 131/259 | 338.9 | 1.309 | ok |  |
| serialization__dual | serialization | dual | top_k | raw_stuffing | question_only | 50.19 | 130/259 | 337.1 | 1.302 | ok |  |
| serialization__kv_pairs_no_category | serialization | kv_pairs_no_category | top_k | raw_stuffing | question_only | 49.42 | 128/259 | 344.6 | 1.331 | ok | Category ablation |
| retrieval__score_threshold | retrieval | kv_pairs | score_threshold | raw_stuffing | question_only | 53.28 | 138/259 | 353.9 | 1.367 | ok | baseline 대비 최고 개선 |
| retrieval__mmr | retrieval | kv_pairs | mmr | raw_stuffing | question_only | 50.97 | 132/259 | 366.6 | 1.415 | ok |  |
| retrieval__top_k_category_filter | retrieval | kv_pairs | top_k_category_filter | raw_stuffing | question_only | 50.97 | 132/259 | 337.6 | 1.304 | ok | Law / Criminal Law 도메인 분리 |
| prompt__labeled_context | prompt | kv_pairs | top_k | labeled_context | question_only | 50.97 | 132/259 | 321.9 | 1.243 | ok |  |
| prompt__structured_context | prompt | kv_pairs | top_k | structured_context | question_only | 51.74 | 134/259 | 405.3 | 1.565 | ok | baseline와 동일 |
| prompt__few_shot_envelope | prompt | kv_pairs | top_k | few_shot_envelope | question_only | 48.26 | 125/259 | 433.7 | 1.675 | ok | 성능 저하 |
| query_representation__question_plus_choices | query_representation | kv_pairs | top_k | raw_stuffing | question_plus_choices | 55.60 | 144/259 | 392.9 | 1.517 | ok | 선택지 결합 쿼리, 전체 최고 |

##### 축별 최고 후보 정리표

| 축 | baseline | 최고 후보 | accuracy(%) | baseline 대비 변화 | 채택 여부 | 비고 |
| --- | --- | --- | ---: | ---: | --- | --- |
| serialization | kv_pairs | 없음 | 51.74 | 0.00 | baseline 유지 | 모든 serialization 변형이 baseline 이하 |
| retrieval | top_k | score_threshold | 53.28 | +1.54 | 채택 후보 | retrieval 축 최고 성능 |
| prompt | raw_stuffing | structured_context | 51.74 | 0.00 | 보류 | baseline와 동일, 속도는 더 느림 |
| query_representation | question_only | question_plus_choices | 55.60 | +3.86 | 강력 채택 후보 | 전체 실험 중 최고 성능 |

##### 1차 실험 해석 요약

| 항목 | 해석 |
| --- | --- |
| baseline | `kv_pairs + top_k + raw_stuffing + question_only` 조합으로 51.74% 확보 |
| serialization 축 | baseline보다 나은 변형 없음. 현재 단계에서는 `kv_pairs` 유지가 합리적 |
| retrieval 축 | `score_threshold`가 가장 효과적. `mmr`, `category_filter`는 baseline 이하 |
| prompt 축 | `structured_context`는 baseline와 동일, `few_shot_envelope`는 성능 저하 |
| query 축 | `question_plus_choices`가 가장 큰 개선폭을 보임. 2차 실험 핵심 후보 |
| 2차 우선 조합 후보 | `kv_pairs + score_threshold + raw_stuffing(or structured_context) + question_plus_choices` |






---



---

### 참고 자료 및 외부 조사



#### 한국어 법률 NLP 및 RAG 관련

| 자료 | 링크 | 핵심 내용 |
|:---|:---|:---|
| korean-law-mcp (GitHub) | [링크](https://github.com/chrisryugj/korean-law-mcp) | 법제처 Open API 기반 89개 법령 검색 도구. MCP Server 형태로 법률 RAG 시스템 설계 시 참고. |
| KBL (Korean Benchmark for Legal LLM) | [링크](https://arxiv.org/abs/) | 한국 법률 QA 벤치마크. 사법시험 기반 문제 포함. 본 과제 데이터셋과 유사한 구조. |
| LBOX OPEN (NeurIPS 2022) | [링크](https://neurips.cc/) | 대규모 한국 법률 데이터셋. 분류·판결 예측·요약 포함. 법률 RAG 코퍼스 구성 참고. |
| LRAGE (Legal RAG Evaluation Tool) | [링크](https://arxiv.org/abs/) | 법률 도메인 RAG 시스템 평가 전용 도구. 검색 정확도 및 생성 충실도 측정. |
| ACL Anthology (한국 법률 계층적 세그먼테이션) | [링크](https://aclanthology.org/) | 법령·판례에서 3단계 계층 분리(조-항-목)가 검색 정확도에 미치는 영향 분석. |
