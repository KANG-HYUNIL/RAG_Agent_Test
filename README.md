# RAG_Agent_Test

## 디렉토리 구조

```text

```

## 초기 Agent System 구축 및 평가 스크립트 실행 방법

## Agent System 구조

```
[dev.csv 질문 입력]
      ↓
[query 임베딩 (text-embedding-3-small)]
      ↓
[train.csv 벡터 DB에서 코사인 유사도 검색 → Top-K 유사 문제 추출]
      ↓
[Few-shot Prompt 구성 (유사 문제+정답 예시 포함)]
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

#### 현재 단계 방향

- 우선 FastAPI + Docker 기반 실행 구조부터 고정
- inference API DTO와 서버 실행 경로 먼저 정리
- 이후 RAG 내부 구성요소를 단계적으로 연결

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


#### 구현 우선순위


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

5. Synthetic query expansion
- Doc2Query 계열처럼 retrieval 개선 여지 있음
- 다만 노이즈 query가 성능을 해칠 수 있어 후순위




### 260407_로컬FAISSVS Docker Qdrant

| **비교 항목** | **로컬 파이썬 Vector DB (FAISS)** | **Docker 컨테이너 Vector DB (Qdrant, Milvus)** |
| --- | --- | --- |
| **적재/실행 방식** | Python RAM 메모리에 곧바로 적재 후 실행 `(uv sync만으로 완료)` | Docker 데몬 스핀업 후 통신 연결 `(docker-compose up 필요)` |
| **속도 (Latency)** | 파이썬 앱 내에서 C++ 바이너리로 즉각 호출* | 컨테이너 간 통신을 거침 |
| **제출물 용량** | 제출 폴더에 영향을 주지 않으며, 설치 패키지도 30MB 안팎 | 스탠드얼론 이미지의 크기는 500MB~1GB 정도 |

로컬 FAISS(float 16) 채택, train.csv 크기가 그렇게까지 수천 row 수준이라, InMemory VectorDB 채용해도 괜찮겠다 판단.
또한 실제 현업 RAG 프로덕션에서는 서버 비용과 연산 속도를 위해 float16이나 int8 Quantization까지 도입하며 약간의 Recall 하락을 감수하기도 하니, 괜찮다고 봄.



### 260407_유사도 검색 고민

| 방법 | 설명 | 장점 | 단점 | 언제 쓰기 좋은가 | 현재 평가 | 근거 |
| --- | --- | --- | --- | --- | --- | --- |
| 고정 Top-k | query에 대해 가장 가까운 k개만 그대로 반환 | 구현 가장 단순, baseline으로 적합 | k가 작으면 recall 부족, 크면 noise 증가 | 첫 실험, 기준선 확보 | 높음 | FAISS의 기본 `search`는 “at most k vectors”를 반환하는 k-NN 검색이다. ([Faiss](https://faiss.ai/cpp_api/struct/structfaiss_1_1Index.html)) |
| Top-k + score threshold | 먼저 top-k를 뽑고, 그중 score 기준을 넘는 것만 유지 | irrelevant 문서 유입 감소, top-k 단점 일부 보완 | metric마다 threshold 의미가 다름. L2는 작을수록 좋고, IP/cosine은 클수록 좋음 | cosine/IP 기반 점수 해석이 안정적일 때 | 매우 높음 | LangChain도 retriever search type으로 `similarity_score_threshold`를 공식 지원한다.  |
| Top-k + MMR | top-k 후보에서 relevance와 diversity를 함께 고려해 다시 고름 | 비슷한 문서만 반복해서 들어오는 문제 완화 | λ 같은 하이퍼파라미터 추가, 계산량 증가 | 유사 문항이 몰려 나오는 경우 | 높음 | . ([LangChain](https://reference.langchain.com/v0.3/python/mongodb/utils/langchain_mongodb.utils.maximal_marginal_relevance.html?utm_source=chatgpt.com)) |
| Hybrid (dense + sparse) | FAISS dense 검색과 BM25/키워드 검색을 결합 | 의미 유사성과 정확한 용어 일치를 함께 반영 | 구현 복잡도 증가 | 법률처럼 전문 용어와 exact match가 중요한 경우 | 높음 |  ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)) |


#### 구축 및 실험 순서

1. 1차 baseline: 고정 Top-k
2. 2차 개선: Top-k + score threshold
3. 3차 개선: Top-k + MMR
4. 4차 개선: Hybrid dense + sparse

### 260407_LLM 전달 방식 고민


| 방법 | 핵심 구조 | 시스템 / 사용자 prompt 설계 | 장점 | 단점 | 언제 쓰기 좋은가 | 근거 |
| --- | --- | --- | --- | --- | --- | --- |
| 원문 그대로 stuffing | query와 retrieved chunk 원문을 순서대로 그대로 붙여 넣음 | system에는 역할·제약·근거 사용 규칙만 두고, user에는 질문 + retrieved docs block을 그대로 둠 | 구현 가장 단순, baseline으로 적합 | context가 길어질수록 noise와 position bias가 커짐 | 첫 실험, 기준선 확보 | RAG의 기본 형태는 retrieved docs를 prompt에 붙이는 방식이며, OpenAI는 system에 전역 지침, user에 작업 세부사항을 두라고 안내한다. ([OpenAI 개발자](https://developers.openai.com/api/docs/guides/prompting/?utm_source=chatgpt.com)) |
| 구조화 evidence packet | retrieved docs를 문서 ID, 제목, 점수, 핵심 내용 같은 필드로 정리해 전달 | system에는 “필드 외 추론 금지 / 근거 없는 답변 금지”를 두고, user에는 질문 + Evidence 1..N 블록을 넣음 | 필드 경계가 명확해 파싱·디버깅 쉬움, 답변 형식 통제 쉬움 | 자연어 연속성이 약해질 수 있음 | 객관식, 정답/근거 추적이 중요한 과제 | OpenAI는 YAML/불릿 등 스캔하기 쉬운 구조와 Structured Outputs 사용을 권장한다. ([OpenAI 개발자](https://developers.openai.com/api/docs/guides/prompting/?utm_source=chatgpt.com)) |
| 위치 최적화 ordering | 가장 중요한 근거를 앞이나 끝에 배치하고, 덜 중요한 근거를 중간에 둠 | system에는 “상위 근거 우선 사용” 규칙, user에는 relevance 순서가 드러나는 문서 배열 사용 | long context에서 핵심 정보 활용률을 높일 수 있음 | ordering 규칙이 잘못되면 오히려 성능 저하 가능 | retrieved docs가 여러 개이고 길이가 길 때 | Lost in the Middle은 관련 정보가 긴 컨텍스트의 중간에 있을 때 성능이 떨어질 수 있음을 보여준다. ([ACL Anthology](https://aclanthology.org/2024.tacl-1.9/?utm_source=chatgpt.com)) |
| contextualized chunk | 각 chunk 앞에 짧은 설명문, 헤더, 출처 요약을 붙여 chunk 의미를 보강 | system에는 “설명문은 맥락용, 답은 원문 근거 기반” 규칙, user에는 `[context note] + raw chunk` 형태로 전달 | chunk가 단독으로도 의미를 가지기 쉬워짐 | 전처리 비용 증가, 부정확한 설명문이 노이즈가 될 수 있음 | chunk 단위 검색에서 문맥 손실이 큰 경우 | Anthropic의 Contextual Retrieval은 chunk 앞에 contextual information을 추가해 retrieval 정확도를 높이는 방법을 제안한다. ([Anthropic](https://www.anthropic.com/news/contextual-retrieval?utm_source=chatgpt.com)) |
| compress / summarize then pass | retrieved docs를 그대로 넣지 않고, 추출 요약 또는 생성 요약으로 압축해 전달 | system에는 “요약된 evidence만 사용하되, 불확실하면 보수적으로 판단” 규칙, user에는 질문 + compressed evidence를 넣음 | 토큰 절감, long-context 부담 감소 | 압축 과정에서 근거 누락 또는 왜곡 위험 | 문서가 길고 비용/지연이 민감할 때 | RECOMP는 retrieved docs를 summary로 압축해 prepend하는 방식이 비용을 줄이고 성능을 유지 또는 개선할 수 있다고 보고한다. LongLLMLingua도 key information density를 높이는 압축의 효과를 제시한다. ([arXiv](https://arxiv.org/abs/2310.04408?utm_source=chatgpt.com)) |
| selective augmentation | retrieval 결과가 약하거나 무관하면 일부 또는 전부를 아예 전달하지 않음 | system에는 “유의미한 근거가 없으면 근거 부족 상태로 답하라”를 두고, user에는 빈 evidence도 허용 | irrelevant context로 인한 오답 유발을 줄일 수 있음 | retrieval 품질 판단 로직이 필요함 | 애매한 query, 노이즈 retrieval이 많은 경우 | RECOMP는 irrelevant retrieval에 대해 empty string을 반환하는 selective augmentation을 포함한다. ([arXiv](https://arxiv.org/abs/2310.04408?utm_source=chatgpt.com)) |
| multilingual normalization / translation | query와 retrieved docs의 언어를 맞추거나, 원문+번역문을 함께 전달 | system에는 “질문 언어/출력 언어/근거 언어 처리 규칙”을 두고, user에는 질문 + 원문 evidence + 필요 시 번역 evidence를 함께 제공 | 언어 불일치 문제를 줄일 수 있음 | 번역 품질이 낮으면 사실 왜곡 가능 | query 언어와 문서 언어가 다른 경우 | 다국어 RAG 연구는 query translation보다 multilingual retrieval이나 retrieved docs translation이 더 안정적일 수 있음을 보고했고, CrossRAG는 retrieved docs를 공통 언어로 번역해 generation에 전달하는 접근을 제안했다. ([arXiv](https://arxiv.org/html/2504.03616v1?utm_source=chatgpt.com)) |
| few-shot answer envelope | system/user 안에 “질문-근거-정답 형식”의 소수 예시를 함께 넣어 응답 양식을 고정 | system에는 역할·판정 원칙, user에는 현재 질문 + retrieved evidence + 1~3개 예시를 넣음 | 객관식 답변 형식, 근거 인용 형식을 더 안정적으로 맞출 수 있음 | prompt 길이 증가 | 출력 형식 일관성이 중요한 경우 | OpenAI와 Azure 문서는 few-shot examples가 작업 형식과 답변 패턴을 고정하는 데 유용하다고 설명한다. ([OpenAI Help Center](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api?utm_source=chatgpt.com)) |
| citation-constrained structured answer | LLM 출력 자체를 `정답`, `근거 문서 ID`, `근거 요약` 같은 schema로 강제 | system에는 “반드시 지정 schema로 반환”을 두고, user에는 질문 + evidence packet을 넣음 | 평가, 디버깅, 근거 추적이 쉬움 | 모델 자유도가 줄어 자연어 답변 유연성 감소 | 객관식 채점, 자동 평가, 로그 분석이 중요한 경우 | OpenAI Structured Outputs는 JSON Schema를 강제해 enum, 필수 키 등을 안정적으로 제어할 수 있다고 설명한다. ([OpenAI 개발자](https://developers.openai.com/api/docs/guides/structured-outputs/?utm_source=chatgpt.com)) |


---

#### 구축 및 실험 순서

1. 원문 stuffing
2. 구조화 evidence packet
3. 위치 최적화 ordering
4. compress / summarize then pass
5. citation-constrained structured answer

### 참고 자료 및 외부 조사



#### 한국어 법률 NLP 및 RAG 관련

| 자료 | 링크 | 핵심 내용 |
|:---|:---|:---|
| korean-law-mcp (GitHub) | [링크](https://github.com/chrisryugj/korean-law-mcp) | 법제처 Open API 기반 89개 법령 검색 도구. MCP Server 형태로 법률 RAG 시스템 설계 시 참고. |
| KBL (Korean Benchmark for Legal LLM) | [링크](https://arxiv.org/abs/) | 한국 법률 QA 벤치마크. 사법시험 기반 문제 포함. 본 과제 데이터셋과 유사한 구조. |
| LBOX OPEN (NeurIPS 2022) | [링크](https://neurips.cc/) | 대규모 한국 법률 데이터셋. 분류·판결 예측·요약 포함. 법률 RAG 코퍼스 구성 참고. |
| LRAGE (Legal RAG Evaluation Tool) | [링크](https://arxiv.org/abs/) | 법률 도메인 RAG 시스템 평가 전용 도구. 검색 정확도 및 생성 충실도 측정. |
| ACL Anthology (한국 법률 계층적 세그먼테이션) | [링크](https://aclanthology.org/) | 법령·판례에서 3단계 계층 분리(조-항-목)가 검색 정확도에 미치는 영향 분석. |
