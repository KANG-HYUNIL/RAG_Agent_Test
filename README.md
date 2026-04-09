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

[문제이해](docs/260406_01_최초문제이해.md)

---

### 260406_데이터 구조 파악

[데이터 구조](docs/260406_02_데이터구조파악.md)

---

### 260406_청킹 전략 고민

[임베딩?](docs/260406_03_청킹과임베딩고민.md)

---

### 260407_유사도 검색 고민

[유사도검색 순위?](docs/260407_01_유사도검색고민.md)

---

### 260407_LLM User Prompt 방식 고민

[프롬프트설계?](docs/260407_02_프롬프트고민.md)

---

### 260407_실제 실험(훈련) 진행 순서

[1차 실험](docs/260407_03_1차실험.md)

---

### 260407_제2차 실험 진행 순서

[2차 실험](docs/260407_04_2차실험.md)

---

### 260408_제3차 실험 진행

[3차 실험](docs/260408_01_3차실험.md)

---

### 260408_제4차 실험(4차-1 Clean Confirmation) 진행 순서

[4차 실험](docs/260408_02_4차실험.md)

---


### 260408_제5차 실험_Retrieval 강화? 


[5차 실험](docs/260408_03_5차실험.md)

---

### 260408_제6차실험

[6차 실험](docs/260408_04_6차실험.md)

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
