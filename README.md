# RAG_Agent_Test


## 디렉토리 구조

```text

```

## 260406_최초 문제 이해

### 목표
- 법률 객관식 문제를 입력받아 `A / B / C / D` 중 하나를 반환하는 RAG Agent System 구성
- FastAPI 기반 inference 서버 구성
- 구축한 서버를 호출하는 dev set 평가 코드 작성 포함

### 요구사항
- Python 3.13 사용
- uv 사용
- ruff, pyright 적용
- `make lint`, `make format`, `make type-check` 통과
- Docker에서 빌드 및 실행 가능해야 함
- `docker run` 시 inference 서버가 떠야 함
- inference는 healthy 상태 진입 후 10분 이내 완료 조건 고려
- README에 문제 이해, 구조, 실행 방법, 평가 방법, dev 성능 정리

### API 조건
- 입력: `query: str`
- 출력: `answer: "A" | "B" | "C" | "D"`

### 제약사항
- 생성 모델: `gpt-4o-mini`
- 임베딩 모델: `text-embedding-3-small`
- 제출 압축 파일 200MB 이하
- API Key 제출 금지

### 현재 단계 방향
- 우선 FastAPI + Docker 기반 실행 구조부터 고정
- inference API DTO와 서버 실행 경로 먼저 정리
- 이후 RAG 내부 구성요소를 단계적으로 연결


