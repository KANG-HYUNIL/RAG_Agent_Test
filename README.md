# RAG_Agent_Test

## 디렉토리 구조

```text
.
├── app/
│   ├── api/          # FastAPI 엔드포인트 정의
│   ├── core/         # 설정 및 RAG 파이프라인 로직
│   ├── models/       # Pydantic DTO 모델
│   └── main.py       # 서버 진입점
├── data/             # 법률 데이터셋
├── tests/            # 평가 스크립트
├── Dockerfile        # 컨테이너 설정
├── pyproject.toml    # uv 기반 의존성 관리
└── Makefile          # 개발 도구 명령어
```

## 초기 Agent System 구축 및 평가 스크립트 실행 방법

1. **환경 설정**: `uv sync`를 통해 의존성을 설치합니다.
2. **서버 실행**: `make run` (또는 `docker build` 후 `docker run`)을 통해 서버를 실행합니다.
3. **평가 실행**: `python tests/evaluate.py`를 통해 dev set에 대한 성능을 측정합니다.

## Agent System 구조

데이터 로드 > 청킹 > 임베딩 > 유사도 서치 > 프롬프트 생성 > 검색 > LLM 추론(A/B/C/D)

## Inference 서버 실행 방법

- `docker build -t rag-agent .`
- `docker run -p 8000:8000 rag-agent`
- `/docs` 경로에서 Swagger UI를 통해 API 테스트 가능

## 단계별 사고 및 전략, 조사

### 260406_최초 문제 이해

#### 목표
- 법률 객관식 문제를 입력받아 `A / B / C / D` 중 하나를 반환하는 RAG Agent System 구성
- FastAPI 기반 inference 서버 구성 및 dev set 평가 코드 작성

#### 요구사항
- Python 3.13, uv, ruff, pyright 적용
- `make lint`, `make format`, `make type-check` 통과
- Docker 빌드 및 실행 가능 (10분 이내 완료 조건)

#### API 조건
- 입력: `query: str`
- 출력: `answer: "A" | "B" | "C" | "D"`

#### 제약사항
- 모델: `gpt-4o-mini`, `text-embedding-3-small`
- 제출물: 200MB 이하, API Key 미포함

### 260406_에이전트 방식 결정: Tool-calling vs 고정형 RAG

#### 검토한 방식
| 방식 | 특징 | 이 과제 적합성 |
|:---|:---|:---|
| **Tool-calling Agent** | LLM이 스스로 검색 도구를 호출할지 판단 | 불필요한 복잡성 증가 |
| **고정형 RAG (Non-tool)** | 검색 → 컨텍스트 구성 → 생성이 코드 레벨에서 순서 고정 | A/B/C/D 고정 출력에 최적, 속도·안정성 우위 |

#### 결정
고정형 RAG 방식 채택. 입력과 출력 형식이 고정되어 있어 LLM의 자율적 판단보다 파이프라인의 결정론적 제어가 성능과 비용 면에서 유리함.

### 260406_참고 자료
- [Korean Law MCP](https://github.com/chrisryugj/korean-law-mcp) : 법률 데이터 처리 및 RAG 구현 참조용 레포지토리