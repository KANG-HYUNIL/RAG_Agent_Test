# 빌드 단계: 모든 의존성 설치
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# (선택) uv 캐시 활용을 위한 설정
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 설정 파일 복사
COPY pyproject.toml uv.lock ./

# 의존성 설치 (.venv 폴더에 설치됨)
RUN uv sync --frozen --no-install-project --no-dev

# ---------------------------------------------------------
# 실행 단계: 가벼운 런타임 구성
FROM python:3.13-slim-bookworm

WORKDIR /app

# 빌드 단계의 설치 결과물(.venv)만 가져옴
COPY --from=builder /app/.venv /app/.venv

# 모든 소스(src 패키지 및 과제 데이터 폴더) 복사
COPY src/ /app/src/
COPY ai-assigment+/ /app/ai-assigment+/

# 가상환경 바이너리를 PATH의 맨 앞에 두어 우선 사용하게 함
ENV PATH="/app/.venv/bin:$PATH"

# [중요] /app을 PYTHONPATH에 포함시켜 src 패키지가 root가 되도록 함
ENV PYTHONPATH="/app"

# API 실행 포트
EXPOSE 8000

# 모듈 방식(-m) 실행으로 패키지 임포트 유연성 확보
CMD ["python", "-m", "src.main"]
