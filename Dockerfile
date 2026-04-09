# ─────────────────────────────────────────────────────────────
# 빌드 단계: uv로 의존성을 .venv 폴더에 설치
# ─────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ─────────────────────────────────────────────────────────────
# 실행 단계: 최소 런타임 이미지
# ─────────────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

WORKDIR /app

# 빌드 단계의 .venv만 복사 (소스는 별도)
COPY --from=builder /app/.venv /app/.venv

# 소스 복사
# src/ 내용을 /app/src/ 에 두고, PYTHONPATH=/app/src 로 설정해
# `from app.server import ...`, `from config.config import ...` 가 정상 동작합니다.
COPY src/ /app/src/
COPY data/ /app/data/
COPY configs/ /app/configs/

# 가상환경 우선 실행
ENV PATH="/app/.venv/bin:$PATH"

# /app/src 를 모듈 루트로 설정
# → app/, config/, agent/ 패키지가 최상위로 인식됨
ENV PYTHONPATH="/app/src"

# healthcheck용 curl 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# 컨테이너 헬스체크:
#   - interval: 10초마다 체크
#   - timeout: 응답 대기 5초
#   - start-period: 서버 시작 후 30초 동안은 실패해도 unhealthy로 간주 안 함
#     (RAG 파이프라인 초기화 시 더 길게 조정 필요 — 60~120s 권장)
#   - retries: 3회 연속 실패 시 unhealthy
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 메인 모듈을 통해 실행하여 main.py의 설정 초기화 로직이 작동하도록 함
# PYTHONPATH=/app/src 설정이 되어 있으므로 src.main 으로 접근 가능합니다.
CMD ["python", "-m", "main"]
