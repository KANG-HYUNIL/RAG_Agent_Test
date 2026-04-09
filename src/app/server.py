"""
server.py

FastAPI 애플리케이션 인스턴스를 생성하고 라우터를 등록하는 모듈입니다.
create_app() 팩토리 함수를 통해 애플리케이션을 구성하며,
lifespan 이벤트 핸들러로 시작/종료 시 처리를 관리합니다.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.router.health_router import router as health_router
from app.router.inference_router import router as inference_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    FastAPI 애플리케이션 생명주기(lifespan) 관리 컨텍스트 매니저.

    서버 시작 시(yield 이전)와 종료 시(yield 이후)에 실행할
    초기화/정리 작업을 정의합니다.

    app.state.ready 플래그를 통해 초기화 완료 여부를 /health 엔드포인트에 노출합니다.
    Docker HEALTHCHECK + start_period 조합으로 준비 완료 전 트래픽 유입을 방지합니다.

    Args:
        app: FastAPI 애플리케이션 인스턴스

    Yields:
        None
    """
    # --- 서버 시작 시 실행 ---
    print("[startup] RAG Agent 서버가 시작됩니다.")

    from config.config import get_settings
    from agent.agent_core import LegalRAGAgent

    try:
        settings = get_settings()
        # RAG 파이프라인(DataLoader, Retriever, Index 등) 초기화
        app.state.agent = LegalRAGAgent(settings=settings)
        app.state.ready = True
        print(f"[startup] 초기화 완료. (전략: {settings.rag_retrieval_strategy} / {settings.rag_prompt_strategy})")
    except Exception as e:
        app.state.ready = False
        print(f"[startup] 초기화 실패: {e}")
        raise e

    yield

    # --- 서버 종료 시 실행 ---
    app.state.ready = False
    print("[shutdown] RAG Agent 서버가 종료됩니다.")


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고 설정합니다.

    팩토리 함수 패턴을 사용하여 테스트 시에도 동일한 방식으로
    앱 인스턴스를 생성할 수 있습니다.

    Returns:
        구성이 완료된 FastAPI 애플리케이션 인스턴스
    """
    app = FastAPI(
        title="RAG Agent API",
        description=(
            "한국 법률 객관식 문제를 입력받아 A/B/C/D 정답을 반환하는 RAG Agent 추론 서버입니다. "
            "text-embedding-3-small 임베딩 + gpt-4o-mini 생성 파이프라인을 사용합니다."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # 헬스체크 라우터: GET /health
    app.include_router(health_router)

    # 추론 라우터: POST /
    app.include_router(inference_router)

    return app
