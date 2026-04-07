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

    # TODO: 전략 레지스트리 기반 RAG 파이프라인 초기화
    #   (DataLoader → Chunker → Embedder → Retriever → FAISS 색인)
    #   전략 설정 방식 결정 후 여기에 구현 예정.
    #   실제 파이프라인 초기화 시 app.state.ready = True 를 초기화 완료 후로 이동할 것.

    app.state.ready = True  # 초기화 완료 → /health 가 200 반환
    print("[startup] 초기화 완료.")

    yield

    # --- 서버 종료 시 실행 ---
    app.state.ready = False
    # TODO: 리소스 정리(벡터 스토어 클리어 등) 추가 예정
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
