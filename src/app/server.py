"""
server.py

FastAPI 애플리케이션 인스턴스를 생성하고 라우터를 등록하는 모듈입니다.
create_app() 팩토리 함수를 통해 애플리케이션을 구성하며,
lifespan 이벤트 핸들러로 시작/종료 시 처리를 관리합니다.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.router.inference_router import router as inference_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 애플리케이션 생명주기(lifespan) 관리 컨텍스트 매니저.

    서버 시작 시(yield 이전)와 종료 시(yield 이후)에 실행할
    초기화/정리 작업을 정의합니다.

    Args:
        app: FastAPI 애플리케이션 인스턴스

    Yields:
        None
    """
    # --- 서버 시작 시 실행 ---
    # TODO: 벡터 스토어 초기화, train.csv 인덱싱 등 시작 시 처리 추가 예정
    print("[startup] RAG Agent 서버가 시작됩니다.")

    yield

    # --- 서버 종료 시 실행 ---
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
    # FastAPI 인스턴스를 생성합니다.
    app = FastAPI(
        title="RAG Agent API",
        description=(
            "한국 법률 객관식 문제를 입력받아 A/B/C/D 정답을 반환하는 RAG Agent 추론 서버입니다. "
            "text-embedding-3-small 임베딩 + gpt-4o-mini 생성 파이프라인을 사용합니다."
        ),
        version="0.1.0",
        # lifespan 이벤트 핸들러를 등록합니다.
        lifespan=lifespan,
    )

    # 추론 라우터를 루트 경로에 마운트합니다.
    # POST / 가 최종적으로 이 라우터를 통해 처리됩니다.
    app.include_router(inference_router)

    return app
