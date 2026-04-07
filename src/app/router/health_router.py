"""
health_router.py

서버 헬스체크 엔드포인트를 등록하는 라우터 모듈입니다.
Docker HEALTHCHECK 및 외부 모니터링 도구가 이 경로를 호출합니다.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.controller.health_controller import HealthController, get_health_controller

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="헬스체크",
    description=(
        "서버의 준비 상태를 반환합니다. "
        "초기화(lifespan startup) 완료 전에는 503을 반환하며, "
        '완료 후에는 200 {"status": "ok"}를 반환합니다.'
    ),
)
async def health(
    request: Request,
    controller: Annotated[HealthController, Depends(get_health_controller)],
) -> dict[str, str]:
    """
    GET /health — 서버 준비 상태 확인.

    Args:
        request: FastAPI Request (app.state 접근용)
        controller: 의존성 주입된 HealthController 인스턴스

    Returns:
        {"status": "ok"} 또는 HTTPException(503)
    """
    return await controller.health(request=request)
