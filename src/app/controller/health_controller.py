"""
health_controller.py

서버 헬스체크 요청을 처리하는 컨트롤러 모듈입니다.
app.state.ready 플래그를 확인하여 서버가 준비 완료된 경우에만 200을 반환합니다.
준비 중이면 503을 반환하여 Docker healthcheck가 실패하도록 합니다.
"""

from fastapi import HTTPException, Request


class HealthController:
    """
    헬스체크 요청을 처리하는 컨트롤러.

    lifespan에서 app.state.ready = True 로 설정된 이후에만 정상 응답합니다.
    Docker HEALTHCHECK + start_period 조합으로 초기화 완료 전 트래픽 유입을 방지합니다.
    """

    async def health(self, request: Request) -> dict[str, str]:
        """
        GET /health — 서버 준비 상태를 반환합니다.

        Args:
            request: FastAPI Request (app.state 접근용)

        Returns:
            {"status": "ok"} — 서버가 준비 완료된 경우

        Raises:
            HTTPException(503): 서버가 아직 초기화 중인 경우
        """
        ready: bool = getattr(request.app.state, "ready", False)
        if not ready:
            raise HTTPException(status_code=503, detail="Service is initializing")
        return {"status": "ok"}


def get_health_controller() -> HealthController:
    """
    HealthController 인스턴스를 반환하는 의존성 팩토리 함수.
    """
    return HealthController()
