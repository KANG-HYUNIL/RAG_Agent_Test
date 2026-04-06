"""
inference_controller.py

추론(inference) 관련 HTTP 요청을 처리하는 컨트롤러 모듈입니다.
FastAPI의 Depends()를 통해 OpenAIService를 주입받아 사용합니다.
"""

from typing import Annotated

from fastapi import Depends

from app.service.openai_service import OpenAIService
from app.types.inference_dto import InferenceRequest, InferenceResponse
from config.config import Settings, get_settings


def get_openai_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OpenAIService:
    """
    OpenAIService 인스턴스를 생성하여 반환하는 의존성 팩토리 함수.

    FastAPI의 Depends()에서 호출되며, 요청마다 설정을 기반으로
    서비스 인스턴스를 제공합니다.

    Args:
        settings: get_settings()로부터 주입받은 애플리케이션 설정

    Returns:
        OpenAIService 인스턴스
    """
    return OpenAIService(settings=settings)


class InferenceController:
    """
    추론 관련 HTTP 요청을 처리하는 컨트롤러 클래스.

    라우터에서 호출되며, 요청 DTO를 서비스 레이어에 전달하고
    응답 DTO를 구성하여 반환하는 역할을 담당합니다.
    """

    def __init__(self, service: OpenAIService) -> None:
        """
        InferenceController 초기화.

        Args:
            service: OpenAIService 인스턴스 (Depends를 통해 주입)
        """
        # 서비스 인스턴스를 컨트롤러에 주입합니다.
        self._service: OpenAIService = service

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        추론 요청을 처리하고 결과를 InferenceResponse로 반환합니다.

        Args:
            request: 클라이언트로부터 수신한 추론 요청 DTO

        Returns:
            정답 선지(A/B/C/D)를 담은 InferenceResponse DTO

        Raises:
            HTTPException: 서비스 레이어에서 오류가 발생한 경우
        """
        # 서비스 레이어에 추론을 위임합니다.
        answer = await self._service.infer(query=request.query)

        # 응답 DTO를 구성하여 반환합니다.
        return InferenceResponse(answer=answer)


def get_inference_controller(
    service: Annotated[OpenAIService, Depends(get_openai_service)],
) -> InferenceController:
    """
    InferenceController 인스턴스를 생성하여 반환하는 의존성 팩토리 함수.

    라우터의 Depends()에서 호출됩니다.

    Args:
        service: get_openai_service()로부터 주입받은 OpenAIService 인스턴스

    Returns:
        InferenceController 인스턴스
    """
    return InferenceController(service=service)
