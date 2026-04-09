"""
inference_controller.py

추론(inference) 관련 HTTP 요청을 처리하는 컨트롤러 모듈입니다.
FastAPI의 Depends()를 통해 OpenAIService를 주입받아 사용합니다.
"""

from typing import Annotated

from fastapi import Depends

from app.service.legal_rag_service import LegalRAGService, get_legal_rag_service
from app.types.inference_dto import InferenceRequest, InferenceResponse


class InferenceController:
    """
    추론 관련 HTTP 요청을 처리하는 컨트롤러 클래스.

    라우터에서 호출되며, 요청 DTO를 서비스 레이어에 전달하고
    응답 DTO를 구성하여 반환하는 역할을 담당합니다.
    """

    def __init__(self, service: LegalRAGService) -> None:
        """
        InferenceController 초기화.

        Args:
            service: LegalRAGService 인스턴스 (Depends를 통해 주입)
        """
        self._service: LegalRAGService = service

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        추론 요청을 처리하고 결과를 InferenceResponse로 반환합니다.

        Args:
            request: 클라이언트로부터 수신한 추론 요청 DTO

        Returns:
            정답 선지(A/B/C/D)를 담은 InferenceResponse DTO
        """
        # 서비스 레이어에 추론을 위임합니다.
        answer = await self._service.ask(query=request.query)

        # 응답 DTO를 구성하여 반환합니다.
        return InferenceResponse(answer=answer)


def get_inference_controller(
    service: Annotated[LegalRAGService, Depends(get_legal_rag_service)],
) -> InferenceController:
    """
    InferenceController 인스턴스를 생성하여 반환하는 의존성 팩토리 함수.
    """
    return InferenceController(service=service)
