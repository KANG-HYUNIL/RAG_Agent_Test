"""
inference_router.py

추론(inference) 관련 HTTP 라우트를 등록하는 라우터 모듈입니다.
FastAPI APIRouter를 사용하여 POST / 엔드포인트를 정의합니다.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.controller.inference_controller import (
    InferenceController,
    get_inference_controller,
)
from app.types import InferenceRequest, InferenceResponse

# 추론 도메인 라우터 인스턴스
# tags를 지정하여 Swagger UI에서 그룹핑됩니다.
router = APIRouter(tags=["inference"])


@router.post(
    "/",
    response_model=InferenceResponse,
    summary="법률 객관식 문제 추론",
    description=(
        "법률 객관식 문제 텍스트(query)를 입력받아 RAG Agent가 추론한 정답 선지(A/B/C/D)를 반환합니다. "
        "내부적으로 text-embedding-3-small 임베딩 + 유사 문서 검색 + gpt-4o-mini 생성 파이프라인을 실행합니다."
    ),
)
async def infer(
    request: InferenceRequest,
    controller: Annotated[InferenceController, Depends(get_inference_controller)],
) -> InferenceResponse:
    """
    POST / — 추론 엔드포인트.

    법률 객관식 문제를 받아 RAG Agent의 추론 결과(A/B/C/D)를 반환합니다.

    Args:
        request: 추론 요청 DTO (query 필드 포함)
        controller: 의존성 주입된 InferenceController 인스턴스

    Returns:
        InferenceResponse: 정답 선지(A/B/C/D)를 담은 응답 DTO
    """
    # 컨트롤러에 실제 추론 처리를 위임합니다.
    return await controller.infer(request=request)
