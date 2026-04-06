"""
inference_dto.py

추론(inference) 엔드포인트의 요청/응답 DTO(Data Transfer Object) 정의 모듈입니다.
Pydantic BaseModel을 사용하여 입력 유효성 검사와 OpenAPI 문서 자동 생성을 지원합니다.
"""

from typing import Literal

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """
    POST / 추론 요청 DTO.

    법률 객관식 문제 텍스트를 담아 서버에 전달합니다.
    """

    query: str = Field(
        description="추론 대상 법률 객관식 문제 텍스트. 문제 본문과 선지(A/B/C/D)를 포함해야 합니다.",
        min_length=1,
        examples=["다음 중 형사소송법상 증거능력이 없는 것은? A. ... B. ... C. ... D. ..."],
    )


class InferenceResponse(BaseModel):
    """
    POST / 추론 응답 DTO.

    RAG Agent가 선택한 정답 선지를 반환합니다.
    반환값은 반드시 A, B, C, D 중 하나입니다.
    """

    answer: Literal["A", "B", "C", "D"] = Field(
        description="RAG Agent가 추론한 정답 선지. A, B, C, D 중 하나의 문자열로만 반환됩니다.",
        examples=["A"],
    )
