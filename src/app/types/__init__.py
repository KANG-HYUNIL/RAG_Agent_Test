"""
types 패키지

FastAPI 요청/응답에 사용되는 Pydantic DTO(Data Transfer Object) 모음 패키지입니다.
"""

from .inference_dto import InferenceRequest, InferenceResponse

__all__ = ["InferenceRequest", "InferenceResponse"]
