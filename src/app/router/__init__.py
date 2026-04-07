"""
router 패키지

FastAPI APIRouter를 통해 HTTP 라우트를 등록하는 패키지입니다.
각 도메인별 라우터를 이 패키지에서 관리합니다.
"""

from .health_router import router as health_router
from .inference_router import router as inference_router

__all__ = ["health_router", "inference_router"]
