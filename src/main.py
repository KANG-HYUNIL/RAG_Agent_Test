"""
main.py

RAG Agent FastAPI 서버의 외부 엔트리포인트 모듈입니다.

이 파일을 직접 실행하면 uvicorn ASGI 서버가 기동됩니다.

실행 방법:
    uv run python -m src.main
    또는
    uv run uvicorn src.app.server:create_app --factory --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.server import create_app
from config.config import get_settings


def main() -> None:
    """
    uvicorn 서버를 기동하는 메인 함수.

    애플리케이션 설정(Settings)에서 호스트와 포트를 읽어
    uvicorn ASGI 서버를 시작합니다.
    """
    # 애플리케이션 설정을 로드합니다.
    settings = get_settings()

    # FastAPI 앱 인스턴스를 생성합니다.
    app = create_app()

    # uvicorn 서버를 실행합니다.
    uvicorn.run(
        app,
        host=settings.server_host,  # 기본값: 0.0.0.0
        port=settings.server_port,  # 기본값: 8000
        log_level="info",
    )


if __name__ == "__main__":
    main()
