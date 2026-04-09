"""
config.py

애플리케이션 전역 설정을 관리하는 모듈입니다.
pydantic-settings 대신 python-dotenv를 사용하여 .env 파일에서
환경변수를 로드하고, dataclass 형태로 설정값을 제공합니다.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# .env 파일이 존재하면 자동으로 환경변수를 로드합니다.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    애플리케이션 전역 설정 클래스.

    frozen=True를 통해 인스턴스 생성 후 불변(immutable)으로 유지합니다.
    환경변수 미설정 시 기본값 또는 ValueError를 발생시킵니다.
    """

    # OpenAI API 키 (필수 — 미설정 시 ValueError 발생)
    openai_api_key: str = field(default_factory=lambda: _require_env("OPENAI_API_KEY"))

    # OpenAI 생성 모델 이름 (기본값: gpt-4o-mini)
    openai_chat_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )

    # OpenAI 임베딩 모델 이름 (기본값: text-embedding-3-small)
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )

    # 서버 호스트 (기본값: 0.0.0.0)
    server_host: str = field(
        default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0")
    )

    # 서버 포트 (기본값: 8000)
    server_port: int = field(
        default_factory=lambda: int(os.getenv("SERVER_PORT", "8000"))
    )

    # RAG 검색 전략 선택 (configs/retrieval/*.yaml)
    rag_retrieval_strategy: str = field(
        default_factory=lambda: os.getenv("RAG_RETRIEVAL_STRATEGY", "top_k")
    )

    # RAG 프롬프트 전략 선택 (configs/prompt/*.yaml)
    rag_prompt_strategy: str = field(
        default_factory=lambda: os.getenv("RAG_PROMPT_STRATEGY", "raw_stuffing")
    )

    # RAG 질의 표현 전략 선택 (configs/query_representation/*.yaml)
    rag_query_representation_strategy: str = field(
        default_factory=lambda: os.getenv("RAG_QUERY_REPR_STRATEGY", "question_plus_choices")
    )

    # RAG 직렬화 전략 선택 (configs/serialization/*.yaml)
    rag_serialization_strategy: str = field(
        default_factory=lambda: os.getenv("RAG_SERIALIZATION_STRATEGY", "kv_pairs")
    )

    # RAG 추가 설정 (Hydra override 형식, 쉼표로 구분)
    rag_extra_overrides: str = field(
        default_factory=lambda: os.getenv("RAG_EXTRA_OVERRIDES", "")
    )


def _require_env(key: str) -> str:
    """
    필수 환경변수를 반환합니다.

    해당 환경변수가 설정되어 있지 않으면 ValueError를 발생시킵니다.

    Args:
        key: 환경변수 이름

    Returns:
        환경변수 값 (str)

    Raises:
        ValueError: 환경변수가 설정되어 있지 않은 경우
    """
    value = os.getenv(key)
    if value is None:
        msg = f"필수 환경변수 '{key}'가 설정되지 않았습니다. .env 파일을 확인하세요."
        raise ValueError(msg)
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Settings 싱글턴 인스턴스를 반환합니다.

    lru_cache를 활용하여 애플리케이션 전체에서 동일한 인스턴스를
    재사용합니다. FastAPI의 Depends()와 함께 사용할 수 있습니다.

    Returns:
        Settings 인스턴스
    """
    return Settings()
