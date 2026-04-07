"""
query_encoder 패키지

dev query 텍스트를 구성하는 방법론을 관리합니다.
corpus-side serialization(embedder)과는 독립적으로, query 임베딩 전 텍스트 구성을 담당합니다.

외부 import 예시:
    from agent.query_encoder import build_query_text
"""

from .query_encoder import build_query_text

__all__ = ["build_query_text"]
