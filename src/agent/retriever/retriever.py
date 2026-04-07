import logging
from typing import Any, cast

import faiss
import numpy as np
from omegaconf import DictConfig

from ._registry import RETRIEVAL_STRATEGIES, _FaissIndex

log = logging.getLogger(__name__)


class Retriever:
    """
    유사도 검색(Retrieval)을 수행하는 역할을 합니다.
    Registry를 통해 동적으로 설정된 Strategy(Top-K, Score Threshold, MMR, Hybrid 등)를 사용합니다.
    """

    def __init__(self, config: DictConfig, embedding_dim: int = 1536):
        self.config = config
        self.embedding_dim = embedding_dim

        # 내적(Inner Product) 인덱스 생성. (top_k 기본 동작이 Cosine Sim이므로 FlatIP)
        # cast: faiss의 SWIG 스텁은 런타임 패치 전 시그니처를 노출하므로 _FaissIndex Protocol로 캐스팅
        self.index: _FaissIndex = cast(_FaissIndex, faiss.IndexFlatIP(embedding_dim))

        # 원본 청크(문서 내용)를 식별하기 위해 인메모리 리스트로 유지합니다.
        self.documents: list[dict[str, Any]] = []

        # Strategy 인스턴스화
        method_name = self.config.get("method", "top_k")
        if method_name not in RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"알 수 없는 Retrieval Strategy: {method_name}. "
                f"사용 가능: {list(RETRIEVAL_STRATEGIES.keys())}"
            )

        self.strategy = RETRIEVAL_STRATEGIES[method_name](self.config)
        log.info(f"Retriever Strategy '{method_name}' 초기화 완료.")

    def add_documents(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None:
        """FAISS 벡터 DB 및 구조화 문서 추가"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks의 개수와 embeddings의 개수가 일치하지 않습니다.")
        if not embeddings:
            return

        embeddings_np = np.array(embeddings, dtype=np.float32)
        # 길이를 1로 정규화
        faiss.normalize_L2(embeddings_np)

        self.index.add(embeddings_np)
        self.documents.extend(chunks)

    def search(
        self, query_embedding: list[float], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Registry 패턴으로 생성된 Strategy를 통해 검색"""
        if self.index.ntotal == 0:
            return []

        # config상의 top_k 오버라이드 지원 (호출부 하위호환성 유지 용도)
        if top_k is not None:
            self.strategy.top_k = top_k

        query_np = np.array([query_embedding], dtype=np.float32)

        # query normalize 설정
        if self.config.get("normalize_query", True):
            faiss.normalize_L2(query_np)

        return self.strategy.search(self.index, self.documents, query_np)
