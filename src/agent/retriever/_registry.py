import faiss
import numpy as np
from typing import Dict, List, Any, Type
from omegaconf import DictConfig

# ==========================================
# Retrieval Strategy Registry
# ==========================================

RETRIEVAL_STRATEGIES: Dict[str, Type["BaseRetrievalStrategy"]] = {}


def register_strategy(name: str):
    """Retrieval Strategy 등록을 위한 데코레이터"""
    def decorator(cls: Type["BaseRetrievalStrategy"]) -> Type["BaseRetrievalStrategy"]:
        RETRIEVAL_STRATEGIES[name] = cls
        return cls
    return decorator


class BaseRetrievalStrategy:
    """Retrieval 전략 생성을 위한 베이스 클래스"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.top_k = config.get("top_k", 5)

    def search(
        self,
        index: faiss.Index,
        documents: List[Dict[str, Any]],
        query_np: np.ndarray,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("서브타입에서 search를 구현해야 합니다.")
