from typing import List

from ._registry import register_preprocess, BasePreprocessStrategy, _PREPROCESS_REGISTRY


@register_preprocess("dual_representation")
class DualRepresentationPreprocessStrategy(BasePreprocessStrategy):
    """
    4. Dual Representation
    같은 row를 두 가지 표현(KV Pairs + Narrativized-lite)으로 각각 색인.
    """

    def process(self, row: dict) -> List[str]:
        rep1 = _PREPROCESS_REGISTRY["kv_pairs"](openai_service=self.openai_service).process(row)
        rep2 = _PREPROCESS_REGISTRY["narrativized_lite"](openai_service=self.openai_service).process(row)
        return [rep1, rep2]
