from ._registry import _PREPROCESS_REGISTRY, BasePreprocessStrategy, register_preprocess


@register_preprocess("dual_representation")
class DualRepresentationPreprocessStrategy(BasePreprocessStrategy):
    """
    4. Dual Representation
    같은 row를 두 가지 표현(KV Pairs + Narrativized-lite)으로 각각 색인.
    """

    def process(self, row: dict) -> list[str]:
        rep1 = _PREPROCESS_REGISTRY["kv_pairs"](
            openai_service=self.openai_service
        ).process(row)
        rep2 = _PREPROCESS_REGISTRY["narrativized_lite"](
            openai_service=self.openai_service
        ).process(row)
        # 두 전략 모두 str을 반환하지만 BasePreprocessStrategy.process 반환 타입이
        # Union[str, List[str]]이므로 str 타입으로 보장
        r1 = rep1 if isinstance(rep1, str) else " ".join(rep1)
        r2 = rep2 if isinstance(rep2, str) else " ".join(rep2)
        return [r1, r2]
