from ._registry import _PREPROCESS_REGISTRY, BasePreprocessStrategy, register_preprocess


@register_preprocess("dual_representation")
class DualRepresentationPreprocessStrategy(BasePreprocessStrategy):
    """
    4. Dual Representation
    같은 row를 두 가지 표현(KV Pairs + Narrativized-lite)으로 각각 색인.
    """

    def process(self, row: dict, exclude_fields: list[str] | None = None) -> list[str]:
        
        
        rep1 = _PREPROCESS_REGISTRY["kv_pairs"](
            openai_service=self.openai_service
        ).process(row, exclude_fields=exclude_fields)
        
        
        rep2 = _PREPROCESS_REGISTRY["narrativized_lite"](
            openai_service=self.openai_service
        ).process(row, exclude_fields=exclude_fields)

        
        r1 = rep1 if isinstance(rep1, str) else " ".join(rep1)
        r2 = rep2 if isinstance(rep2, str) else " ".join(rep2)
        return [r1, r2]
