from ._registry import BasePreprocessStrategy, register_preprocess


@register_preprocess("kv_pairs")
class KVPairsPreprocessStrategy(BasePreprocessStrategy):
    """
    1. KV Pairs
    가장 단순하고 구조 보존력이 높음.
    question, A, B, C, D, Category 등의 키-값 쌍 직렬화.
    """

    def process(self, row: dict, exclude_fields: list[str] | None = None) -> str:
        excluded = set(
            exclude_fields
            if exclude_fields is not None
            else ["answer", "Human Accuracy"]
        )
        items = [f"{k}: {v}" for k, v in row.items() if k not in excluded]
        return " / ".join(items)
