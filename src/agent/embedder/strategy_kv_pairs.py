from ._registry import register_preprocess, BasePreprocessStrategy


@register_preprocess("kv_pairs")
class KVPairsPreprocessStrategy(BasePreprocessStrategy):
    """
    1. KV Pairs
    가장 단순하고 구조 보존력이 높음.
    question, A, B, C, D, Category 등의 키-값 쌍 직렬화.
    """

    def process(self, row: dict) -> str:
        items = []
        for k, v in row.items():
            if k not in ["answer", "Human Accuracy"]:
                items.append(f"{k}: {v}")
        return " / ".join(items)
