from ._registry import BasePreprocessStrategy, register_preprocess


@register_preprocess("field_weighted_kv")
class FieldWeightedKVPreprocessStrategy(BasePreprocessStrategy):
    """
    3. Field-Weighted KV
    내용 순서가 성능에 영향을 줄 수 있으므로 중요한 필드를 먼저, 혹은 텍스트상에서 명시적으로 강조.
    """

    def process(self, row: dict, exclude_fields: list[str] | None = None) -> str:
        excluded = set(exclude_fields or [])
        parts: list[str] = []
        if "question" not in excluded:
            parts.append(f"핵심 질문(Question): {row.get('question', '')}.")
        if "Category" not in excluded:
            parts.append(f"영역(Category): [ {row.get('Category', '')} ].")
        choices = [
            f"선택지{o} ({row.get(o, '')})"
            for o in ["A", "B", "C", "D"]
            if o in row and o not in excluded
        ]
        if choices:
            parts.append(f"보기(Options): {' / '.join(choices)}.")
        return " ".join(parts)
