from ._registry import BasePreprocessStrategy, register_preprocess


@register_preprocess("field_weighted_kv")
class FieldWeightedKVPreprocessStrategy(BasePreprocessStrategy):
    """
    3. Field-Weighted KV
    내용 순서가 성능에 영향을 줄 수 있으므로 중요한 필드를 먼저, 혹은 텍스트상에서 명시적으로 강조.
    """

    def process(self, row: dict) -> str:
        question_part = f"핵심 질문(Question): {row.get('question', '')}. "
        category_part = f"영역(Category): [ {row.get('Category', '')} ]. "

        opt_str = " / ".join(
            [f"선택지{o} ({row.get(o, '')})" for o in ["A", "B", "C", "D"] if o in row]
        )
        options_part = f"보기(Options): {opt_str}."

        return question_part + category_part + options_part
