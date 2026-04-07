from ._registry import BasePreprocessStrategy, register_preprocess


@register_preprocess("narrativized_lite")
class NarrativizedLitePreprocessStrategy(BasePreprocessStrategy):
    """
    2. Narrativized-lite
    문장 형태로 풀어씀.
    """

    def process(self, row: dict, exclude_fields: list[str] | None = None) -> str:
        excluded = set(exclude_fields or [])
        parts: list[str] = []
        if "Category" not in excluded:
            category = row.get("Category", "일반")
            parts.append(f"이 문제는 {category} 카테고리에 관한 내용입니다.")
        if "question" not in excluded:
            question = row.get("question", "")
            parts.append(f"질문은 '{question}' 이며,")
        options = [
            f"{opt}. {row[opt]}"
            for opt in ["A", "B", "C", "D"]
            if opt in row and opt not in excluded
        ]
        if options:
            parts.append(f"선택지로는 {' '.join(options)} 가 있습니다.")
        return " ".join(parts)
