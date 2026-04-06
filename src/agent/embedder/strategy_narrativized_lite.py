from ._registry import register_preprocess, BasePreprocessStrategy


@register_preprocess("narrativized_lite")
class NarrativizedLitePreprocessStrategy(BasePreprocessStrategy):
    """
    2. Narrativized-lite
    문장 형태로 풀어씀.
    """

    def process(self, row: dict) -> str:
        category = row.get("Category", "일반")
        question = row.get("question", "")
        options = []
        for opt in ["A", "B", "C", "D"]:
            if opt in row:
                options.append(f"{opt}. {row[opt]}")

        opt_str = " ".join(options)
        return (
            f"이 문제는 {category} 카테고리에 관한 내용입니다. "
            f"질문은 '{question}' 이며, 선택지로는 {opt_str} 가 있습니다."
        )
