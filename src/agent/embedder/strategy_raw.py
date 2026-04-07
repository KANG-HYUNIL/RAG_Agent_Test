from ._registry import BasePreprocessStrategy, register_preprocess


@register_preprocess("raw")
class RawPreprocessStrategy(BasePreprocessStrategy):
    """
    0. Raw (Baseline)
    어떠한 추가 로직도 없이, 딕셔너리 내용을 단순히 문자열로 반환합니다.
    (예: JSON 형식 또는 Python dict 표기법 그대로)
    """

    def process(self, row: dict) -> str:
        return str(row)
