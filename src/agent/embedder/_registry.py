from typing import Any

# ==========================================
# 전처리(Preprocessing) 전략 레지스트리
# ==========================================

_PREPROCESS_REGISTRY: dict[str, type["BasePreprocessStrategy"]] = {}


def register_preprocess(name: str):
    """지정된 이름으로 전처리 전략 클래스를 레지스트리에 등록하는 데코레이터"""

    def decorator(
        cls: type["BasePreprocessStrategy"],
    ) -> type["BasePreprocessStrategy"]:
        _PREPROCESS_REGISTRY[name] = cls
        return cls

    return decorator


class BasePreprocessStrategy:
    """전처리 전략 생성을 위한 베이스 클래스"""

    def __init__(self, openai_service: Any | None = None):
        # synthetic_query_expansion 등 LLM 호출이 필요한 전략에서 사용
        self.openai_service = openai_service

    def process(self, row: dict, exclude_fields: list[str] | None = None) -> str | list[str]:
        raise NotImplementedError("서브타입에서 process를 구현해야 합니다.")
