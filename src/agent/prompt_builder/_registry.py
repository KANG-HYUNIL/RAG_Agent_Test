from typing import Dict, Type, Any

# ==========================================
# 프롬프트 전략 레지스트리
# ==========================================

_PROMPT_REGISTRY: Dict[str, Type["BasePromptStrategy"]] = {}


def register_prompt(name: str):
    """지정된 이름으로 프롬프트 전략 클래스를 레지스트리에 등록하는 데코레이터"""
    def decorator(cls: Type["BasePromptStrategy"]) -> Type["BasePromptStrategy"]:
        _PROMPT_REGISTRY[name] = cls
        return cls
    return decorator


class BasePromptStrategy:
    """프롬프트 전략 생성을 위한 베이스 클래스"""

    def build(self, **kwargs: Any) -> str:
        raise NotImplementedError("서브타입에서 build를 구현해야 합니다.")
