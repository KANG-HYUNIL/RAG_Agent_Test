"""
test_refactor.py

폴더 분리 리팩터링 검증 테스트.
OpenAI API 호출 없이 순수 로직·import·registry 무결성을 확인합니다.
"""
import sys
import os

# src 모듈 접근을 위해 PYTHONPATH 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from omegaconf import OmegaConf


# ============================================================
# 1. Import 테스트 (패키지 경로 정합성)
# ============================================================

def test_imports():
    """공개 API import가 모두 성공하는지 확인"""
    from agent.embedder import Embedder
    from agent.retriever import Retriever
    from agent.prompt_builder import PromptBuilder

    assert Embedder is not None
    assert Retriever is not None
    assert PromptBuilder is not None
    print("[PASS] test_imports")


def test_registry_imports():
    """_registry 모듈 직접 import 및 내부 객체 접근 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY, BasePreprocessStrategy, register_preprocess
    from agent.retriever._registry import RETRIEVAL_STRATEGIES, BaseRetrievalStrategy, register_strategy
    from agent.prompt_builder._registry import _PROMPT_REGISTRY, BasePromptStrategy, register_prompt

    assert isinstance(_PREPROCESS_REGISTRY, dict)
    assert isinstance(RETRIEVAL_STRATEGIES, dict)
    assert isinstance(_PROMPT_REGISTRY, dict)
    print("[PASS] test_registry_imports")


# ============================================================
# 2. Registry 완결성 테스트
# ============================================================

def test_embedder_registry_completeness():
    """embedder에 6개 전처리 전략이 모두 등록되어 있는지 확인"""
    from agent.embedder import Embedder  # __init__ import가 등록을 트리거
    from agent.embedder._registry import _PREPROCESS_REGISTRY

    expected = {
        "raw", "kv_pairs", "narrativized_lite",
        "field_weighted_kv", "dual_representation", "synthetic_query_expansion"
    }
    registered = set(_PREPROCESS_REGISTRY.keys())
    missing = expected - registered
    assert not missing, f"누락된 전처리 전략: {missing}"
    print(f"[PASS] test_embedder_registry_completeness — registered: {sorted(registered)}")


def test_retriever_registry_completeness():
    """retriever에 4개 검색 전략이 모두 등록되어 있는지 확인"""
    from agent.retriever import Retriever  # __init__ import가 등록을 트리거
    from agent.retriever._registry import RETRIEVAL_STRATEGIES

    expected = {"top_k", "score_threshold", "mmr", "hybrid"}
    registered = set(RETRIEVAL_STRATEGIES.keys())
    missing = expected - registered
    assert not missing, f"누락된 검색 전략: {missing}"
    print(f"[PASS] test_retriever_registry_completeness — registered: {sorted(registered)}")


def test_prompt_registry_completeness():
    """prompt_builder에 mcq 전략이 등록되어 있는지 확인"""
    from agent.prompt_builder import PromptBuilder
    from agent.prompt_builder._registry import _PROMPT_REGISTRY

    assert "mcq" in _PROMPT_REGISTRY, "mcq 전략이 _PROMPT_REGISTRY에 없음"
    print(f"[PASS] test_prompt_registry_completeness — registered: {list(_PROMPT_REGISTRY.keys())}")


# ============================================================
# 3. 전처리 전략 로직 테스트 (API 불필요)
# ============================================================

SAMPLE_ROW = {
    "question": "다음 중 민법상 성년의 나이는?",
    "A": "만 18세",
    "B": "만 19세",
    "C": "만 20세",
    "D": "만 21세",
    "answer": "2",
    "Category": "Law",
    "Human Accuracy": "0.85",
}


def test_preprocess_raw():
    """raw 전략: str 반환 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY
    strategy = _PREPROCESS_REGISTRY["raw"]()
    result = strategy.process(SAMPLE_ROW)
    assert isinstance(result, str)
    assert "민법상" in result
    print("[PASS] test_preprocess_raw")


def test_preprocess_kv_pairs():
    """kv_pairs 전략: answer/Human Accuracy 필드 제외, str 반환 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY
    strategy = _PREPROCESS_REGISTRY["kv_pairs"]()
    result = strategy.process(SAMPLE_ROW)
    assert isinstance(result, str)
    assert "answer" not in result
    assert "Human Accuracy" not in result
    assert "민법상" in result
    print("[PASS] test_preprocess_kv_pairs")


def test_preprocess_narrativized_lite():
    """narrativized_lite 전략: 자연어 문장 형태 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY
    strategy = _PREPROCESS_REGISTRY["narrativized_lite"]()
    result = strategy.process(SAMPLE_ROW)
    assert isinstance(result, str)
    assert "카테고리" in result
    assert "선택지" in result
    print("[PASS] test_preprocess_narrativized_lite")


def test_preprocess_dual_representation():
    """dual_representation 전략: List[str] 반환 (2개 표현) 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY
    strategy = _PREPROCESS_REGISTRY["dual_representation"]()
    result = strategy.process(SAMPLE_ROW)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, str) for r in result)
    print("[PASS] test_preprocess_dual_representation")


def test_preprocess_synthetic_no_service():
    """synthetic_query_expansion: openai_service 없으면 kv_pairs 결과 fallback 확인"""
    from agent.embedder._registry import _PREPROCESS_REGISTRY
    strategy = _PREPROCESS_REGISTRY["synthetic_query_expansion"](openai_service=None)
    result = strategy.process(SAMPLE_ROW)
    assert isinstance(result, str)
    # openai_service 없으면 kv_pairs 결과와 동일해야 함
    kv_result = _PREPROCESS_REGISTRY["kv_pairs"]().process(SAMPLE_ROW)
    assert result == kv_result
    print("[PASS] test_preprocess_synthetic_no_service")


def test_embedder_preprocess_dispatch():
    """Embedder.preprocess가 registry를 통해 올바르게 전략을 디스패치하는지 확인"""
    from agent.embedder import Embedder
    embedder = Embedder(openai_service=None)

    result_kv = embedder.preprocess(SAMPLE_ROW, method="kv_pairs")
    assert isinstance(result_kv, str)

    result_dual = embedder.preprocess(SAMPLE_ROW, method="dual_representation")
    assert isinstance(result_dual, list) and len(result_dual) == 2

    try:
        embedder.preprocess(SAMPLE_ROW, method="nonexistent_method")
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass

    print("[PASS] test_embedder_preprocess_dispatch")


# ============================================================
# 4. Retriever 동작 테스트 (mock 임베딩 사용)
# ============================================================

def test_retriever_add_and_search():
    """Retriever: add_documents → search가 올바른 결과를 반환하는지 확인"""
    from agent.retriever import Retriever

    cfg = OmegaConf.create({"method": "top_k", "top_k": 2, "normalize_query": True})
    retriever = Retriever(config=cfg, embedding_dim=4)

    # 3개 mock 문서와 임베딩 추가
    chunks = [
        {"chunk_id": "c0", "content_dict": {"question": "Q0"}},
        {"chunk_id": "c1", "content_dict": {"question": "Q1"}},
        {"chunk_id": "c2", "content_dict": {"question": "Q2"}},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    retriever.add_documents(chunks, embeddings)
    assert retriever.index.ntotal == 3

    # c0 방향으로 쿼리 → top_k=2 결과 반환
    results = retriever.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c0"
    assert "score" in results[0]
    print("[PASS] test_retriever_add_and_search")


def test_retriever_empty_search():
    """Retriever: 문서가 없을 때 빈 리스트 반환 확인"""
    from agent.retriever import Retriever

    cfg = OmegaConf.create({"method": "top_k", "top_k": 3, "normalize_query": True})
    retriever = Retriever(config=cfg, embedding_dim=4)
    results = retriever.search([1.0, 0.0, 0.0, 0.0])
    assert results == []
    print("[PASS] test_retriever_empty_search")


def test_retriever_chunk_embedding_mismatch():
    """add_documents: chunks/embeddings 개수 불일치 시 ValueError 발생 확인"""
    from agent.retriever import Retriever

    cfg = OmegaConf.create({"method": "top_k", "top_k": 3, "normalize_query": True})
    retriever = Retriever(config=cfg, embedding_dim=4)
    try:
        retriever.add_documents([{"chunk_id": "c0"}], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass
    print("[PASS] test_retriever_chunk_embedding_mismatch")


# ============================================================
# 5. PromptBuilder 테스트
# ============================================================

def test_prompt_builder_mcq():
    """PromptBuilder.build_mcq_prompt: 프롬프트 핵심 요소 포함 확인"""
    from agent.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    contexts = [{"content_dict": {"question": "테스트 질문", "A": "선택지A"}}]
    choices = {"A": "만 18세", "B": "만 19세", "C": "만 20세", "D": "만 21세"}

    prompt = builder.build_mcq_prompt("다음 중 성년 나이는?", choices, contexts)

    assert "참고자료" in prompt
    assert "만 18세" in prompt
    assert "만 19세" in prompt
    assert "정답 알파벳" in prompt
    assert builder.system_prompt  # 비어있지 않음
    print("[PASS] test_prompt_builder_mcq")


def test_prompt_builder_empty_context():
    """PromptBuilder: context 없을 때 fallback 문구 포함 확인"""
    from agent.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    prompt = builder.build_mcq_prompt("질문", {"A": "a", "B": "b", "C": "c", "D": "d"}, [])
    assert "참고할 자료가 없습니다" in prompt
    print("[PASS] test_prompt_builder_empty_context")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    tests = [
        test_imports,
        test_registry_imports,
        test_embedder_registry_completeness,
        test_retriever_registry_completeness,
        test_prompt_registry_completeness,
        test_preprocess_raw,
        test_preprocess_kv_pairs,
        test_preprocess_narrativized_lite,
        test_preprocess_dual_representation,
        test_preprocess_synthetic_no_service,
        test_embedder_preprocess_dispatch,
        test_retriever_add_and_search,
        test_retriever_empty_search,
        test_retriever_chunk_embedding_mismatch,
        test_prompt_builder_mcq,
        test_prompt_builder_empty_context,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Result: {passed} passed / {failed} failed / {len(tests)} total")
    print('='*50)
    if failed > 0:
        sys.exit(1)
