import io
import json
import logging
import os
import sys
import time
from datetime import datetime

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# 콘솔 출력 시 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# src 모듈 임포트를 위해 src 폴더를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

log = logging.getLogger(__name__)

# 1,2,3,4 정답을 A,B,C,D로 맵핑
_LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _normalize_text(text: str) -> str:
    """텍스트 비교를 위해 공백 및 특수문자를 제거하고 소문자로 변환합니다."""
    import re

    if not text:
        return ""
    # 공백 제거 및 소문자 변환
    text = re.sub(r"\s+", "", text).lower()
    # 일부 특수문자 제거 (필요시 확장)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _parse_prediction_label(text: str) -> str:
    """LLM 답변에서 정답 레이블(A, B, C, D)을 추출합니다."""
    import re

    if not text:
        return "ERR"

    # 텍스트 내에서 처음으로 등장하는 A, B, C, D 중 하나를 찾습니다 (단독 문자이거나 괄호가 붙은 경우 등)
    # 예: "(A)", "A)", "Answer: A"
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)

    # 만약 정규표현식으로 못 찾았다면 첫 글자가 A-D인지 확인
    first_char = text.strip()[:1].upper()
    if first_char in ["A", "B", "C", "D"]:
        return first_char

    return "ERR"


def _print_table(df: pd.DataFrame, title: str = "") -> None:
    """DataFrame을 로그에 보기 좋게 출력합니다."""
    if title:
        log.info(title)
    log.info("\n" + df.to_string(index=False))


def _format_retrieved_nodes(nodes: list, max_count: int, show_answer: bool) -> str:
    """Retrieved nodes를 가독성 좋은 텍스트로 포맷팅합니다."""
    if not nodes:
        return "No contexts retrieved."

    lines = []
    lines.append(f"[Retrieved Contexts] (Top {min(len(nodes), max_count)})")
    for i, node in enumerate(nodes[:max_count]):
        score = node.get("score", 0.0)
        content = node.get("content_dict", {})
        category = content.get("Category", "N/A")

        lines.append("-" * 50)
        lines.append(f"[{i + 1}] Score: {score:.3f} | Category: {category}")

        question = content.get("question", "")
        lines.append(f"Q: {question}")

        choices = []
        for k in ["A", "B", "C", "D"]:
            if k in content:
                choices.append(f"{k}: {content[k]}")
        lines.append(" | ".join(choices))

        if show_answer and "answer" in content:
            lines.append(f"Ans: {content['answer']}")

    lines.append("-" * 50)
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from agent.chunker import Chunker
    from agent.data_loader import DataLoader
    from agent.embedder import Embedder
    from agent.prompt_builder import PromptBuilder
    from agent.query_encoder import build_query_text
    from agent.retriever import Retriever
    from app.service.openai_service import OpenAIService
    from config.config import get_settings

    log.info("=" * 60)
    log.info("RAG Pipeline Benchmark")
    log.info("=" * 60)
    log.info("Active Configuration:\n" + OmegaConf.to_yaml(cfg))
    log.info("=" * 60)

    # ──────────────────────────────────────────────────────────
    # Phase 1: 파이프라인 초기화 및 인덱싱
    # ──────────────────────────────────────────────────────────
    data_loader = DataLoader()
    chunker = Chunker()
    openai_service = OpenAIService(settings=get_settings())
    embedder = Embedder(openai_service=openai_service)
    retriever = Retriever(config=cfg.retrieval, embedding_dim=1536)
    prompt_builder = PromptBuilder()

    # prompt config → method 키와 전략별 kwargs 분리
    prompt_cfg_raw = OmegaConf.to_container(cfg.prompt, resolve=True)
    if not isinstance(prompt_cfg_raw, dict):
        msg = "prompt 설정은 dict여야 합니다."
        raise TypeError(msg)
    prompt_cfg: dict = prompt_cfg_raw
    prompt_method: str = str(prompt_cfg.pop("method"))
    prompt_cfg.pop("description", None)  # 설명 필드는 LLM에 전달하지 않음
    prompt_kwargs: dict = prompt_cfg  # 나머지가 전략별 파라미터

    top_k: int = int(cfg.retrieval.top_k) if "top_k" in cfg.retrieval else 3

    train_data_path = os.path.join(project_root, "data", "train.csv")

    log.info("\n[Phase 1] Indexing (데이터 적재)")

    log.info("[1/4] 데이터 로드 중...")
    train_rows = data_loader.load_csv(train_data_path)
    log.info(f"  -> 총 {len(train_rows)}개의 학습 문서 로드 완료.")

    log.info("[2/4] 데이터 청킹 (Chunking)...")
    chunks = chunker.chunk_data(train_rows)
    log.info(f"  -> {len(chunks)}개의 청크 구조화 완료.")

    log.info(
        f"[3/4] 전처리 및 임베딩 텍스트 추출 중... (직렬화 전략: {cfg.serialization.method})"
    )

    serial_cfg_raw = OmegaConf.to_container(cfg.serialization, resolve=True)
    if not isinstance(serial_cfg_raw, dict):
        msg = "serialization 설정은 dict여야 합니다."
        raise TypeError(msg)
    serial_exclude: list[str] = list(
        serial_cfg_raw.get("exclude_fields", ["answer", "Human Accuracy"])
    )

    expand_chunks: list = []
    texts_to_embed: list[str] = []

    for chunk in chunks:
        preprocessed = embedder.preprocess(
            chunk["content_dict"],
            method=cfg.serialization.method,
            exclude_fields=serial_exclude,
        )
        if isinstance(preprocessed, list):
            for text in preprocessed:
                expand_chunks.append(chunk.copy())
                texts_to_embed.append(text)
        else:
            expand_chunks.append(chunk)
            texts_to_embed.append(preprocessed)

    log.info(f"  -> API 호출을 통하여 총 {len(texts_to_embed)}개의 벡터를 생성합니다.")

    log.info("[4/4] 임베딩 생성 및 FAISS 적재 중...")
    batch_size = 500
    all_embeddings: list = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_text = texts_to_embed[i : i + batch_size]
        batch_embeds = embedder.embed_batch(batch_text)
        all_embeddings.extend(batch_embeds)
        log.info(
            f"  -> ({min(i + batch_size, len(texts_to_embed))} / {len(texts_to_embed)}) 임베딩 완료..."
        )

    retriever.add_documents(expand_chunks, all_embeddings)
    log.info(f"  -> 적재 완료! (총 인덱스 수: {retriever.index.ntotal})")
    log.info("\n[✔] RAG 시스템 초기화 완료.")

    # ──────────────────────────────────────────────────────────
    # Phase 2: dev 데이터 추론 및 평가
    # ──────────────────────────────────────────────────────────
    query_repr_method: str = str(cfg.query_representation.method)
    category_filter_enabled: bool = bool(
        cfg.retrieval.get("category_filter", {}).get("enabled", False)
    )

    log.info("\n[Phase 2] Inference & Evaluation (dev 데이터 평가)")
    log.info(
        f"  프롬프트 전략: {prompt_method}  |  top_k: {top_k}"
        f"  |  query_repr: {query_repr_method}"
        f"  |  category_filter: {category_filter_enabled}"
    )

    dev_data_path = os.path.join(project_root, "data", "dev.csv")
    dev_rows = data_loader.load_csv(dev_data_path)
    total_eval = len(dev_rows)
    log.info(f"  -> 평가 데이터 로드 완료 (총 {total_eval}개)\n")

    logging_cfg = getattr(cfg, "logging", {})
    verbose_query = bool(logging_cfg.get("verbose_query", False))
    log_retrieved_contexts = bool(logging_cfg.get("log_retrieved_contexts", False))
    max_logged_contexts = int(logging_cfg.get("max_logged_contexts", 3))
    full_question_text = bool(logging_cfg.get("full_question_text", False))
    only_log_errors_in_detail = bool(
        logging_cfg.get("only_log_errors_in_detail", False)
    )
    show_answer_in_contexts = bool(logging_cfg.get("show_answer_in_contexts", False))

    save_trace_file = bool(logging_cfg.get("save_trace_file", False))
    trace_include_answer = bool(logging_cfg.get("trace_include_answer_debug", False))

    retrieval_method = str(cfg.retrieval.get("method", "top_k"))

    records: list[dict] = []
    trace_records: list[dict] = []
    correct_count = 0
    row_times: list[float] = []

    benchmark_start = time.perf_counter()  # ← 전체 소요 시간 측정 시작

    for idx, row in enumerate(dev_rows):
        row_start = time.perf_counter()  # ← 개별 소요 시간 측정 시작

        question = str(row.get("question", ""))
        raw_answer_idx = str(row.get("answer", "")).strip()
        answer_label = _LABEL_MAP.get(raw_answer_idx, raw_answer_idx)

        choices = {
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", "")),
        }

        pred = "ERR"
        is_correct = False
        query_text = ""
        prompt_result = None
        retrieved_nodes = []
        error_msg = ""

        try:
            # 1. 쿼리 텍스트 구성 및 임베딩
            query_text = build_query_text(
                method=query_repr_method,
                question=question,
                choices=choices,
            )
            query_vector = embedder.embed(query_text)

            # 2. 관련 컨텍스트 검색 (category_filter 설정 시 metadata 전달)
            query_category = str(row.get("Category", ""))
            metadata_filter = (
                {"Category": query_category}
                if category_filter_enabled and query_category
                else None
            )
            retrieved_nodes = retriever.search(
                query_vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                query_text=query_text,
            )

            # 3. 프롬프트 생성 — PromptResult(system_prompt, user_prompt) 반환
            prompt_result = prompt_builder.build_prompt(
                method=prompt_method,
                question=question,
                choices=choices,
                contexts=retrieved_nodes,
                **prompt_kwargs,
            )

            # 4. LLM 답변 생성 — system_prompt / user_prompt 분리 전달
            generated_answer = openai_service.generate_text(
                prompt=prompt_result.user_prompt,
                system_prompt=prompt_result.system_prompt,
            )

            # 5. 정답 추출 및 확인
            pred = _parse_prediction_label(generated_answer)
            if pred == answer_label:
                correct_count += 1
                is_correct = True

            # 6. Hit@k 계산 제거 (무의미한 지표로 판단되어 비활성화)
            pass

        except Exception as e:
            error_msg = str(e)
            if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
                log.error("\n[!] OpenAI API Key 오류. 평가 루프를 중지합니다.")
                log.error("[!] OPENAI_API_KEY를 확인 후 다시 실행하세요.")
                break
            else:
                log.error(f"[!] Error on row {idx}: {e}")

        row_elapsed = time.perf_counter() - row_start  # ← 개별 소요 시간 측정 끝
        row_times.append(row_elapsed)

        # 버퍼 로깅 로직
        log_buffer = []
        log_buffer.append("=" * 50)
        log_buffer.append(f"[Benchmark Row #{idx + 1}]")

        if verbose_query:
            log_buffer.append(f"[Query Original]\nQ: {question}")
            choices_str = " | ".join([f"{k}: {v}" for k, v in choices.items() if v])
            log_buffer.append(choices_str)
            log_buffer.append(f"\n[Query Embedded Text]\n{query_text}")

        if log_retrieved_contexts:
            log_buffer.append("")
            log_buffer.append(
                _format_retrieved_nodes(
                    retrieved_nodes, max_logged_contexts, show_answer_in_contexts
                )
            )

        # Prompt Preview 삭제/축약 가능 (verbose일때만 일부)
        # Prediction Summary
        time_ms = f"{row_elapsed:.3f}s"
        log_buffer.append(
            f"-> GT: {answer_label} | Pred: {pred} | [{'O' if is_correct else 'X'}] 정답여부 (Time: {time_ms})"
        )
        if error_msg:
            log_buffer.append(f"ERROR: {error_msg}")
        log_buffer.append("=" * 50)

        should_print_detail = True
        if only_log_errors_in_detail and is_correct:
            should_print_detail = False

        if should_print_detail:
            log.info("\n" + "\n".join(log_buffer))
        else:
            if idx % 10 == 0:
                log.info(
                    f"[{idx + 1:>4}/{total_eval}] GT: {answer_label} | Pred: {pred} | "
                    f"{'O' if is_correct else 'X'} | {time_ms} | Q: {question[:30]}..."
                )

        question_to_save = question if full_question_text else question[:50]
        records.append(
            {
                "#": idx + 1,
                "Question": question_to_save,
                "GT": answer_label,
                "Pred": pred,
                "Correct": "O" if is_correct else "X",
                "Category": str(row.get("Category", "N/A")),
                "Time(s)": round(row_elapsed, 3),
            }
        )

        if save_trace_file:
            contexts_list = []
            for n in retrieved_nodes:
                content_dict = n.get("content_dict", {})
                context_dict = {
                    "rank": len(contexts_list) + 1,
                    "score": n.get("score"),
                    "Category": content_dict.get("Category", ""),
                    "question": content_dict.get("question", ""),
                    "A": content_dict.get("A", ""),
                    "B": content_dict.get("B", ""),
                    "C": content_dict.get("C", ""),
                    "D": content_dict.get("D", ""),
                }
                if trace_include_answer:
                    context_dict["answer"] = content_dict.get("answer", "")
                contexts_list.append(context_dict)

            trace_records.append(
                {
                    "row_index": idx + 1,
                    "Category": str(row.get("Category", "")),
                    "question_original": question,
                    "query_embedded_text": query_text,
                    "gt_answer": answer_label,
                    "pred_answer": pred,
                    "correct": is_correct,
                    "latency_sec": row_elapsed,
                    "prompt_strategy": prompt_method,
                    "retrieval_method": retrieval_method,
                    "query_representation": query_repr_method,
                    "retrieved_contexts": contexts_list,
                }
            )

    total_elapsed = time.perf_counter() - benchmark_start  # ← 전체 소요 시간 측정 끝

    # ──────────────────────────────────────────────────────────
    # 결과 집계
    # ──────────────────────────────────────────────────────────
    evaluated = len(row_times)
    accuracy = (correct_count / evaluated * 100) if evaluated > 0 else 0.0
    avg_time = sum(row_times) / evaluated if evaluated > 0 else 0.0

    results_df = pd.DataFrame(records)

    # ── Category별 정확도 계산 ────────────────────────────
    category_acc_df = (
        results_df.groupby("Category")["Correct"]
        .apply(lambda x: (x == "O").mean() * 100)
        .reset_index()
    )
    category_acc_df.columns = ["Category", "Accuracy(%)"]
    _print_table(category_acc_df, title="\n[Category-wise Accuracy]")

    log.info("\n" + "=" * 60)
    log.info("SUMMARY STATISTICS")
    log.info("-" * 60)
    log.info(f"Total Accuracy: {accuracy:.2f}%")
    log.info(f"Total Time: {total_elapsed:.2f}s | Avg/Row: {avg_time:.3f}s")
    log.info("=" * 60)

    # ── CSV 저장 ─────────────────────────────────────────────
    # HydraConfig에서 실제 run.dir 경로를 가져옴 (Hydra 1.3+ job.chdir=False 대응)
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_{prompt_method}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # per-row 결과 + 요약 행 함께 저장
    summary_row = pd.DataFrame(
        [
            {
                "#": "SUMMARY",
                "Question": f"strategy={prompt_method}",
                "GT": "-",
                "Pred": "-",
                "Correct": f"{correct_count}/{evaluated}",
                "Time(s)": f"total={total_elapsed:.2f}s  avg={avg_time:.3f}s  acc={accuracy:.2f}%",
            }
        ]
    )
    pd.concat([results_df, summary_row], ignore_index=True).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    log.info(f"\n[✔] CSV 저장 완료: {csv_path}")

    if getattr(cfg, "logging", {}).get("save_trace_file", False) and trace_records:
        trace_format_cfg = str(
            getattr(cfg, "logging", {}).get("trace_format", "jsonl")
        ).lower()
        if trace_format_cfg == "csv":
            trace_filename = f"benchmark_trace_{prompt_method}_{timestamp}.csv"
            trace_path = os.path.join(output_dir, trace_filename)
            trace_df = pd.DataFrame(trace_records)
            # CSV 형태일 때 리스트 형태를 JSON string으로 변환
            if "retrieved_contexts" in trace_df.columns:
                trace_df["retrieved_contexts"] = trace_df["retrieved_contexts"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                )
            trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")
            log.info(f"[✔] CSV (Trace 로그) 저장 완료: {trace_path}")
        else:  # default jsonl
            trace_filename = f"benchmark_trace_{prompt_method}_{timestamp}.jsonl"
            trace_path = os.path.join(output_dir, trace_filename)
            with open(trace_path, "w", encoding="utf-8") as f:
                for rec in trace_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.info(f"[✔] JSONL (Trace 로그) 저장 완료: {trace_path}")

    # ── 오답 리스트 별도 저장 (JSONL & Excel) ──────────────────
    failed_trace = [r for r in trace_records if not r["correct"]]
    if failed_trace:
        # JSONL 저장
        fail_jsonl_path = os.path.join(
            output_dir, f"errors_{prompt_method}_{timestamp}.jsonl"
        )
        with open(fail_jsonl_path, "w", encoding="utf-8") as f:
            for rec in failed_trace:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(f"[✔] 오답 JSONL 저장 완료: {fail_jsonl_path}")

        # Excel 저장 (pandas 활용)
        fail_xlsx_path = os.path.join(
            output_dir, f"errors_{prompt_method}_{timestamp}.xlsx"
        )
        try:
            # 엑셀 저장을 위해 분석하기 좋은 플랫한 구조로 변환
            failed_df = pd.DataFrame(failed_trace)
            # contexts는 너무 기므로 요약이나 제외 검토 (여기서는 문자열화)
            if "retrieved_contexts" in failed_df.columns:
                failed_df["retrieved_contexts"] = failed_df["retrieved_contexts"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                )

            failed_df.to_excel(fail_xlsx_path, index=False)
            log.info(f"[✔] 오답 Excel 저장 완료: {fail_xlsx_path}")
        except Exception as e:
            log.warning(f"[!] Excel 저장 실패 (openpyxl 설치 여부 확인 필요): {e}")
            # 대체제로 CSV 저장
            fail_csv_path = os.path.join(
                output_dir, f"errors_{prompt_method}_{timestamp}.csv"
            )
            pd.DataFrame(failed_trace).to_csv(
                fail_csv_path, index=False, encoding="utf-8-sig"
            )
            log.info(f"[✔] 대신 CSV로 오답 저장 완료: {fail_csv_path}")

    log.info(f"[✔] 최종 정확도: {accuracy:.2f}%  ({correct_count}/{evaluated})")
    log.info(f"[✔] 전체 소요 시간: {total_elapsed:.2f}s  |  평균 {avg_time:.3f}s/문항")


if __name__ == "__main__":
    main()
