import io
import logging
import os
import sys
import time
from datetime import datetime

# 콘솔 출력 시 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# src 모듈 임포트를 위해 src 폴더를 PYTHONPATH에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import hydra  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from agent.chunker import Chunker  # noqa: E402
from agent.data_loader import DataLoader  # noqa: E402
from agent.embedder import Embedder  # noqa: E402
from agent.prompt_builder import PromptBuilder  # noqa: E402
from agent.retriever import Retriever  # noqa: E402
from app.service.openai_service import OpenAIService  # noqa: E402
from config.config import get_settings  # noqa: E402

log = logging.getLogger(__name__)

# 1,2,3,4 정답을 A,B,C,D로 맵핑
_LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _print_table(df: pd.DataFrame, title: str = "") -> None:
    """DataFrame을 로그에 보기 좋게 출력합니다."""
    if title:
        log.info(title)
    log.info("\n" + df.to_string(index=False))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
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
    prompt_cfg: dict = OmegaConf.to_container(cfg.prompt, resolve=True)  # type: ignore[assignment]
    prompt_method: str = str(prompt_cfg.pop("method"))
    prompt_cfg.pop("description", None)  # 설명 필드는 LLM에 전달하지 않음
    prompt_kwargs: dict = prompt_cfg  # 나머지가 전략별 파라미터

    top_k: int = int(cfg.retrieval.k) if "k" in cfg.retrieval else 3

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
    expand_chunks: list = []
    texts_to_embed: list[str] = []

    for chunk in chunks:
        preprocessed = embedder.preprocess(
            chunk["content_dict"], method=cfg.serialization.method
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
    log.info("\n[Phase 2] Inference & Evaluation (dev 데이터 평가)")
    log.info(f"  프롬프트 전략: {prompt_method}  |  top_k: {top_k}")

    dev_data_path = os.path.join(project_root, "data", "dev.csv")
    dev_rows = data_loader.load_csv(dev_data_path)
    total_eval = len(dev_rows)
    log.info(f"  -> 평가 데이터 로드 완료 (총 {total_eval}개)\n")

    records: list[dict] = []
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

        try:
            # 1. 쿼리 임베딩
            query_vector = embedder.embed(question)

            # 2. 관련 컨텍스트 검색
            retrieved_nodes = retriever.search(query_vector, top_k=top_k)

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

            # 5. 정답 확인
            pred = generated_answer.strip().upper()
            if pred == answer_label:
                correct_count += 1
                is_correct = True

        except Exception as e:
            if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
                log.error("\n[!] OpenAI API Key 오류. 평가 루프를 중지합니다.")
                log.error("[!] OPENAI_API_KEY를 확인 후 다시 실행하세요.")
                break
            else:
                log.error(f"[!] Error on row {idx}: {e}")

        row_elapsed = time.perf_counter() - row_start  # ← 개별 소요 시간 측정 끝
        row_times.append(row_elapsed)

        records.append(
            {
                "#": idx + 1,
                "Question (50c)": question[:50],
                "GT": answer_label,
                "Pred": pred,
                "Correct": "O" if is_correct else "X",
                "Time(s)": round(row_elapsed, 3),
            }
        )

        if idx % 10 == 0:
            log.info(
                f"[{idx + 1:>4}/{total_eval}] GT: {answer_label} | Pred: {pred} | "
                f"{'O' if is_correct else 'X'} | {row_elapsed:.3f}s | Q: {question[:30]}..."
            )

    total_elapsed = time.perf_counter() - benchmark_start  # ← 전체 소요 시간 측정 끝

    # ──────────────────────────────────────────────────────────
    # 결과 집계
    # ──────────────────────────────────────────────────────────
    evaluated = len(row_times)
    accuracy = (correct_count / evaluated * 100) if evaluated > 0 else 0.0
    avg_time = sum(row_times) / evaluated if evaluated > 0 else 0.0

    results_df = pd.DataFrame(records)

    # ── 콘솔 테이블 출력 ──────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 60)
    _print_table(results_df, title="[Per-Row Results]")

    summary_df = pd.DataFrame(
        [
            {
                "Prompt Strategy": prompt_method,
                "Evaluated": evaluated,
                "Correct": correct_count,
                "Accuracy(%)": f"{accuracy:.2f}",
                "Total Time(s)": f"{total_elapsed:.2f}",
                "Avg Time/Row(s)": f"{avg_time:.3f}",
            }
        ]
    )
    _print_table(summary_df, title="\n[Summary]")
    log.info("=" * 60)

    # ── CSV 저장 ─────────────────────────────────────────────
    # Hydra output_dir은 현재 작업 디렉터리(CWD)로 자동 변경되어 있음
    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_{prompt_method}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # per-row 결과 + 요약 행 함께 저장
    summary_row = pd.DataFrame(
        [
            {
                "#": "SUMMARY",
                "Question (50c)": f"strategy={prompt_method}",
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
    log.info(f"[✔] 최종 정확도: {accuracy:.2f}%  ({correct_count}/{evaluated})")
    log.info(f"[✔] 전체 소요 시간: {total_elapsed:.2f}s  |  평균 {avg_time:.3f}s/문항")


if __name__ == "__main__":
    main()
