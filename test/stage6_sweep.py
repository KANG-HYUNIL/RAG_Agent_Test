"""
stage6_sweep.py

6차 실험 실행 스크립트 — Dense 후단 보정(Post-Retrieval Reranking) 실험.

기준선: stage5__baseline__dense_qpc (score_threshold k=5 t=0.35 + question_plus_choices)
6차 목표: dense 후보를 더 잘 재선별하여 precision 개선

실험군:
  1. stage6__baseline_recheck          : rerank(보정OFF, retrieve_k=5) — 재현성 검증
  2. stage6__dense_rerank_polarity      : polarity mismatch 후단 감점
  3. stage6__dense_rerank_bm25aux       : BM25 보조 신호 가산 (후보 내 lexical 재정렬)
  4. stage6__dense_rerank_polarity_bm25aux : polarity + BM25 aux 동시 적용
  5. stage6__dense_rerank_polarity_bm25aux_f3 : 위 + final_k=3 (precision 집중)
  6. stage6__dense_rerank_template_penalty : OX/연결형 템플릿 과적합 감점

실행 방법:
  python test/stage6_sweep.py          # 전체 실험
  python test/stage6_sweep.py --yes    # 확인 없이 전체 실행
  python test/stage6_sweep.py --mode baseline_only --yes
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 실험군 정의
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [
    {
        "name": "stage6__baseline_recheck",
        "overrides": [
            "retrieval=rerank_base",
            "query_representation=question_plus_choices",
        ],
        "remark": "stage5 최강 기준선 재확인 (재현성 검증, 보정 OFF)",
    },
    {
        "name": "stage6__dense_rerank_polarity",
        "overrides": [
            "retrieval=rerank_polarity",
            "query_representation=question_plus_choices",
        ],
        "remark": "polarity mismatch 후단 감점 — 부정/긍정 방향 오염 후보 제거",
    },
    {
        "name": "stage6__dense_rerank_bm25aux",
        "overrides": [
            "retrieval=rerank_bm25aux",
            "query_representation=question_plus_choices",
        ],
        "remark": "BM25 보조 신호 가산 — dense 후보 내 lexical overlap 재정렬",
    },
    {
        "name": "stage6__dense_rerank_polarity_bm25aux",
        "overrides": [
            "retrieval=rerank_polarity_bm25aux",
            "query_representation=question_plus_choices",
        ],
        "remark": "polarity penalty + BM25 aux 동시 적용 — 조합 효과 확인",
    },
    {
        "name": "stage6__dense_rerank_polarity_bm25aux_f3",
        "overrides": [
            "retrieval=rerank_polarity_bm25aux_f3",
            "query_representation=question_plus_choices",
        ],
        "remark": "polarity + BM25 + final_k=3 — precision 집중 (top-3만 LLM 전달)",
    },
    {
        "name": "stage6__dense_rerank_template_penalty",
        "overrides": [
            "retrieval=rerank_template",
            "query_representation=question_plus_choices",
        ],
        "remark": "OX/연결형 템플릿 과적합 감점 — 형식 유사 distractor 억제",
    },
]

# 모든 실험 공통 고정 조건
_FIXED_CONDITIONS = [
    "serialization=kv_pairs",
    "prompt=raw_stuffing",
    "logging.save_trace_file=true",
]


# ─────────────────────────────────────────────────────────────────────────────
# 단일 실험 실행
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(exp: dict) -> bool:
    name: str = exp["name"]
    overrides: list[str] = exp["overrides"]

    run_conditions = [
        *_FIXED_CONDITIONS,
        f"experiment_name={name}",
        f"hydra.run.dir=outputs/stage6/{name}",
        *overrides,
    ]

    cmd = [sys.executable, "test/benchmark.py", *run_conditions]
    print(f"\n>>> Running: {name}")
    print(f"    {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] 실험 {name} 실패: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 결과 집계
# ─────────────────────────────────────────────────────────────────────────────


def _load_latest_csv(exp_dir: str) -> pd.DataFrame | None:
    """지정 디렉토리에서 가장 최신 benchmark_*.csv를 로드합니다."""
    pattern = os.path.join(exp_dir, "benchmark_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        return None
    latest = max(csv_files, key=os.path.getmtime)
    df = pd.read_csv(latest)
    result: pd.DataFrame = df[df["#"] != "SUMMARY"].copy()  # type: ignore[assignment]
    return result


def aggregate_results(experiments: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(">>> 6차 실험 결과 집계")
    print("=" * 70)

    summary_rows: list[dict] = []
    category_rows: list[dict] = []

    for exp in experiments:
        name: str = exp["name"]
        remark: str = exp.get("remark", "")
        exp_dir = f"outputs/stage6/{name}"

        data_df = _load_latest_csv(exp_dir)
        if data_df is None:
            summary_rows.append(
                {
                    "실험명": name,
                    "retrieval": "-",
                    "query_repr": "-",
                    "accuracy(%)": "-",
                    "correct/total": "N/A",
                    "avg_time(s/q)": "-",
                    "status": "FAILED",
                    "비고": remark,
                }
            )
            continue

        # ── 정확도 계산 ─────────────────────────────────────────────────────
        data_df["IsCorrect"] = data_df["Correct"].map({"O": 1, "X": 0})  # type: ignore[arg-type]
        total = len(data_df)
        correct = int(float(data_df["IsCorrect"].sum()))  # type: ignore[arg-type]
        accuracy = (correct / total * 100) if total > 0 else 0.0

        # ── 시간 계산 ────────────────────────────────────────────────────────
        time_numeric: pd.Series = pd.to_numeric(  # type: ignore[assignment]
            data_df["Time(s)"], errors="coerce"
        )
        avg_time: float = float(time_numeric.mean())

        # retrieval / query_repr 파싱 (overrides에서)
        overrides = exp["overrides"]
        ret_type = next(
            (o.split("=")[1] for o in overrides if o.startswith("retrieval=")),
            "unknown",
        )
        qr_type = next(
            (
                o.split("=")[1]
                for o in overrides
                if o.startswith("query_representation=")
            ),
            "unknown",
        )

        summary_rows.append(
            {
                "실험명": name,
                "retrieval": ret_type,
                "query_repr": qr_type,
                "accuracy(%)": f"{accuracy:.2f}",
                "correct/total": f"{correct}/{total}",
                "avg_time(s/q)": f"{avg_time:.3f}" if avg_time == avg_time else "-",
                "status": "COMPLETED",
                "비고": remark,
            }
        )

        # ── Category별 정확도 ────────────────────────────────────────────────
        cat_stats = data_df.groupby("Category")["IsCorrect"].agg(["sum", "count"])

        law = cat_stats.loc["Law"] if "Law" in cat_stats.index else None
        crim = (
            cat_stats.loc["Criminal Law"] if "Criminal Law" in cat_stats.index else None
        )

        cat_row: dict = {"실험명": name, "비고": remark}
        if law is not None:
            law_n = int(float(law["sum"]))  # type: ignore[arg-type]
            law_total = int(float(law["count"]))  # type: ignore[arg-type]
            law_acc = law_n / law_total * 100 if law_total > 0 else 0.0
            cat_row["Law correct/total"] = f"{law_n}/{law_total}"
            cat_row["Law accuracy(%)"] = f"{law_acc:.2f}"
        else:
            cat_row["Law correct/total"] = "-"
            cat_row["Law accuracy(%)"] = "-"

        if crim is not None:
            crim_n = int(float(crim["sum"]))  # type: ignore[arg-type]
            crim_total = int(float(crim["count"]))  # type: ignore[arg-type]
            crim_acc = crim_n / crim_total * 100 if crim_total > 0 else 0.0
            cat_row["Criminal Law correct/total"] = f"{crim_n}/{crim_total}"
            cat_row["Criminal Law accuracy(%)"] = f"{crim_acc:.2f}"
        else:
            cat_row["Criminal Law correct/total"] = "-"
            cat_row["Criminal Law accuracy(%)"] = "-"

        category_rows.append(cat_row)

    # ── 테이블 1: 전체 정확도 ─────────────────────────────────────────────────
    print("\n#### 6차 실험 기록표\n")
    cols1 = [
        "실험명",
        "retrieval",
        "query_repr",
        "accuracy(%)",
        "correct/total",
        "avg_time(s/q)",
        "status",
        "비고",
    ]
    header = " | ".join(cols1)
    sep = " | ".join(["---"] * len(cols1))
    print(f"| {header} |")
    print(f"| {sep} |")
    for r in summary_rows:
        vals = " | ".join(str(r.get(c, "-")) for c in cols1)
        print(f"| {vals} |")

    # ── 테이블 2: Category별 ──────────────────────────────────────────────────
    print("\n#### 6차 Category별 정확도\n")
    cols2 = [
        "실험명",
        "Law correct/total",
        "Law accuracy(%)",
        "Criminal Law correct/total",
        "Criminal Law accuracy(%)",
        "비고",
    ]
    header2 = " | ".join(cols2)
    sep2 = " | ".join(["---"] * len(cols2))
    print(f"| {header2} |")
    print(f"| {sep2} |")
    for r in category_rows:
        vals2 = " | ".join(str(r.get(c, "-")) for c in cols2)
        print(f"| {vals2} |")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "outputs/stage6"
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, f"stage6_summary_{ts}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[✔] 요약 CSV 저장: {summary_path}")

    cat_path = os.path.join(out_dir, f"stage6_category_{ts}.csv")
    pd.DataFrame(category_rows).to_csv(cat_path, index=False, encoding="utf-8-sig")
    print(f"[✔] Category별 CSV 저장: {cat_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6차 실험 스윕 실행기 (Post-Retrieval Reranking)")
    parser.add_argument(
        "--mode",
        choices=["all", "baseline_only"],
        default="all",
        help="실행 모드 (기본: all)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="실행 전 확인 없이 진행",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    exps = EXPERIMENTS if args.mode == "all" else [EXPERIMENTS[0]]

    print("\n" + "=" * 70)
    print(f"6차 실험 스윕 — 총 {len(exps)}개 실험 (Dense 후단 보정)")
    print("=" * 70)
    for e in exps:
        print(f"  {e['name']}")
        print(f"    └ {e['remark']}")
    print()

    if not args.yes:
        confirm = input("실험을 시작하시겠습니까? [y/N] ").strip().lower()
        if confirm != "y":
            print("취소.")
            return

    success = 0
    for exp in exps:
        if run_experiment(exp):
            success += 1

    print(f"\n>>> {success}/{len(exps)} 실험 완료.")
    aggregate_results(exps)


if __name__ == "__main__":
    main()
