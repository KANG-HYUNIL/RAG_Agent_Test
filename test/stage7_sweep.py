"""
stage7_sweep.py

7차 실험 실행 스크립트 — Conservative Deterministic Reranking.
추가 LLM 호출 없이 법령명, 극성, 선택지 키워드를 활용한 보수적 후단 보정 전략 실험.

실험군:
  1. stage7_1_total         : 모든 기능 적용 (Statute + Polarity + Choice Bonus)
  2. stage7_2_no_statute   : 법령명 매칭 제외 (Ablation)
  3. stage7_3_no_polarity  : 극성 불일치 감점 제외 (Ablation)
  4. stage7_4_no_choice    : 선택지 고유 키워드 보너스 제외 (Ablation)
  5. stage7_5_suffix       : total + 자연어 suffix 기반 쿼리 표현 (stage7_suffix_polarity)

실행 방법:
  python test/stage7_sweep.py
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

# Windows 환경에서 한국어 인코딩 문제 해결
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# 실험군 정의
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [
    {
        "name": "stage7__baseline_recheck",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.35",
            "query_representation=question_plus_choices",
        ],
        "backbone": "dense (IP)",
        "post_retrieval": "없음",
        "remark": "6차 최고 기준선 재확인",
    },
    {
        "name": "stage7__dense_rerank_statute",
        "overrides": [
            "retrieval=rerank_stage7_total",
            "retrieval.rerank.penalty_polarity=1.0",
            "retrieval.rerank.bonus_choice=1.0",
            "query_representation=question_plus_choices",
        ],
        "backbone": "dense_qpc",
        "post_retrieval": "statute penalty",
        "remark": "법령 mismatch 보정 효과",
    },
    {
        "name": "stage7__dense_rerank_polarity_soft",
        "overrides": [
            "retrieval=rerank_stage7_total",
            "retrieval.rerank.penalty_conflict=1.0",
            "retrieval.rerank.penalty_missing=1.0",
            "retrieval.rerank.bonus_choice=1.0",
            "query_representation=question_plus_choices",
        ],
        "backbone": "dense_qpc",
        "post_retrieval": "soft polarity penalty",
        "remark": "polarity mismatch 완화 효과",
    },
    {
        "name": "stage7__dense_rerank_statute_polarity",
        "overrides": [
            "retrieval=rerank_stage7_total",
            "retrieval.rerank.bonus_choice=1.0",
            "query_representation=question_plus_choices",
        ],
        "backbone": "dense_qpc",
        "post_retrieval": "statute + polarity penalty",
        "remark": "7차 핵심 후보",
    },
    {
        "name": "stage7__dense_rerank_statute_polarity_choice",
        "overrides": [
            "retrieval=rerank_stage7_total",
            "query_representation=question_plus_choices",
        ],
        "backbone": "dense_qpc",
        "post_retrieval": "statute + polarity + choice bonus",
        "remark": "choice overlap 보조 효과",
    },
    {
        "name": "stage7__dense_suffix_natural_polarity",
        "overrides": [
            "retrieval=score_threshold",
            "retrieval.top_k=5",
            "retrieval.score_threshold.value=0.35",
            "query_representation=stage7_suffix_polarity",
        ],
        "backbone": "dense (IP)",
        "post_retrieval": "없음",
        "remark": "자연문장형 polarity suffix 단독 검증",
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


def run_experiment(exp: dict, sweep_dir: str) -> bool:
    name: str = exp["name"]
    overrides: list[str] = exp["overrides"]

    run_conditions = [
        *_FIXED_CONDITIONS,
        f"experiment_name={name}",
        f"hydra.run.dir={sweep_dir}/{name}",
        *overrides,
    ]

    cmd = [sys.executable, "test/benchmark.py", *run_conditions]
    print(f"\n>>> Running Stage 7: {name}")
    print(f"    {' '.join(cmd)}")

    try:
        # 10분 제한 시간 체크 (필요시 subprocess timeout 설정 가능)
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] 실험 {name} 실패: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 결과 집계
# ─────────────────────────────────────────────────────────────────────────────


def _load_latest_csv(exp_dir: str) -> pd.DataFrame | None:
    pattern = os.path.join(exp_dir, "benchmark_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        return None
    latest = max(csv_files, key=os.path.getmtime)
    df = pd.read_csv(latest)
    result: pd.DataFrame = df[df["#"] != "SUMMARY"].copy()
    return result


def aggregate_results(experiments: list[dict], sweep_dir: str, ts: str) -> None:
    print("\n" + "=" * 70)
    print(">>> Stage 7 실험 결과 집계")
    print("=" * 70)

    summary_rows: list[dict] = []
    category_rows: list[dict] = []

    for exp in experiments:
        name: str = exp["name"]
        remark: str = exp.get("remark", "")
        backbone: str = exp.get("backbone", "dense_qpc")
        post_ret: str = exp.get("post_retrieval", "unknown")
        # 해당 스윕 폴더 내부에서 로드
        exp_dir = os.path.join(sweep_dir, name)

        data_df = _load_latest_csv(exp_dir)
        if data_df is None:
            summary_rows.append({
                "실험명": name,
                "retrieval backbone": backbone,
                "post-retrieval": post_ret,
                "accuracy(%)": "0.00",
                "correct/total": "0/0",
                "avg_time(s/q)": "0.000",
                "status": "FAILED",
                "비고": remark
            })
            continue

        data_df["IsCorrect"] = data_df["Correct"].map({"O": 1, "X": 0})
        total = len(data_df)
        correct = int(float(data_df["IsCorrect"].sum()))
        accuracy = (correct / total * 100) if total > 0 else 0.0

        time_numeric = pd.to_numeric(data_df["Time(s)"], errors="coerce")
        avg_time = float(time_numeric.mean())

        summary_rows.append({
            "실험명": name,
            "retrieval backbone": backbone,
            "post-retrieval": post_ret,
            "accuracy(%)": f"{accuracy:.2f}",
            "correct/total": f"{correct}/{total}",
            "avg_time(s/q)": f"{avg_time:.3f}",
            "status": "COMPLETED",
            "비고": remark,
        })

        # Category별 세분화 집계
        cat_stats = data_df.groupby("Category")["IsCorrect"].agg(["sum", "count"])
        cat_row = {"실험명": name, "비고": remark}
        for cat in ["Law", "Criminal Law"]:
            if cat in cat_stats.index:
                n = int(float(cat_stats.loc[cat, "sum"]))
                tot = int(float(cat_stats.loc[cat, "count"]))
                acc = n / tot * 100 if tot > 0 else 0.0
                cat_row[f"{cat} correct/total"] = f"{n}/{tot}"
                cat_row[f"{cat} accuracy(%)"] = f"{acc:.2f}"
            else:
                cat_row[f"{cat} correct/total"] = "0/0"
                cat_row[f"{cat} accuracy(%)"] = "0.00"
        category_rows.append(cat_row)

    # 출력 결과 및 파일 저장을 위한 버퍼링
    report = []
    report.append("\n#### 7차 실험 기록표\n")
    df_s = pd.DataFrame(summary_rows)
    report.append(df_s.to_markdown(index=False))

    report.append("\n\n#### 7차 Category별 정확도 기록표\n")
    cat_columns = [
        "실험명",
        "Law correct/total", "Law accuracy(%)",
        "Criminal Law correct/total", "Criminal Law accuracy(%)",
        "비고"
    ]
    df_c = pd.DataFrame(category_rows, columns=cat_columns)
    report.append(df_c.to_markdown(index=False))

    # 화면 출력
    full_report = "\n".join(report)
    print(full_report)

    # 파일 저장 (CSV)
    csv_path = os.path.join(sweep_dir, f"summary_all_{ts}.csv")
    df_s.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 파일 저장 (Markdown - 문서 복사용)
    md_path = os.path.join(sweep_dir, f"summary_table_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\n[SAVED] 요약 CSV: {csv_path}")
    print(f"[SAVED] 요약 MD : {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="7차 실험 스윕 실행기")
    parser.add_argument("--yes", action="store_true", help="확인 없이 진행")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join("outputs", "stage7", f"run_{ts}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    print(f"\n7차 실험 스윕 시작 — 총 {len(EXPERIMENTS)}개 실험")
    print(f"출력 디렉토리: {sweep_dir}")
    
    if not args.yes:
        if input("진행하시겠습니까? [y/N] ").lower() != "y":
            return

    for exp in EXPERIMENTS:
        run_experiment(exp, sweep_dir)
    
    aggregate_results(EXPERIMENTS, sweep_dir, ts)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
