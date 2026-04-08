"""
stage3_sweep.py — 3차 하이퍼파라미터 튜닝 실험 자동화

목적:
  - 2차 실험 기준, 전체 최고 조합은 kv_pairs + top_k + raw_stuffing + question_plus_choices
  - score_threshold retrieval의 top_k(초기 후보 수) × threshold 조합을 full factorial로 탐색
  - 고정 축: serialization=kv_pairs, prompt=raw_stuffing, query_representation=question_plus_choices
  - 탐색 파라미터: k ∈ {3, 5, 7, 10}, threshold ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
  - 총 36개 (4 × 9) 실험
  - 결과를 outputs/stage3_sweep/{timestamp}/ 에 요약 CSV + JSON 저장

실행 방법 (프로젝트 루트에서):
  python test/stage3_sweep.py         # 전체 3차 실험 (36개)
  python test/stage3_sweep.py --yes   # 확인 프롬프트 스킵
"""

import argparse
import os
import sys
from datetime import datetime

from sweep_utils import print_summary_table, run_experiment, save_summary

# ─── 3차 실험 고정 축 ────────────────────────────────────────
_FIXED = {
    "serialization": "kv_pairs",
    "retrieval": "score_threshold",
    "prompt": "raw_stuffing",
    "query_representation": "question_plus_choices",
}

# ─── 탐색 파라미터 ───────────────────────────────────────────
_K_VALUES = [3, 5, 7, 10]
_THRESHOLD_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_run_list() -> list[dict]:
    """k × threshold full factorial 실험 목록 생성."""
    runs: list[dict] = []
    for k in _K_VALUES:
        for t in _THRESHOLD_VALUES:
            t_str = f"{int(t * 100):03d}"  # 0.30 → "030"
            name = f"stage3__k{k}__t{t_str}"
            runs.append(
                {
                    "name": name,
                    "axis": f"k{k}",
                    **_FIXED,
                    # extra_overrides 전용 메타 필드 (run_experiment에서 pop)
                    "_extra": [
                        f"retrieval.top_k={k}",
                        f"retrieval.score_threshold.value={t:.2f}",
                    ],
                    # 요약 테이블·CSV용 정보
                    "_k": k,
                    "_threshold": t,
                }
            )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage3 Sweep: retrieval 하이퍼파라미터 full factorial 탐색"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage3_sweep",
        help="결과 저장 루트 디렉터리 (기본: outputs/stage3_sweep)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="실행 전 확인 프롬프트를 스킵합니다",
    )
    args = parser.parse_args()

    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, sweep_ts)

    runs = build_run_list()

    # ── 실행 계획 출력 ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"Stage3 Sweep  |  총 {len(runs)}개 실험  (k={_K_VALUES} × threshold={_THRESHOLD_VALUES})")
    print(f"고정 축       |  {_FIXED}")
    print(f"출력 경로     |  {sweep_dir}")
    print("=" * 80)
    print(f"{'#':<4} {'실험명':<35} {'k':>4} {'threshold':>10}")
    print(f"{'─' * 4} {'─' * 35} {'─' * 4} {'─' * 10}")
    for i, r in enumerate(runs, 1):
        print(f"{i:<4} {r['name']:<35} {r['_k']:>4} {r['_threshold']:>10.2f}")

    if not args.yes:
        print("\n실험을 시작합니다. Enter를 누르면 계속, Ctrl+C로 취소합니다...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n취소됨.")
            sys.exit(0)

    # ── 실험 실행 ────────────────────────────────────────────
    results: list[dict] = []
    for run_cfg in runs:
        # _extra / _k / _threshold 는 run_experiment에 전달하지 않음
        extra = run_cfg.pop("_extra")
        run_cfg.pop("_k")
        run_cfg.pop("_threshold")

        result = run_experiment(run_cfg, sweep_dir, extra_overrides=extra)
        results.append(result)

    # ── 결과 출력 + 저장 ─────────────────────────────────────
    print_summary_table(results)
    save_summary(results, sweep_dir, prefix="stage3")


if __name__ == "__main__":
    main()
