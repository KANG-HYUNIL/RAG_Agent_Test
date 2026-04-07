"""
oaat_sweep.py — One-At-A-Time (OAAT) Baseline + 단일 축 변경 실험 자동화

목적:
  - baseline 1회 실행
  - 각 실험 축(serialization / retrieval / prompt / query_representation)을
    한 번에 하나씩 변경하여 실험
  - 결과를 outputs/oaat_sweep/{timestamp}/ 에 요약 CSV + JSON 및 개별 실험 로그로 저장

실행 방법 (프로젝트 루트에서):
  python test/oaat_sweep.py                       # oaat (기본)
  python test/oaat_sweep.py --mode baseline_only  # baseline만
  python test/oaat_sweep.py --yes                 # 확인 프롬프트 스킵
"""

import argparse
import os
import sys
from datetime import datetime

from sweep_utils import _FIELDS, print_summary_table, run_experiment, save_summary

# ─── Baseline 정의 (config.yaml defaults와 일치) ──────────────
BASELINE: dict[str, str] = {
    "serialization": "kv_pairs",
    "retrieval": "top_k",
    "prompt": "raw_stuffing",
    "query_representation": "question_only",
}

# ─── 축별 후보 (yaml 파일명 = Hydra group value, baseline 제외) ──
AXES: dict[str, list[str]] = {
    "serialization": [
        "raw",
        "narrative",
        "weighted",
        "dual",
        "kv_pairs_no_category",
    ],
    "retrieval": [
        "score_threshold",
        "mmr",
        "top_k_category_filter",
    ],
    "prompt": [
        "labeled_context",
        "structured_context",
        "few_shot_envelope",
    ],
    "query_representation": [
        "question_plus_choices",
    ],
}

__all__ = ["_FIELDS"]


def build_run_list(mode: str) -> list[dict]:
    """실행 목록 생성. baseline은 항상 첫 번째."""
    runs: list[dict] = []

    runs.append({"name": "baseline", "axis": "baseline", **BASELINE})

    if mode == "baseline_only":
        return runs

    # oaat: 각 축을 하나씩 바꾸고 나머지는 baseline 고정
    for axis, variants in AXES.items():
        for variant in variants:
            cfg = {**BASELINE, axis: variant}
            runs.append(
                {
                    "name": f"{axis}__{variant}",
                    "axis": axis,
                    **cfg,
                }
            )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OAAT Sweep: baseline + 축별 단독 변경 실험 자동 실행"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline_only", "oaat"],
        default="oaat",
        help="실험 모드 (기본: oaat)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oaat_sweep",
        help="결과 저장 루트 디렉터리 (기본: outputs/oaat_sweep)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="실행 전 확인 프롬프트를 스킵합니다",
    )
    args = parser.parse_args()

    # sweep 실행마다 타임스탬프 폴더로 격리
    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, sweep_ts)

    runs = build_run_list(args.mode)

    # ── 실행 계획 출력 ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"OAAT Sweep  |  mode={args.mode}  |  총 {len(runs)}개 실험")
    print(f"출력 경로   |  {sweep_dir}")
    print("=" * 80)
    print(f"{'#':<4} {'실험명':<40} {'축':<22}")
    print(f"{'─' * 4} {'─' * 40} {'─' * 22}")
    for i, r in enumerate(runs, 1):
        print(f"{i:<4} {r['name']:<40} {r['axis']:<22}")

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
        result = run_experiment(run_cfg, sweep_dir)
        results.append(result)

    # ── 결과 출력 + 저장 ─────────────────────────────────────
    print_summary_table(results)
    save_summary(results, sweep_dir, prefix="oaat")


if __name__ == "__main__":
    main()
