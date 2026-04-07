"""
stage2_sweep.py — 2차 조합 실험 자동화

목적:
  - 1차 OAAT 실험 결과를 바탕으로 우수 후보 축들의 조합 효과 검증
  - serialization = kv_pairs 고정
  - retrieval × prompt × query_representation 조합 탐색
  - 결과를 outputs/stage2_sweep/{timestamp}/ 에 요약 CSV + JSON 저장

실행 방법 (프로젝트 루트에서):
  python test/stage2_sweep.py         # 전체 2차 실험 (5개)
  python test/stage2_sweep.py --yes   # 확인 프롬프트 스킵
"""

import argparse
import os
import sys
from datetime import datetime

from sweep_utils import print_summary_table, run_experiment, save_summary

# ─── 2차 실험 조합 정의 ─────────────────────────────────────
# serialization = kv_pairs 고정
# 우선순위: 최우선 → 우선 → 탐색 순으로 정렬
STAGE2_RUNS: list[dict] = [
    {
        "name": "stage2__score_threshold__raw__qpc",
        "axis": "stage2",
        "serialization": "kv_pairs",
        "retrieval": "score_threshold",
        "prompt": "raw_stuffing",
        "query_representation": "question_plus_choices",
    },
    {
        "name": "stage2__score_threshold__structured__qpc",
        "axis": "stage2",
        "serialization": "kv_pairs",
        "retrieval": "score_threshold",
        "prompt": "structured_context",
        "query_representation": "question_plus_choices",
    },
    {
        "name": "stage2__top_k__structured__qpc",
        "axis": "stage2",
        "serialization": "kv_pairs",
        "retrieval": "top_k",
        "prompt": "structured_context",
        "query_representation": "question_plus_choices",
    },
    {
        "name": "stage2__mmr__raw__qpc",
        "axis": "stage2",
        "serialization": "kv_pairs",
        "retrieval": "mmr",
        "prompt": "raw_stuffing",
        "query_representation": "question_plus_choices",
    },
    {
        "name": "stage2__category_filter__raw__qpc",
        "axis": "stage2",
        "serialization": "kv_pairs",
        "retrieval": "top_k_category_filter",
        "prompt": "raw_stuffing",
        "query_representation": "question_plus_choices",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage2 Sweep: 1차 결과 기반 조합 실험 자동 실행"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage2_sweep",
        help="결과 저장 루트 디렉터리 (기본: outputs/stage2_sweep)",
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

    # ── 실행 계획 출력 ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"Stage2 Sweep  |  총 {len(STAGE2_RUNS)}개 실험")
    print(f"출력 경로     |  {sweep_dir}")
    print("=" * 80)
    print(f"{'#':<4} {'실험명':<55} {'retrieval':<22} {'prompt':<22} {'qr'}")
    print(f"{'─' * 4} {'─' * 55} {'─' * 22} {'─' * 22} {'─' * 22}")
    for i, r in enumerate(STAGE2_RUNS, 1):
        print(
            f"{i:<4} {r['name']:<55}"
            f" {r['retrieval']:<22}"
            f" {r['prompt']:<22}"
            f" {r['query_representation']}"
        )

    if not args.yes:
        print("\n실험을 시작합니다. Enter를 누르면 계속, Ctrl+C로 취소합니다...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n취소됨.")
            sys.exit(0)

    # ── 실험 실행 ────────────────────────────────────────────
    results: list[dict] = []
    for run_cfg in STAGE2_RUNS:
        result = run_experiment(run_cfg, sweep_dir)
        results.append(result)

    # ── 결과 출력 + 저장 ─────────────────────────────────────
    print_summary_table(results)
    save_summary(results, sweep_dir, prefix="stage2")


if __name__ == "__main__":
    main()
