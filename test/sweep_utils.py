"""
sweep_utils.py — OAAT / Stage2 공통 유틸리티

oaat_sweep.py 와 stage2_sweep.py 가 공유하는
실험 실행·파싱·출력·저장 로직을 모아 둔 모듈.
"""

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# ─── 결과 CSV 필드 순서 ──────────────────────────────────────
_FIELDS = [
    "name",
    "axis",
    "serialization",
    "retrieval",
    "prompt",
    "query_representation",
    "accuracy_pct",
    "correct",
    "total",
    "total_time_s",
    "avg_time_s",
    "status",
]


def run_experiment(
    run_cfg: dict,
    hydra_root: str,
    extra_overrides: list[str] | None = None,
) -> dict:
    """
    단일 실험을 subprocess로 실행하고 결과를 파싱하여 반환.

    benchmark.py는 Hydra를 통해 실행되며,
    hydra.run.dir 오버라이드로 각 실험의 출력 경로를 분리합니다.

    extra_overrides: 추가 Hydra dot-notation 오버라이드 목록.
                     예: ["retrieval.top_k=3", "retrieval.score_threshold.value=0.30"]
    """
    name = run_cfg["name"]
    folder_name = (
        f"serial-{run_cfg['serialization']}"
        f"__ret-{run_cfg['retrieval']}"
        f"__prompt-{run_cfg['prompt']}"
        f"__qr-{run_cfg['query_representation']}"
    )
    run_dir = os.path.join(hydra_root, folder_name)

    cmd = [
        sys.executable,
        "test/benchmark.py",
        f"serialization={run_cfg['serialization']}",
        f"retrieval={run_cfg['retrieval']}",
        f"prompt={run_cfg['prompt']}",
        f"query_representation={run_cfg['query_representation']}",
        f"experiment_name={name}",
        f"hydra.run.dir={run_dir}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    print(f"\n{'─' * 60}")
    print(f"[RUN] {name}")
    print(
        f"      serial={run_cfg['serialization']}"
        f"  retrieval={run_cfg['retrieval']}"
        f"  prompt={run_cfg['prompt']}"
        f"  qr={run_cfg['query_representation']}"
    )
    print(f"{'─' * 60}")

    result: dict = {
        "name": name,
        "axis": run_cfg["axis"],
        "serialization": run_cfg["serialization"],
        "retrieval": run_cfg["retrieval"],
        "prompt": run_cfg["prompt"],
        "query_representation": run_cfg["query_representation"],
        "accuracy_pct": None,
        "correct": None,
        "total": None,
        "total_time_s": None,
        "avg_time_s": None,
        "status": "pending",
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        combined = proc.stdout + proc.stderr

        # "[✔] 최종 정확도: 72.50%  (29/40)"
        acc_m = re.search(r"최종 정확도:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", combined)
        if acc_m:
            result["accuracy_pct"] = float(acc_m.group(1))
            result["correct"] = int(acc_m.group(2))
            result["total"] = int(acc_m.group(3))

        # "[✔] 전체 소요 시간: 42.31s  |  평균 1.058s/문항"
        time_m = re.search(r"전체 소요 시간:\s*([\d.]+)s", combined)
        if time_m:
            result["total_time_s"] = float(time_m.group(1))

        avg_m = re.search(r"평균\s*([\d.]+)s/문항", combined)
        if avg_m:
            result["avg_time_s"] = float(avg_m.group(1))

        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            result["status"] = f"error(rc={proc.returncode})"

        # ── run.log 저장 (subprocess stdout+stderr 전체) ────
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(combined)

        # ── 진행 상황 출력 ──────────────────────────────────
        if result["accuracy_pct"] is not None:
            print(
                f"[OK]  accuracy={result['accuracy_pct']:.2f}%"
                f"  ({result['correct']}/{result['total']})"
                f"  total={result['total_time_s']:.1f}s"
                f"  avg={result['avg_time_s']:.3f}s/q"
            )
        else:
            print(f"[WARN] 결과 파싱 실패. status={result['status']}")
            if proc.returncode != 0:
                tail = combined[-500:].strip()
                print(f"--- STDERR tail ---\n{tail}\n---")

    except Exception as exc:
        result["status"] = f"exception:{exc}"
        print(f"[ERROR] {exc}")

    return result


def print_summary_table(results: list[dict]) -> None:
    """완료 후 콘솔 요약 테이블 출력."""
    header = f"{'축':<25} {'실험명':<50} {'정확도':>9} {'correct':>8} {'time(s)':>9}"
    print(f"\n{'=' * 95}")
    print("SWEEP 결과 요약")
    print("=" * 95)
    print(header)
    print("─" * 95)
    for r in results:
        acc = f"{r['accuracy_pct']:.2f}%" if r["accuracy_pct"] is not None else "N/A"
        correct = f"{r['correct']}/{r['total']}" if r["correct"] is not None else "N/A"
        t = f"{r['total_time_s']:.1f}" if r["total_time_s"] is not None else "N/A"
        print(f"{r['axis']:<25} {r['name']:<50} {acc:>9} {correct:>8} {t:>9}")
    print("=" * 95)


def save_summary(results: list[dict], output_dir: str, prefix: str = "sweep") -> None:
    """결과를 CSV + JSON으로 저장."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"{prefix}_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, f"{prefix}_summary_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] CSV  → {csv_path}")
    print(f"[SAVED] JSON → {json_path}")
