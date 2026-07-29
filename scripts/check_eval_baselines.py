"""Compare the latest eval run of each task against recorded baselines.

    python scripts/check_eval_baselines.py          # check, exit 1 on regression
    python scripts/check_eval_baselines.py --write  # record current as baseline

Absolute thresholds don't work here: agents are stochastic, so a criterion
scoring 0.8 one week and 0.75 the next is noise, while one that scored 1.0 and
now scores 0.0 is a regression. Baselines make that distinction — the question
is whether a score *dropped*, not whether it cleared some bar.

Criteria scored None need a human and are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "eval" / "baselines.json"
RUNS_DIR = REPO_ROOT / "outputs" / "eval_runs"

# How far a score may fall before it counts as a regression rather than noise.
TOLERANCE = 0.1


def latest_scores() -> dict[str, dict[str, float]]:
    """Most recent run per task -> {criterion: score}, skipping None scores."""
    newest: dict[str, tuple[float, dict]] = {}
    for path in RUNS_DIR.glob("*.json"):
        try:
            run = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        task_id, started = run.get("task_id"), run.get("started_at", 0)
        if task_id and started >= newest.get(task_id, (0, {}))[0]:
            newest[task_id] = (started, run)

    return {
        task_id: {k: v for k, v in (run.get("scores") or {}).items() if v is not None}
        for task_id, (_started, run) in newest.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="record instead of check")
    args = parser.parse_args()

    current = latest_scores()
    if not current:
        print("no eval runs found in outputs/eval_runs/ — nothing to compare")
        return 0

    if args.write:
        BASELINES.parent.mkdir(parents=True, exist_ok=True)
        BASELINES.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        n = sum(len(v) for v in current.values())
        print(f"recorded {n} criterion score(s) across {len(current)} task(s)")
        return 0

    if not BASELINES.exists():
        print(f"no baselines at {BASELINES} — run with --write to record them")
        return 0

    baseline = json.loads(BASELINES.read_text())
    regressions, improvements, new = [], [], []

    for task_id, scores in sorted(current.items()):
        for criterion, score in sorted(scores.items()):
            was = baseline.get(task_id, {}).get(criterion)
            if was is None:
                new.append((task_id, criterion, score))
            elif score < was - TOLERANCE:
                regressions.append((task_id, criterion, was, score))
            elif score > was + TOLERANCE:
                improvements.append((task_id, criterion, was, score))

    for task_id, criterion, was, now in improvements:
        print(f"↑ {task_id}.{criterion}: {was} → {now}")
    for task_id, criterion, score in new:
        print(f"+ {task_id}.{criterion}: {score} (no baseline)")
    for task_id, criterion, was, now in regressions:
        print(f"✗ {task_id}.{criterion}: {was} → {now}")

    if regressions:
        print(f"\n{len(regressions)} regression(s) beyond the {TOLERANCE} tolerance")
        return 1

    print(f"\nno regressions across {sum(len(v) for v in current.values())} score(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
