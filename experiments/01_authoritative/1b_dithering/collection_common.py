#!/usr/bin/env python3
"""
Shared helpers for the window-aware collection (macro-review decision D25).

Why this exists: an AI agent can give different decisions for IDENTICAL input in different time windows, and be perfectly
repeatable inside one window. If a condition is run in its own window, window effects look like data-quality effects. The
collection is therefore a randomized block design:

    plan_collection.py   splits every call (baseline runs, every dithered condition, the clean re-run control) into many
                         BLOCKS so that each block holds a random slice of everything. Blocks are run in sequence, so every
                         condition is spread across many windows and a window effect becomes noise, not a bias.
    run_collection.py    runs the blocks through the (unchanged) agent, resumable, shardable.
    collect_results.py   routes the rows back into the per-condition files the evaluators already read.

Every task gets its own task id (written into the block file's record_id) because the SAME record_id appears in all
conditions; the agent never shows record_id to the model, so this changes nothing in the prompt.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

PLAN_VERSION = 1
VALID_DECISIONS = ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY")


def input_hash(record: Dict[str, Any]) -> str:
    """md5 of the record exactly as the agent sees it (record_id excluded). Must equal the agent's own _input_hash."""
    display_record = {k: v for k, v in record.items() if k != "record_id"}
    return hashlib.md5(json.dumps(display_record, sort_keys=True, default=str).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path, tolerant: bool = False) -> List[Dict[str, Any]]:
    """Read a JSONL file. tolerant=True skips unparseable lines (an interrupted run can leave a truncated last line)."""
    rows = []
    if not Path(path).exists():
        return rows
    for line in open(path):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not tolerant:
                raise
    return rows


def block_name(block: int) -> str:
    return f"block_{block:04d}"


def block_paths(plan_dir: Path, block: int) -> Dict[str, Path]:
    d = Path(plan_dir) / "blocks"
    n = block_name(block)
    return {"input": d / f"{n}.input.jsonl", "decisions": d / f"{n}.decisions.jsonl",
            "window": d / f"{n}.window.json", "log": d / f"{n}.log",
            "batches": d / f"{n}.batches.json", "meta": d / f"{n}.decisions.meta.jsonl"}


def load_plan(plan_dir: Path) -> Tuple[Dict[str, Any], Dict[int, List[Dict[str, Any]]]]:
    """(plan, tasks grouped by block). Refuses a directory without a finished plan.json."""
    plan_path = Path(plan_dir) / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"❌ no plan.json in {plan_dir}: run plan_collection.py first (a half-written plan has no plan.json).")
    plan = json.load(open(plan_path))
    if plan.get("plan_version") != PLAN_VERSION:
        raise SystemExit(f"❌ plan version {plan.get('plan_version')} is not the version this tool reads ({PLAN_VERSION}).")
    tasks_path = Path(plan_dir) / "tasks.jsonl"
    if sha256_file(tasks_path) != plan["tasks_sha256"]:
        raise SystemExit("❌ tasks.jsonl does not match the hash recorded in plan.json: the plan was altered after it was written.")
    by_block: Dict[int, List[Dict[str, Any]]] = {}
    for t in read_jsonl(tasks_path):
        by_block.setdefault(t["block"], []).append(t)
    return plan, by_block


def block_status(plan_dir: Path, block: int, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which of a block's tasks have a valid decision on the input the plan expects?"""
    rows = read_jsonl(block_paths(plan_dir, block)["decisions"], tolerant=True)
    last: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        last[r.get("record_id")] = r                       # a later row for the same task replaces an earlier one
    valid, bad_hash, invalid, missing = {}, [], [], []
    for t in tasks:
        r = last.get(t["task_id"])
        if r is None:
            missing.append(t["task_id"])
        elif r.get("business_decision") not in VALID_DECISIONS:
            invalid.append(t["task_id"])
        elif r.get("input_hash") != t["input_hash"]:
            bad_hash.append(t["task_id"])
        else:
            valid[t["task_id"]] = r
    return {"n": len(tasks), "valid": valid, "bad_hash": bad_hash, "invalid": invalid, "missing": missing,
            "complete": len(valid) == len(tasks)}


# ---------------------------------------------------------------------------------------------------------------------
# Pre-mortem guards (G1 budget, G4 failed calls, G1 era check); decided by the owner 2026-11-11
# ---------------------------------------------------------------------------------------------------------------------

DEFAULT_BUDGET_USD = 150.0     # G1: pause-and-ask between blocks; never cuts a batch off
FAILURE_CEILING = 0.02         # G4: share of attempted tasks still failed after retries (mechanical failures only)
ERA_GAP_HOURS = 6.0            # G1: a gap longer than this between consecutive blocks starts a new era


def spend_so_far(plan_dir: Path, n_blocks: int) -> Dict[str, float]:
    """Recorded cost of every row in every block's decisions file (retried and failed rows included: they were billed)."""
    total, n, priced = 0.0, 0, 0
    for b in range(1, n_blocks + 1):
        for r in read_jsonl(block_paths(plan_dir, b)["decisions"], tolerant=True):
            n += 1
            c = r.get("cost_usd")
            if isinstance(c, (int, float)):
                total += c
                priced += 1
    return {"usd": total, "rows": n, "priced_rows": priced, "mean_per_row": (total / priced) if priced else None}


def failure_summary(plan_dir: Path, by_block: Dict[int, List[Dict[str, Any]]], blocks: List[int]) -> Dict[str, Any]:
    """G4: failed calls among ATTEMPTED blocks (a block counts once it has a decisions file). A failed call is mechanical:
    missing (errored or expired and not recovered), invalid (no HIGH/MEDIUM/LOW decision, i.e. unparseable), or made on the
    wrong input. Odd but well-formed reasoning is data, never a failure."""
    attempted = [b for b in blocks if block_paths(plan_dir, b)["decisions"].exists()]
    n = failed = 0
    for b in attempted:
        st = block_status(plan_dir, b, by_block[b])
        n += st["n"]
        failed += len(st["missing"]) + len(st["invalid"]) + len(st["bad_hash"])
    rate = failed / n if n else 0.0
    return {"attempted_blocks": len(attempted), "tasks": n, "failed": failed, "rate": rate, "ceiling": FAILURE_CEILING,
            "over_ceiling": rate > FAILURE_CEILING}


def _ts(x):
    from datetime import datetime
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def compute_eras(plan_dir: Path, n_blocks: int, gap_hours: float = ERA_GAP_HOURS) -> Dict[str, Any]:
    """G1 era check: order finished blocks by start time; a gap of more than gap_hours between one block's end and the next
    block's start begins a new era. Returns the eras, every gap over the threshold, and each block's era."""
    import json as _json
    windows = []
    for b in range(1, n_blocks + 1):
        w = block_paths(plan_dir, b)["window"]
        if w.exists():
            d = _json.load(open(w))
            windows.append((b, _ts(d["started_at"]), _ts(d["finished_at"])))
    windows.sort(key=lambda t: t[1])
    eras, gaps, block_era = [], [], {}
    for i, (b, start, end) in enumerate(windows):
        if i == 0:
            eras.append({"era": 1, "blocks": [], "start": start.isoformat(), "end": end.isoformat()})
        else:
            prev_end = windows[i - 1][2]
            gap = (start - prev_end).total_seconds() / 3600
            if gap > gap_hours:
                gaps.append({"after_block": windows[i - 1][0], "before_block": b, "hours": round(gap, 2)})
                eras.append({"era": len(eras) + 1, "blocks": [], "start": start.isoformat(), "end": end.isoformat()})
        eras[-1]["blocks"].append(b)
        eras[-1]["end"] = max(_ts(eras[-1]["end"]), end).isoformat()
        block_era[b] = eras[-1]["era"]
    return {"gap_hours_threshold": gap_hours, "n_eras": len(eras), "eras": eras, "gaps_over_threshold": gaps, "block_era": block_era}
