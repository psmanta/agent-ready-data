#!/usr/bin/env python3
"""
Run the planned blocks through the agent (macro-review decision D25).
====================================================================
Runs business_decision_agent.py on each block file, in block order, with --resume, so an interrupted run loses
nothing and a re-run only does the missing work. A block is "complete" only when every task in it has a valid decision
(a real priority, on exactly the input the plan expects); anything else is retried, then reported.

    python3 run_collection.py --plan_dir experiments_output/collection --dry_run        # count, cost, nothing sent
    python3 run_collection.py --plan_dir experiments_output/collection --yes             # run everything
    python3 run_collection.py --plan_dir experiments_output/collection --yes --shard 1/4 # worker 1 of 4

Several workers (--shard i/N) can run at the same time in different terminals: block b belongs to worker
((b - 1) mod N) + 1, so every worker still touches every part of the design, and windows stay mixed.
Each block records when it started and finished (block_NNNN.window.json) for window analysis; the agent also
writes a run fingerprint (block_NNNN.decisions.meta.jsonl).
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collection_common import block_paths, block_status, load_plan, sha256_file  # noqa: E402


def parse_blocks(spec: str, n_blocks: int) -> List[int]:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = sorted(b for b in out if not 1 <= b <= n_blocks)
    if bad:
        raise SystemExit(f"❌ block(s) {bad} are outside 1..{n_blocks}")
    return sorted(out)


def parse_shard(spec: str):
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise SystemExit(f"❌ --shard must look like 2/4, got {spec!r}")
    if not (1 <= i <= n):
        raise SystemExit(f"❌ --shard {spec}: need 1 <= i <= N")
    return i, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan_dir", type=Path, default=Path("experiments_output/collection"))
    ap.add_argument("--blocks", type=str, default=None, help="e.g. 1-6,9 (default: all)")
    ap.add_argument("--shard", type=str, default=None, help="i/N: run only the blocks belonging to worker i of N")
    ap.add_argument("--agent", type=Path, default=Path(__file__).resolve().parent / "business_decision_agent.py")
    ap.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=2, help="extra attempts per block if tasks are still missing or invalid")
    ap.add_argument("--cost_per_call", type=float, default=0.00256, help="for the estimate only")
    ap.add_argument("--dry_run", action="store_true", help="show what would run; send nothing")
    ap.add_argument("--yes", action="store_true", help="confirm that real API calls should be made")
    a = ap.parse_args()

    plan, by_block = load_plan(a.plan_dir)
    n_blocks = plan["blocks"]
    chosen = parse_blocks(a.blocks, n_blocks) if a.blocks else list(range(1, n_blocks + 1))
    if a.shard:
        i, n = parse_shard(a.shard)
        chosen = [b for b in chosen if (b - 1) % n == i - 1]

    todo, done_tasks, todo_tasks = [], 0, 0
    for b in chosen:
        st = block_status(a.plan_dir, b, by_block[b])
        if st["complete"]:
            done_tasks += st["n"]
        else:
            todo.append(b)
            todo_tasks += st["n"] - len(st["valid"])
            done_tasks += len(st["valid"])
    print(f"plan: {plan['n_tasks']:,} tasks in {n_blocks} blocks | selected {len(chosen)} blocks: "
          f"{len(chosen) - len(todo)} complete, {len(todo)} to run ({todo_tasks:,} tasks, about ${todo_tasks * a.cost_per_call:,.2f})")
    if a.dry_run:
        print("dry run: nothing was sent.")
        return 0
    if not todo:
        print("✅ nothing to do: every selected block is complete.")
        return 0
    if not a.yes:
        print("❌ this would make real API calls. Re-run with --yes to confirm (or --dry_run to just look).")
        return 2

    failed = []
    for b in todo:
        paths = block_paths(a.plan_dir, b)
        if sha256_file(paths["input"]) != plan["block_input_sha256"][b - 1]:
            print(f"❌ block {b}: the block input file does not match the plan (it was altered); refusing to run it.")
            failed.append(b)
            continue
        started = datetime.now(timezone.utc).isoformat()
        attempts = 0
        st = block_status(a.plan_dir, b, by_block[b])
        while not st["complete"] and attempts <= a.retries:
            attempts += 1
            cmd = [sys.executable, str(a.agent), "--input", str(paths["input"]), "--output", str(paths["decisions"]),
                   "--resume", "--model", a.model, "--temperature", str(a.temperature)]
            t0 = time.time()
            with open(paths["log"], "a") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            st = block_status(a.plan_dir, b, by_block[b])
            print(f"  block {b:>3}/{n_blocks}: attempt {attempts}, exit {rc}, {len(st['valid'])}/{st['n']} valid "
                  f"({time.time() - t0:,.0f}s)")
        if st["complete"]:
            paths["window"].write_text(json.dumps({"block": b, "started_at": started,
                                                   "finished_at": datetime.now(timezone.utc).isoformat(),
                                                   "attempts": attempts}, indent=2))
        else:
            failed.append(b)
            print(f"  ❌ block {b} is still incomplete after {attempts} attempt(s): {len(st['missing'])} missing, "
                  f"{len(st['invalid'])} invalid, {len(st['bad_hash'])} on the wrong input. See {paths['log']}.")
    if failed:
        print(f"\n❌ {len(failed)} block(s) incomplete: {failed}. Re-run the same command to retry them.")
        return 1
    print(f"\n✅ all {len(todo)} selected block(s) complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
