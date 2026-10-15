#!/usr/bin/env python3
"""
Plan the collection as a randomized block design (macro-review decision D25).
===========================================================================
Every agent call in the experiment is a TASK: one of the five baseline runs of a customer, one record of a dithered
condition, or one record of the clean re-run control. The plan deals ALL tasks into BLOCKS (time windows) so that:

  * every block holds a random slice of every condition and of the baseline, so a window effect lands as noise on
    every condition instead of as a bias on whichever condition happened to run in a "bad" window;
  * each customer's five baseline runs land in five different blocks, spread across the whole collection (a fixed
    stride apart), so "stable" means "stable across windows", not "stable within one window";
  * blocks are the same size (to within a handful of tasks), and tasks are shuffled inside each block.

The plan is deterministic from its seed, written once, and never edited. It records the hash of every source input
file, of every block file and of the task list, so any later change is detected.

Usage (from 1b_dithering/, after generate_dithered_data.py):
    python3 plan_collection.py --experiments_output experiments_output --blocks 48 --seed 20261015
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collection_common import (PLAN_VERSION, block_name, input_hash, read_jsonl, sha256_file)  # noqa: E402


def build_plan(baseline: List[Dict[str, Any]], conditions: Dict[str, List[Dict[str, Any]]],
               blocks: int, baseline_runs: int, seed: int) -> List[Dict[str, Any]]:
    """The dealing algorithm, separated from file handling so it can be tested on its own.

    Returns task dicts: stratum, source_line, block (0-based), plus the source record."""
    if baseline_runs < 1 or blocks < baseline_runs:
        raise ValueError(f"need blocks >= baseline_runs (got {blocks} blocks, {baseline_runs} runs): each customer's runs "
                         "must land in different blocks")
    rng = random.Random(seed)
    stride = blocks // baseline_runs                      # runs of one customer sit a full stride apart
    tasks: List[Dict[str, Any]] = []

    # Baseline: customers in random order; customer i starts at block (offset + i) and its runs are one stride apart.
    order = list(range(len(baseline)))
    rng.shuffle(order)
    offset = rng.randrange(blocks)
    for i, cust in enumerate(order):
        start = (offset + i) % blocks
        homes = [(start + j * stride) % blocks for j in range(baseline_runs)]
        runs = list(range(1, baseline_runs + 1))
        rng.shuffle(runs)                                 # which run number lands in which window is random
        for run, blk in zip(runs, homes):
            tasks.append({"stratum": f"baseline_run_{run}", "source_line": cust, "rec": baseline[cust], "block": blk})

    # Conditions: random order, random record order, dealt round-robin with ONE continuing pointer, so every
    # condition (and the grand total) is spread over the blocks to within one task.
    ptr = rng.randrange(blocks)
    names = sorted(conditions)
    rng.shuffle(names)
    for name in names:
        recs = conditions[name]
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        for ix in idx:
            tasks.append({"stratum": f"cond:{name}", "source_line": ix, "rec": recs[ix], "block": ptr % blocks})
            ptr += 1
    return tasks


def write_plan(experiments_output: Path, out_dir: Path, blocks: int, baseline_runs: int, seed: int) -> Dict[str, Any]:
    eo = Path(experiments_output)
    plan_path = out_dir / "plan.json"
    if plan_path.exists():
        raise SystemExit(f"❌ {plan_path} already exists. A plan is written once and never edited; to start over, delete "
                         f"{out_dir} yourself (only if no decisions were collected from it).")
    base_path = eo / "baseline" / "agent_input" / "baseline_customers.jsonl"
    if not base_path.exists():
        raise SystemExit(f"❌ not found: {base_path}. Run generate_dithered_data.py first.")
    baseline = read_jsonl(base_path)
    cond_dirs = sorted(p for p in (eo / "conditions").iterdir() if (p / "agent_input.jsonl").exists()) \
        if (eo / "conditions").exists() else []
    if not baseline or not cond_dirs:
        raise SystemExit("❌ need a baseline and at least one condition with an agent_input.jsonl.")
    conditions = {p.name: read_jsonl(p / "agent_input.jsonl") for p in cond_dirs}
    for what, recs in [("baseline", baseline)] + list(conditions.items()):
        if not recs or any("record_id" not in r for r in recs):
            raise SystemExit(f"❌ {what}: empty, or a record has no record_id.")

    tasks = build_plan(baseline, conditions, blocks, baseline_runs, seed)

    by_block: Dict[int, List[Dict[str, Any]]] = {b: [] for b in range(blocks)}
    for t in tasks:
        by_block[t["block"]].append(t)
    rng = random.Random(seed + 1)                          # a separate stream for the within-block shuffle
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blocks").mkdir(exist_ok=True)
    task_lines, block_hashes, sizes, task_no = [], [], [], 0
    for b in range(blocks):
        group = by_block[b]
        rng.shuffle(group)
        lines = []
        for pos, t in enumerate(group, start=1):
            task_no += 1
            task_id = f"T{task_no:06d}"
            rec = dict(t["rec"])
            rec["record_id"] = task_id                      # keeps the key's original position; the model never sees it
            lines.append(json.dumps(rec, default=str))
            task_lines.append(json.dumps({"task_id": task_id, "block": b + 1, "pos": pos, "stratum": t["stratum"],
                                          "source_record_id": t["rec"]["record_id"], "source_line": t["source_line"],
                                          "input_hash": input_hash(t["rec"])}))
        bp = out_dir / "blocks" / f"{block_name(b + 1)}.input.jsonl"
        bp.write_text("".join(l + "\n" for l in lines))
        block_hashes.append(sha256_file(bp))
        sizes.append(len(group))
    (out_dir / "tasks.jsonl").write_text("".join(l + "\n" for l in task_lines))

    strata = [{"name": "baseline", "tasks": len(baseline) * baseline_runs, "records": len(baseline),
               "source": "baseline/agent_input/baseline_customers.jsonl", "source_sha256": sha256_file(base_path)}]
    for p in cond_dirs:
        strata.append({"name": f"cond:{p.name}", "tasks": len(conditions[p.name]), "records": len(conditions[p.name]),
                       "source": f"conditions/{p.name}/agent_input.jsonl", "source_sha256": sha256_file(p / "agent_input.jsonl")})
    plan = {"plan_version": PLAN_VERSION, "design": "randomized complete block; baseline runs one stride apart",
            "seed": seed, "blocks": blocks, "baseline_runs": baseline_runs, "stride": blocks // baseline_runs,
            "n_tasks": len(tasks), "block_sizes": sizes, "block_input_sha256": block_hashes,
            "tasks_sha256": sha256_file(out_dir / "tasks.jsonl"), "strata": strata}
    plan_path.write_text(json.dumps(plan, indent=2))       # last: a plan directory without plan.json is not a plan
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_output", type=Path, default=Path("experiments_output"))
    ap.add_argument("--out_dir", type=Path, default=None, help="default: <experiments_output>/collection")
    ap.add_argument("--blocks", type=int, default=48, help="number of windows to spread everything across (default 48)")
    ap.add_argument("--baseline_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--cost_per_call", type=float, default=0.00256, help="for the estimate only")
    ap.add_argument("--calls_per_minute", type=float, default=20.0, help="for the estimate only (one sequential worker)")
    a = ap.parse_args()
    out_dir = a.out_dir or (a.experiments_output / "collection")
    try:
        plan = write_plan(a.experiments_output, out_dir, a.blocks, a.baseline_runs, a.seed)
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    sizes = plan["block_sizes"]
    print(f"✅ plan written to {out_dir}")
    print(f"   {plan['n_tasks']:,} tasks in {plan['blocks']} blocks of {min(sizes):,} to {max(sizes):,} tasks "
          f"(seed {plan['seed']}); each customer's {plan['baseline_runs']} baseline runs are {plan['stride']} blocks apart")
    print(f"   estimate: about ${plan['n_tasks'] * a.cost_per_call:,.2f}, and about "
          f"{plan['n_tasks'] / a.calls_per_minute / 60:,.1f} hours for one sequential worker ({a.calls_per_minute:.0f} calls/min); "
          f"run_collection.py --shard splits the blocks across several workers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
