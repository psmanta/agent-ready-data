#!/usr/bin/env python3
"""
Route the block results back into the files the evaluators already read (macro-review decision D25).
===================================================================================================
After run_collection.py has completed the blocks:

    baseline run k      -> <experiments_output>/baseline/decisions/run<k>.decisions.jsonl
    each condition      -> <experiments_output>/conditions/<condition>/decisions.jsonl     (including h0_clean_rerun_<k>)

Every row gets its ORIGINAL record_id back (the block files carry unique task ids) and two extra fields,
collection_block and collection_task_id, so any result can be analysed by window. Rows keep source order.
A task counts only if it has a real priority on exactly the input the plan expects; a PARSE_ERROR, a missing row
or a row made on a different input is reported and nothing is written for that stratum.

    python3 collect_results.py --plan_dir experiments_output/collection --status     # progress; writes nothing
    python3 collect_results.py --plan_dir experiments_output/collection              # write (refuses if incomplete)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collection_common import block_paths, block_status, load_plan, read_jsonl  # noqa: E402


def output_path_for(experiments_output: Path, stratum: str) -> Path:
    if stratum.startswith("baseline_run_"):
        return experiments_output / "baseline" / "decisions" / f"run{stratum.rsplit('_', 1)[1]}.decisions.jsonl"
    return experiments_output / "conditions" / stratum.split(":", 1)[1] / "decisions.jsonl"


def collect(plan_dir: Path, experiments_output: Path, allow_incomplete: bool = False, overwrite: bool = False,
            status_only: bool = False) -> Dict[str, Any]:
    plan, by_block = load_plan(plan_dir)
    rows_by_task: Dict[str, Dict[str, Any]] = {}
    per_block, missing, invalid, bad_hash = {}, [], [], []
    for b in range(1, plan["blocks"] + 1):
        st = block_status(plan_dir, b, by_block[b])
        rows_by_task.update(st["valid"])
        per_block[b] = {"n": st["n"], "valid": len(st["valid"])}
        missing += st["missing"]; invalid += st["invalid"]; bad_hash += st["bad_hash"]
    tasks_by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for b in by_block:
        for t in by_block[b]:
            tasks_by_stratum.setdefault(t["stratum"], []).append(t)
    complete_strata = sorted(s for s, ts in tasks_by_stratum.items() if all(t["task_id"] in rows_by_task for t in ts))
    incomplete_strata = sorted(set(tasks_by_stratum) - set(complete_strata))
    report = {"n_tasks": plan["n_tasks"], "valid_tasks": len(rows_by_task), "missing": len(missing),
              "invalid": len(invalid), "wrong_input": len(bad_hash), "complete_strata": len(complete_strata),
              "incomplete_strata": incomplete_strata, "per_block": per_block, "written": []}
    if status_only:
        return report
    if incomplete_strata and not allow_incomplete:
        report["refused"] = True
        return report

    targets = {s: output_path_for(experiments_output, s) for s in complete_strata}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not overwrite:
        raise SystemExit(f"❌ {len(existing)} output file(s) already exist (for example {existing[0]}); "
                         "pass --overwrite only if you mean to replace them.")
    cost = 0.0
    for s in complete_strata:
        ts = sorted(tasks_by_stratum[s], key=lambda t: t["source_line"])
        out = targets[s]
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for t in ts:
                row = dict(rows_by_task[t["task_id"]])
                cost += row.get("cost_usd") or 0.0
                row["record_id"] = t["source_record_id"]
                row["collection_block"] = t["block"]
                row["collection_task_id"] = t["task_id"]
                f.write(json.dumps(row, default=str) + "\n")
        report["written"].append(str(out))
    windows = {}
    for b in range(1, plan["blocks"] + 1):
        w = block_paths(plan_dir, b)["window"]
        if w.exists():
            windows[b] = json.load(open(w))
    report["total_cost_usd"] = round(cost, 4)
    report["windows"] = windows
    (plan_dir / "collection_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan_dir", type=Path, default=Path("experiments_output/collection"))
    ap.add_argument("--experiments_output", type=Path, default=None, help="default: the plan directory's parent")
    ap.add_argument("--status", action="store_true", help="show progress and write nothing")
    ap.add_argument("--allow_incomplete", action="store_true", help="write the strata that ARE complete; list the rest")
    ap.add_argument("--overwrite", action="store_true", help="replace existing decisions files")
    a = ap.parse_args()
    eo = a.experiments_output or a.plan_dir.resolve().parent
    r = collect(a.plan_dir, eo, a.allow_incomplete, a.overwrite, a.status)
    pct = 100 * r["valid_tasks"] / r["n_tasks"]
    print(f"{r['valid_tasks']:,} of {r['n_tasks']:,} tasks have a valid decision ({pct:.1f}%): {r['missing']:,} missing, "
          f"{r['invalid']:,} invalid (PARSE_ERROR or unusable), {r['wrong_input']:,} made on the wrong input")
    print(f"{r['complete_strata']} strata complete; {len(r['incomplete_strata'])} incomplete")
    if a.status:
        return 0
    if r.get("refused"):
        print(f"❌ not writing: {len(r['incomplete_strata'])} stratum(s) incomplete, for example {r['incomplete_strata'][:3]}. "
              "Finish them with run_collection.py, or pass --allow_incomplete to write only the complete ones.")
        return 1
    print(f"✅ wrote {len(r['written'])} decisions file(s); total cost ${r['total_cost_usd']}; report: {a.plan_dir / 'collection_report.json'}")
    if r["incomplete_strata"]:
        print(f"   (skipped {len(r['incomplete_strata'])} incomplete stratum/strata)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
