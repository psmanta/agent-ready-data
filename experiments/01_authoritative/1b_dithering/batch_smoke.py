#!/usr/bin/env python3
"""
Batch-mode smoke test (exploratory; pilot customers only; disclosed, never confirmatory).
========================================================================================
Two questions before the main run uses the Message Batches API:

  1. Does the plumbing work with real calls?  (submit, wait, collect, rows identical in schema to the standard path)
  2. Is ONE BATCH still ONE WINDOW?  The collection design treats a block as a window: inside a window the agent has been
     perfectly repeatable (October smoke check: all five runs agreed for all 30 customers); across windows, identical input
     flipped for 3 of 30. In batch mode Anthropic decides when each request runs, so we measure it: put five copies of
     each customer in one batch and see whether they still agree.

Design (450 calls, well under $1): on the 30 seed-777 pilot customers, in this order, each step starting after the last ends
    B1  one batch, 30 customers x 5 copies
    S   standard mode, 30 customers x 5 runs, one call at a time (the pilot's way, today)
    B2  a second batch, 30 x 5
Reported: within-B1, within-B2 and within-S agreement (do a customer's five copies agree?); B1 vs B2 and batch vs S
(majority decision per customer); today vs the 3 October pilot baseline (context only: different day, different window).

    python3 batch_smoke.py --dry_run          # what would be sent, and the cost; nothing is sent
    python3 batch_smoke.py --yes              # run it (resumable: re-run the same command after an interruption)
    python3 batch_smoke.py --report           # re-print the report from what is on disk

Uses only experiments_output/baseline (the seed-777 pilot population); writes to experiments_output/batch_smoke/.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import business_decision_agent as A  # noqa: E402
import batch_runner as BR  # noqa: E402
from collection_common import read_jsonl, VALID_DECISIONS  # noqa: E402

COPIES = 5
PILOT_SEED_NOTE = "seed 777, 30 customers (the pilot population; never the confirmatory seed 20261016)"


def copies(records, tag):
    """Five copies of each record; only record_id differs, and the model never sees record_id."""
    out = []
    for r in records:
        for k in range(1, COPIES + 1):
            out.append({**r, "record_id": f"{tag}_{r['record_id']}_c{k}"})
    return out


def base_id(rid: str) -> str:
    return rid.split("_", 1)[1].rsplit("_c", 1)[0]


def decisions_by_customer(rows):
    by = {}
    for r in rows:
        if r.get("business_decision") in VALID_DECISIONS:
            by.setdefault(base_id(r["record_id"]), []).append(r["business_decision"])
    return by


def majority(ds):
    c = Counter(ds).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "TIE"


def step_batch(tag, records, out, a):
    p = out / tag
    p.mkdir(parents=True, exist_ok=True)
    dec = p / "decisions.jsonl"
    todo = copies(records, tag)
    for attempt in range(3):
        have = {r["record_id"] for r in read_jsonl(dec, tolerant=True) if r.get("business_decision") in VALID_DECISIONS}
        need = [r for r in todo if r["record_id"] not in have]
        state_open = any(not b.get("collected") for b in BR.load_state(p / "batches.json").get("batches", []))
        if not need and not state_open:
            break
        inp = p / "input.jsonl"
        inp.write_text("".join(json.dumps(r) + "\n" for r in todo))
        BR.run_batch(need, decisions_path=dec, state_path=p / "batches.json", meta_path=p / "decisions.meta.jsonl",
                     input_path=inp, model=a.model, temperature=0.0, poll_seconds=a.poll_seconds)


def step_standard(records, out, a):
    p = out / "S"
    p.mkdir(parents=True, exist_ok=True)
    agent = A.BusinessDecisionAgent(model=a.model, temperature=0.0)
    for k in range(1, COPIES + 1):
        inp = p / f"input_c{k}.jsonl"
        inp.write_text("".join(json.dumps({**r, "record_id": f"S_{r['record_id']}_c{k}"}) + "\n" for r in records))
        agent.process_file(inp, p / f"run{k}.decisions.jsonl", resume=True)


def report(out, records):
    pilot_dir = HERE / "experiments_output" / "baseline" / "decisions"
    pilot = {}
    for k in range(1, 6):
        for r in read_jsonl(pilot_dir / f"run{k}.decisions.jsonl"):
            if r.get("business_decision") in VALID_DECISIONS:
                pilot.setdefault(r["record_id"], []).append(r["business_decision"])
    arms = {"B1": decisions_by_customer(read_jsonl(out / "B1" / "decisions.jsonl", tolerant=True)),
            "S": decisions_by_customer([r for k in range(1, COPIES + 1) for r in read_jsonl(out / "S" / f"run{k}.decisions.jsonl", tolerant=True)]),
            "B2": decisions_by_customer(read_jsonl(out / "B2" / "decisions.jsonl", tolerant=True)),
            "pilot (3 Oct, standard)": pilot}
    ids = [r["record_id"] for r in records]
    rep = {"population": PILOT_SEED_NOTE, "copies": COPIES, "within": {}, "between": {}}
    print("\nWithin one window: do a customer's five copies agree?")
    for name, by in arms.items():
        complete = [i for i in ids if len(by.get(i, [])) == COPIES]
        agree = sum(len(set(by[i])) == 1 for i in complete)
        rep["within"][name] = {"customers_complete": len(complete), "all_five_agree": agree}
        print(f"  {name:<26} {agree:>2} of {len(complete)} customers: all five agree")
    print("\nBetween windows: does the majority decision change?")
    for x, y in (("B1", "B2"), ("B1", "S"), ("B2", "S"), ("S", "pilot (3 Oct, standard)")):
        both = [i for i in ids if len(arms[x].get(i, [])) == COPIES and len(arms[y].get(i, [])) == COPIES]
        diff = [i for i in both if majority(arms[x][i]) != majority(arms[y][i])]
        rep["between"][f"{x} vs {y}"] = {"customers": len(both), "majority_differs": len(diff), "which": diff}
        print(f"  {x + ' vs ' + y:<32} {len(diff):>2} of {len(both)} customers differ")
    print("\nRead it this way: if within-batch agreement is as high as within-S, a batch behaves like a"
          "\nwindow and the block design holds as written. If copies inside one batch disagree, requests inside a batch span"
          "\nwindows, and the clean re-run control (not the block) carries the correction. Exploratory either way; disclosed.")
    (out / "smoke_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--out", type=Path, default=HERE / "experiments_output" / "batch_smoke")
    ap.add_argument("--poll_seconds", type=float, default=30.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--report", action="store_true", help="only print the report from files on disk")
    a = ap.parse_args()

    records = read_jsonl(HERE / "experiments_output" / "baseline" / "agent_input" / "baseline_customers.jsonl")
    if len(records) != 30:
        raise SystemExit(f"expected the 30-customer pilot baseline, found {len(records)} records: refusing (this smoke test must not touch other populations)")
    if a.report:
        report(a.out, records)
        return 0
    n_batch, n_std = 2 * len(records) * COPIES, len(records) * COPIES
    est = n_batch * 0.00256 * 0.5 + n_std * 0.00256
    print(f"plan: B1 {len(records) * COPIES} batch calls, then S {n_std} standard calls, then B2 {len(records) * COPIES} batch calls; "
          f"about ${est:.2f}; population: {PILOT_SEED_NOTE}")
    if a.dry_run:
        print("dry run: nothing was sent.")
        return 0
    if not a.yes:
        print("❌ this makes real API calls. Re-run with --yes.")
        return 2
    import run_collection as RC
    RC.keep_awake(False)
    a.out.mkdir(parents=True, exist_ok=True)
    print("B1: first batch"); step_batch("B1", records, a.out, a)
    print("S: standard mode, one call at a time"); step_standard(records, a.out, a)
    print("B2: second batch"); step_batch("B2", records, a.out, a)
    report(a.out, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
