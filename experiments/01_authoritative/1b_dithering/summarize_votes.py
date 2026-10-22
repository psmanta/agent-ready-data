#!/usr/bin/env python3
"""
Vote summary: for chosen customers, how did each arm vote across its runs?

A tiny helper for follow-up checks on a few customers. Each --arm is LABEL=DIRECTORY, where the directory holds
run*.decisions.jsonl files.

    python3 summarize_votes.py --arm "old emails=DIR1" --arm "new emails=DIR2" --records BASE_AAAA BASE_BBBB
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(directory: Path):
    per = {}
    for f in sorted(directory.glob("run*.decisions.jsonl")):
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("business_decision") in ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"):
                    per.setdefault(r["record_id"], []).append(r)
    return per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=DIRECTORY (repeatable)")
    ap.add_argument("--records", nargs="*", help="record ids to show (default: every record in any arm)")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        label, _, d = spec.partition("=")
        if not d or not Path(d).is_dir():
            print(f"❌ --arm {spec!r}: not LABEL=DIRECTORY, or the directory does not exist")
            return 2
        arms.append((label, load(Path(d))))
    ids = a.records or sorted({rid for _, per in arms for rid in per})
    for rid in ids:
        print(rid)
        for label, per in arms:
            rows = per.get(rid, [])
            if not rows:
                print(f"   {label:<24} (no valid runs)")
                continue
            votes = ", ".join(f"{d.replace('_PRIORITY', '')} x{n}" for d, n in Counter(r["business_decision"] for r in rows).most_common())
            conf = sum(r.get("agent_confidence") or 0 for r in rows) / len(rows)
            print(f"   {label:<24} {votes:<26} mean conf {conf:.2f}   ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
