#!/usr/bin/env python3
"""
check_pilot_disjoint.py: is the confirmatory data what we say it is, and does it share no customer with the pilots?
==================================================================================================================
Run after generating the confirmatory data and before the collection freeze (and again before the first agent call).
Exits 0 only if every check passes; any failure exits 1 with the reason. Read-only.

    python3 check_pilot_disjoint.py                                   # defaults: experiments_output, seed 20261016, n 1000
    python3 check_pilot_disjoint.py --experiments_output experiments_output --expected_seed 20261016 --expected_n 1000

Checks
  1. Manifest: experiments_output/generation_manifest.json exists, and EVERY generation run recorded in it used the
     expected seed and n, with the seed passed explicitly (the generator's default seed, 42, is a pilot seed).
  2. Integrity: every file the last full run hashed is unchanged (nothing was edited or regenerated without a record).
  3. Provenance: regenerating the expected seed and n reproduces canonical_customers.json exactly, so the seed claim
     rests on the data itself, not only on what the manifest says.
  4. Disjointness: no confirmatory customer appears in any pilot population (by default seed 777 with 30 customers and
     seed 42 with 100, the two populations pilot_population_audit.txt found).

How customers are compared. Customer k of a run is generated from the integer seed + k, so two runs whose seeds are close
share customers (seed 43 repeats 99 of seed 42's first 100). Only customer_id (position) and email (which hashes the
customer_id) differ for the same per-customer seed; every other field is identical. Customers are therefore compared on
their full profile WITHOUT customer_id and email. Comparing ids would be meaningless: CUST_000001 exists in every run.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "shared" / "data_generation"))

CONFIRMATORY_SEED = 20261016
CONFIRMATORY_N = 1000
PILOTS: List[Tuple[int, int]] = [(777, 30), (42, 100)]          # (seed, n), from pilot_population_audit.txt
ID_DEPENDENT_FIELDS = ("customer_id", "email")
MANIFEST_NAME = "generation_manifest.json"


def profile_key(customer: Dict[str, Any]) -> str:
    """A customer's identity independent of its position in a run."""
    core = {k: v for k, v in customer.items() if k not in ID_DEPENDENT_FIELDS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def population(seed: int, n: int) -> List[Dict[str, Any]]:
    from base_customer_generator import generate_base_customers
    return generate_base_customers(n=n, seed=seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_checks(eo: Path, expected_seed: int, expected_n: int, pilots: List[Tuple[int, int]]) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    def check(label, passed, detail=""):
        results.append((label, bool(passed), detail))
        return bool(passed)

    # 1. manifest
    mpath = eo / MANIFEST_NAME
    if not check("manifest: generation_manifest.json exists", mpath.exists(), f"not found: {mpath}"):
        return results
    runs = json.load(open(mpath)).get("runs", [])
    check("manifest: at least one generation run is recorded", runs, "the manifest lists no runs")
    wrong = [i + 1 for i, r in enumerate(runs) if r.get("seed") != expected_seed or r.get("n") != expected_n]
    check(f"manifest: every recorded run used seed {expected_seed} and n {expected_n}", not wrong,
          f"run(s) {wrong} used " + ", ".join(f"seed {runs[i - 1].get('seed')} n {runs[i - 1].get('n')}" for i in wrong))
    implicit = [i + 1 for i, r in enumerate(runs) if not r.get("seed_was_explicit")]
    check("manifest: the seed was passed explicitly in every run (never the default)", not implicit, f"run(s) {implicit} relied on the default seed")
    full = [r for r in runs if r.get("mode") == "full"]
    if not check("manifest: a full generation run is recorded", full, "only partial runs (baseline-only or single condition) are recorded"):
        return results

    # 2. integrity
    last = full[-1]
    changed = [f for f, h in last.get("files", {}).items() if not (eo / f).exists() or sha256_file(eo / f) != h]
    check(f"integrity: all {len(last.get('files', {}))} files hashed by the last full run are unchanged", not changed,
          f"{len(changed)} changed or missing, e.g. {changed[:3]}")
    later = [r for r in runs[runs.index(last) + 1:]]
    check("integrity: no partial regeneration after the last full run", not later,
          f"{len(later)} later partial run(s): {[r.get('mode') + ':' + str(r.get('condition')) for r in later]}")

    # 3. provenance
    gt_path = eo / "ground_truth" / "canonical_customers.json"
    if not check("provenance: ground_truth/canonical_customers.json exists", gt_path.exists(), f"not found: {gt_path}"):
        return results
    gt = json.load(open(gt_path))
    gt = gt["customers"] if isinstance(gt, dict) and "customers" in gt else gt
    check(f"provenance: the data holds {expected_n} customers", len(gt) == expected_n, f"found {len(gt)}")
    regenerated = population(expected_seed, expected_n)
    check(f"provenance: regenerating seed {expected_seed} with n {expected_n} reproduces canonical_customers.json exactly",
          [profile_key(c) for c in gt] == [profile_key(c) for c in regenerated] and
          [c.get("customer_id") for c in gt] == [c.get("customer_id") for c in regenerated],
          "the data does not match its claimed seed and n")

    # 4. disjointness
    mine: Set[str] = {profile_key(c) for c in gt}
    for seed, n in pilots:
        theirs = {profile_key(c) for c in population(seed, n)}
        shared = mine & theirs
        check(f"disjoint: no customer shared with the pilot population seed {seed}, n {n} "
              f"(per-customer seeds {seed}..{seed + n - 1}; ours {expected_seed}..{expected_seed + expected_n - 1})",
              not shared, f"{len(shared)} customer(s) shared")
    return results


def parse_pilots(spec: str) -> List[Tuple[int, int]]:
    out = []
    for part in spec.split(","):
        s, n = part.split(":")
        out.append((int(s), int(n)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_output", type=Path, default=HERE / "experiments_output")
    ap.add_argument("--expected_seed", type=int, default=CONFIRMATORY_SEED)
    ap.add_argument("--expected_n", type=int, default=CONFIRMATORY_N)
    ap.add_argument("--pilots", type=str, default=",".join(f"{s}:{n}" for s, n in PILOTS), help="seed:n,seed:n")
    a = ap.parse_args()
    results = run_checks(a.experiments_output, a.expected_seed, a.expected_n, parse_pilots(a.pilots))
    for label, passed, detail in results:
        print(f"  {'✅' if passed else '❌'} {label}" + ("" if passed or not detail else f"\n       {detail}"))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed" + ("" if not failed else ": NOT safe to run the confirmatory collection"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
