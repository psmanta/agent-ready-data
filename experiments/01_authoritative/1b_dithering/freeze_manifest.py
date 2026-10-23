#!/usr/bin/env python3
"""
Freeze manifest -- Experiment 1b pre-registration
=================================================
The Agentic Data Contract . Pillar 1: Authoritative

"Frozen" has to mean something checkable. This tool records, once, the git commit and the
sha256 of every artifact the amendment declares frozen, and later verifies that none of them has
changed. Two stages, because evaluators for the later hypotheses are written after data collection
but must be fixed before anyone looks at results:

  collection   what produces and measures the data: the agent and its prompt, the generator and
               engine, the measurement instruments, the pre-registered decision-rule tooling, the
               evaluators that already exist, and the amendment itself. Write BEFORE the baseline.
  analysis     evaluators for hypotheses not yet built (H5-H8), written before analysis. Write after
               they are committed and BEFORE any results are examined.

Usage (from 1b_dithering/):
    python freeze_manifest.py write  --stage collection
    python freeze_manifest.py verify --stage collection

`write` refuses: a missing artifact; an artifact with uncommitted changes (the manifest would name a
commit that does not contain what was hashed) unless --allow_dirty; and overwriting an existing
manifest unless the recorded content is identical. `verify` exits 1 and names every artifact that is
missing or changed. The amendment cannot contain its own hash; this manifest records it.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SHARED = "../../../shared/data_generation/"
ARTIFACTS: Dict[str, List[str]] = {
    "collection": [
        # the agent and its prompt
        "business_decision_agent.py",
        # data generation
        SHARED + "base_customer_generator.py", SHARED + "dither_engine.py", SHARED + "validate_dama_dimensions.py",
        "generate_dithered_data.py", "generate_h1_replication_conditions.py", "check_and_generate_h8b.py",
        # the window-aware collection (plan, run, collect) and its regression test
        "collection_common.py", "plan_collection.py", "run_collection.py", "collect_results.py", "test_collection.py",
        # batch mode (Message Batches API): one batch per block, and its regression test
        "batch_runner.py", "test_batch_runner.py",
        # the confirmatory population: seed, provenance and disjointness from the pilots
        "check_pilot_disjoint.py", "test_check_pilot_disjoint.py",
        # baseline aggregation and boundary expansion
        "aggregate_baseline.py", "refine_boundary_convergence.py", "run_boundary_expansion.py", "prepare_next_batch.py",
        # measurement instruments and pre-registered decision-rule tooling
        "evaluate_core.py", "classifier_prompt.py", "classify_detection_llm_judge.py", "extract_for_classifier.py",
        "field_redundancy.py", "h1_baseline_replication.py",
        # evaluators that exist, and their validators
        "evaluate_h1.py", "evaluate_h2.py", "evaluate_h3.py", "evaluate_h4.py",
        "customer_halves.py", "validate_h2_schema.py", "test_evaluate_h2.py",
        "validate_h1_schema.py", "validate_h3_schema.py", "validate_h4_schema.py", "test_evaluate_h3.py",
        # regression tests for the frozen tooling
        "test_h1_replication.py", "test_evaluate_h1_manifest.py", "test_evaluate_h1_strata.py", "check_amendment_consistency.py", "freeze_manifest.py", "test_freeze_manifest.py", "test_business_decision_agent.py",
        # the pre-registration itself
        "1b_DESIGN_AMENDMENT_1.md",
    ],
    "analysis": [
        "evaluate_h5.py", "validate_h5_schema.py", "test_evaluate_h5.py",
        "evaluate_h6.py", "evaluate_h7.py", "evaluate_h8.py",
    ],
}


def sha256_file(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git(base: Path, *args: str, keep_leading_space: bool = False) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=base, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        # `git status --porcelain` lines begin with a significant space (" M file"): never strip it
        return r.stdout.rstrip("\n") if keep_leading_space else r.stdout.strip()
    except FileNotFoundError:
        return None


def git_state(base: Path, paths: List[str]) -> Dict[str, Any]:
    commit = _git(base, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": [], "note": "not a git repository (or git unavailable): commit not recorded"}
    status = _git(base, "status", "--porcelain", "--", *paths, keep_leading_space=True) or ""
    dirty = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    return {"commit": commit, "dirty": dirty, "note": None}


def build(base: Path, stage: str, artifacts: Optional[List[str]] = None) -> Dict[str, Any]:
    paths = artifacts if artifacts is not None else ARTIFACTS[stage]
    git = git_state(base, paths)
    return {"freeze_version": "1b-freeze-v1", "stage": stage, "git_commit": git["commit"],
            "artifacts": {p: sha256_file(base / p) for p in paths}, "_dirty": git["dirty"], "_note": git["note"]}


def write(base: Path, stage: str, out: Path, allow_dirty: bool = False, artifacts: Optional[List[str]] = None) -> int:
    m = build(base, stage, artifacts)
    missing = sorted(p for p, h in m["artifacts"].items() if h is None)
    if missing:
        print(f"❌ cannot freeze: {len(missing)} artifact(s) missing:\n  - " + "\n  - ".join(missing))
        return 1
    if m["_dirty"] and not allow_dirty:
        print("❌ cannot freeze: these artifacts have uncommitted changes, so the recorded commit would not contain "
              "what was hashed. Commit them first (or --allow_dirty for a rehearsal):\n  - " + "\n  - ".join(m["_dirty"]))
        return 1
    note, dirty = m.pop("_note"), m.pop("_dirty")
    if note:
        print(f"  ⚠️  {note}")
    if out.exists():
        old = json.load(open(out))
        if old == m:
            print(f"Already frozen with IDENTICAL content ({out}); left untouched.")
            return 0
        print(f"❌ {out} already exists with DIFFERENT content. A freeze is write-once: if this is a deliberate "
              f"rehearsal, delete the file first; if not, something frozen has changed (run `verify`).")
        return 1
    out.write_text(json.dumps(m, indent=2, sort_keys=True))
    print(f"✅ frozen ({stage}): {len(m['artifacts'])} artifacts at commit {str(m['git_commit'])[:12]} -> {out}")
    return 0


def verify(base: Path, stage: str, out: Path, artifacts: Optional[List[str]] = None) -> int:
    if not out.exists():
        print(f"❌ no freeze manifest at {out}")
        return 1
    rec = json.load(open(out))
    problems = []
    for p, h in rec["artifacts"].items():
        cur = sha256_file(base / p)
        if cur is None:
            problems.append(f"{p}: MISSING")
        elif cur != h:
            problems.append(f"{p}: CHANGED since the freeze")
    if rec.get("stage") != stage:
        problems.append(f"manifest is for stage {rec.get('stage')!r}, not {stage!r}")
    if problems:
        print(f"❌ {len(problems)} frozen artifact(s) no longer match {out}:\n  - " + "\n  - ".join(problems))
        return 1
    print(f"✅ all {len(rec['artifacts'])} frozen artifacts match ({stage}, commit {str(rec['git_commit'])[:12]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["write", "verify"])
    ap.add_argument("--stage", choices=sorted(ARTIFACTS), required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: freeze_manifest_<stage>.json next to this script")
    ap.add_argument("--allow_dirty", action="store_true")
    a = ap.parse_args()
    base = Path(__file__).resolve().parent
    out = a.out or (base / f"freeze_manifest_{a.stage}.json")
    return write(base, a.stage, out, a.allow_dirty) if a.action == "write" else verify(base, a.stage, out)


if __name__ == "__main__":
    sys.exit(main())
