#!/usr/bin/env python3
"""
Generate the conditional H1 replication conditions -- Experiment 1b
===================================================================
The Agentic Data Contract . Pillar 1: Authoritative

Reads the FROZEN MANIFEST written by h1_baseline_replication.py. If the
pre-registered trigger fired, it generates one individual 15% condition for
each field in `decision.conditions_to_add`; otherwise it does nothing. It does
not re-evaluate the trigger: it verifies the manifest's `decision_sha256` and
refuses a manifest that was edited after it was frozen.

Why it is built the way it is
  * Customers come from the SAVED canonical_customers.json, never regenerated
    from n/seed (the same approach as check_and_generate_h8b.py). Faker derives
    date of birth from today's date, so regenerating on another day would give
    these conditions different customers than every other condition and the
    baseline. If the manifest recorded the canonical file's hash, this script
    refuses to proceed unless the file is unchanged.
  * Each condition is a clone of the existing individual condition
    h1_individual_nps_score (dataclasses.replace: only fields, seed and
    condition_id differ), so its parameters (15%, drift, correlated, recompute
    on) cannot drift from the twelve it is compared with.
  * Validity checks, record-id assignment and file formats are the main
    generator's own functions (imported, not copied), and the record-id set is
    cross-checked against an existing condition.
  * Existing files are never silently overwritten: identical content is left
    alone; different content aborts.
  * It writes h1_replication_generated.json next to the manifest: the manifest's
    hash, the canonical hash, and the hash of every file it generated.

This script does NOT run the agent. Run each new condition through
business_decision_agent.py (the commands are printed at the end).

Usage
    python generate_h1_replication_conditions.py \
        --manifest experiments_output/evaluation/h1_replication_manifest.json \
        --out experiments_output
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "shared" / "data_generation"))
sys.path.insert(0, str(_HERE))

from dither_engine import DitherEngine, build_all_conditions  # noqa: E402
from field_redundancy import DITHERED_FIELDS  # noqa: E402
import generate_dithered_data as gdd  # noqa: E402
import h1_baseline_replication as H  # noqa: E402

TEMPLATE_CONDITION_ID = "h1_individual_nps_score"
MIN_EXPOSURE = 0.80


def build_config(entry: Dict[str, Any]):
    """Clone the template individual condition; only fields, seed and id change."""
    template = next(c for c in build_all_conditions() if c.condition_id == TEMPLATE_CONDITION_ID)
    return dataclasses.replace(template, fields=[entry["field"]], seed=entry["seed"],
                               condition_id=entry["condition_id"])


def write_if_new(path: Path, content: str) -> str:
    """'written', 'unchanged', or raise: never silently overwrite different content."""
    if path.exists():
        if path.read_text() == content:
            return "unchanged"
        raise SystemExit(f"❌ {path} already exists with DIFFERENT content. Refusing to overwrite; "
                         f"investigate before proceeding.")
    path.write_text(content)
    return "written"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path, help="h1_replication_manifest.json")
    ap.add_argument("--out", type=Path, default=Path("experiments_output"))
    ap.add_argument("--allow-invalid", action="store_true",
                    help="write a condition even if it fails the perturbation validity check")
    a = ap.parse_args()

    try:
        rep = H.load_manifest(a.manifest)          # raises if edited after it was frozen
    except ValueError as e:
        raise SystemExit(f"❌ {e}")
    dec = rep["decision"]
    print(f"\n{'=' * 64}\nH1 conditional conditions ({rep['rule_version']})\n{'=' * 64}")
    print(f"manifest {rep['decision_sha256'][:12]}...: top-5 overlap with 1a {rep['replication']['top5_overlap']} of 5 "
          f"(trigger: <= {rep['frozen']['trigger_max_overlap']})")
    if not dec["triggered"]:
        print("Trigger did not fire. Question A stands as designed. Nothing to generate.")
        return 0
    to_add: List[Dict[str, Any]] = dec["conditions_to_add"]
    if not to_add:
        print("Trigger fired, but every field in the 1b top 5 already has an individual condition: "
              "nothing new to generate. Question A is reported on both lists using existing conditions.")
        return 0

    canonical = a.out / "ground_truth" / "canonical_customers.json"
    if not canonical.exists():
        raise SystemExit(f"❌ {canonical} not found. These conditions must dither the SAME customers as "
                         f"every other condition; do not regenerate them.")
    recorded = (rep["inputs"].get("canonical") or {}).get("sha256")
    actual = hashlib.sha256(canonical.read_bytes()).hexdigest()
    if recorded and recorded != actual:
        raise SystemExit("❌ canonical_customers.json has CHANGED since the baseline analysis recorded it "
                         f"({recorded[:12]}... -> {actual[:12]}...). Faker computes date of birth from today's "
                         "date, so a regenerated customer file differs from the one the baseline used. "
                         "Restore the original file; do not generate against a different one.")
    if not recorded:
        print("  ⚠️  the manifest did not record the canonical file's hash (--canonical was not passed): "
              "cannot verify it is unchanged since the baseline.")
    customers = json.load(open(canonical))
    print(f"Loaded {len(customers)} customers from {canonical} (sha256 {actual[:12]}...)")

    existing_ids = {c.condition_id for c in build_all_conditions()}
    configs = []
    for entry in to_add:
        if entry["field"] not in DITHERED_FIELDS:
            raise SystemExit(f"❌ {entry['field']} is not a field the engine can dither.")
        if entry["condition_id"] in existing_ids:
            raise SystemExit(f"❌ condition_id {entry['condition_id']} collides with an existing condition.")
        configs.append(build_config(entry))
    if len({c.condition_id for c in configs}) != len(configs):
        raise SystemExit("❌ duplicate condition_ids in the replication file.")

    reference_ids = None
    ref_path = a.out / "conditions" / TEMPLATE_CONDITION_ID / "dither_reference.json"
    if ref_path.exists():
        reference_ids = {r["record_id"] for r in json.load(open(ref_path))}

    validity_results, written, file_hashes = [], [], {}
    for config in configs:
        print(f"\n  [{config.condition_id}]  field={config.fields[0]}  seed={config.seed}")
        dithered = DitherEngine(config).apply(customers)
        validity = gdd.check_condition_validity(config, dithered, MIN_EXPOSURE)
        validity_results.append(validity)
        exp = ", ".join(f"{f}={e['raw_rate']:.0%}" for f, e in validity["field_exposure"].items())
        print(f"    exposure: {exp}")
        if not validity["passed"]:
            for msg in validity["failures"]:
                print(f"    ❌ INVALID: {msg}")
            if not a.allow_invalid:
                print("    -> NOT written")
                continue

        with_ids = gdd.assign_record_ids(dithered, prefix="COND")
        gdd.verify_uniqueness(with_ids, context=f"{config.condition_id} agent input")
        gdd.verify_customer_uniqueness(with_ids, context=f"{config.condition_id} dither reference")
        if reference_ids is not None and {r["record_id"] for r in with_ids} != reference_ids:
            raise SystemExit(f"❌ {config.condition_id}: record_ids differ from {TEMPLATE_CONDITION_ID}: "
                             f"not the same customers as the main conditions.")

        cond_dir = a.out / "conditions" / config.condition_id
        cond_dir.mkdir(parents=True, exist_ok=True)
        agent_facing = [gdd.strip_internal_fields(c) for c in with_ids]
        s1 = write_if_new(cond_dir / "agent_input.jsonl",
                          "".join(json.dumps(r, default=str) + "\n" for r in agent_facing))
        s2 = write_if_new(cond_dir / "dither_reference.json", json.dumps(with_ids, indent=2, default=str))
        n_d = sum(1 for c in dithered if c.get("_dither_applied"))
        print(f"    agent_input.jsonl ({len(agent_facing)} records, {s1}); dither_reference.json ({n_d} dithered, {s2})")
        written.append(config.condition_id)
        file_hashes[config.condition_id] = {
            "field": config.fields[0], "seed": config.seed,
            "agent_input_sha256": hashlib.sha256((cond_dir / "agent_input.jsonl").read_bytes()).hexdigest(),
            "dither_reference_sha256": hashlib.sha256((cond_dir / "dither_reference.json").read_bytes()).hexdigest()}

    report = a.out / "validity_report_h1_replication.json"
    report.write_text(json.dumps({"min_exposure": MIN_EXPOSURE, "n_conditions": len(validity_results),
                                  "n_failed": sum(not v["passed"] for v in validity_results),
                                  "canonical_sha256": actual, "conditions": validity_results}, indent=2))
    record = a.manifest.parent / "h1_replication_generated.json"
    record.write_text(json.dumps({"manifest_decision_sha256": rep["decision_sha256"], "canonical_sha256": actual,
                                  "conditions": file_hashes,
                                  "not_written": [v["condition_id"] for v in validity_results if not v["passed"]
                                                  and v["condition_id"] not in file_hashes]}, indent=2))
    failed = [v["condition_id"] for v in validity_results if not v["passed"]]
    if failed and not a.allow_invalid:
        raise SystemExit(f"\n❌ {len(failed)} condition(s) failed the validity check and were not written: "
                         f"{', '.join(failed)}. See {report}.")
    print(f"\n✅ {len(written)} condition(s) ready. Validity report: {report}; generation record: {record}")
    print("\nNEXT: run each through the agent (add --resume if a run is interrupted):")
    for cid in written:
        print(f"  python business_decision_agent.py --input {a.out}/conditions/{cid}/agent_input.jsonl "
              f"--output {a.out}/conditions/{cid}/decisions.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
