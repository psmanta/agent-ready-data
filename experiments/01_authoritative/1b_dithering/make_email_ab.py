#!/usr/bin/env python3
"""
Email A/B smoke test -- step 1: build the two arms.
====================================================
Question: did the old `@example.com` emails act as a "this is test data" cue? Answer it with a controlled A/B on
the owner's EXISTING smoke customers, changing ONLY the email field:

    old_emails/   a verbatim copy of the existing smoke baseline input (the control arm, re-run now)
    new_emails/   the same records with each email replaced by the new realistic address

Why patch the existing input instead of regenerating: Faker computes `dob` from today's date, so regenerating the
customers would change `dob` too and blur the comparison. Here every other field is byte-identical (asserted).

The new email is exactly what the new generator would have produced for that customer (asserted against a fresh
generation), so the A/B tests the real change.

Usage (from 1b_dithering/):
    python3 make_email_ab.py --old_dir experiments_output --out_dir experiments_output_email_ab --seed 777
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "shared" / "data_generation"))
import base_customer_generator as G  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old_dir", type=Path, default=Path("experiments_output"), help="the existing smoke experiments_output")
    ap.add_argument("--out_dir", type=Path, default=Path("experiments_output_email_ab"))
    ap.add_argument("--seed", type=int, default=777, help="the base seed the smoke customers were generated with")
    a = ap.parse_args()

    if not hasattr(G, "synthetic_email"):
        print("❌ base_customer_generator.py has no synthetic_email(): copy the NEW generator into shared/data_generation/ first.")
        return 2
    canon_path = a.old_dir / "ground_truth" / "canonical_customers.json"
    in_path = a.old_dir / "baseline" / "agent_input" / "baseline_customers.jsonl"
    map_path = a.old_dir / "baseline" / "record_id_map.json"
    for p in (canon_path, in_path, map_path):
        if not p.exists():
            print(f"❌ not found: {p}")
            return 2
    canonical = json.load(open(canon_path))
    rid_to_cid = json.load(open(map_path))
    records = [json.loads(line) for line in open(in_path) if line.strip()]
    name_of = {c["customer_id"]: c["name"] for c in canonical}

    # 1. confirm the seed: names do not depend on the date, so a fresh generation must reproduce them
    fresh = G.generate_base_customers(len(canonical), a.seed)
    if [c["name"] for c in fresh] != [c["name"] for c in canonical]:
        print(f"❌ --seed {a.seed} does not reproduce the smoke customers' names. Pass the seed the smoke data was generated with.")
        return 2
    fresh_email = {c["customer_id"]: c["email"] for c in fresh}

    # 2. patch only the email
    patched = []
    for r in records:
        cid = rid_to_cid[r["record_id"]]
        assert r["name"] == name_of[cid], f"{cid}: name mismatch"
        new_email = G.synthetic_email(r["name"], cid, a.seed + int(cid.split("_")[1]) - 1)
        assert new_email == fresh_email[cid], f"{cid}: patched email differs from the generator's own"
        p = dict(r)
        p["email"] = new_email
        assert {k: v for k, v in p.items() if k != "email"} == {k: v for k, v in r.items() if k != "email"}
        patched.append(p)

    old_dir, new_dir = a.out_dir / "old_emails", a.out_dir / "new_emails"
    for d in (old_dir, new_dir):
        d.mkdir(parents=True, exist_ok=True)
    (old_dir / "baseline_customers.jsonl").write_bytes(in_path.read_bytes())
    (new_dir / "baseline_customers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in patched))
    assert [r["record_id"] for r in records] == [r["record_id"] for r in patched]
    assert not any("example.com" in r["email"] for r in patched) and all("example.com" in r["email"] for r in records), \
        "expected the old arm to have example.com and the new arm to have none"
    manifest = {"seed": a.seed, "n_customers": len(records), "only_field_changed": "email",
                "old_input_sha256": sha(old_dir / "baseline_customers.jsonl"), "new_input_sha256": sha(new_dir / "baseline_customers.jsonl")}
    (a.out_dir / "ab_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✅ {len(records)} customers. Only the email differs (asserted for every record).")
    print(f"   old arm : {records[0]['email']}   ...   {records[1]['email']}")
    print(f"   new arm : {patched[0]['email']}   ...   {patched[1]['email']}")
    print(f"\nNext: run the agent 5 times on EACH arm (see the runbook), then compare_email_ab.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
