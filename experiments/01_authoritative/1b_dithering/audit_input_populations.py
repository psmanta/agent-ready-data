#!/usr/bin/env python3
"""
audit_input_populations.py - which seed produced the customers that every run saw?

Why: the generator prints its seed to the console and does not write it anywhere, so the seed of
a directory of agent inputs cannot be read off the files. This tool recovers it by regenerating each
candidate seed at the same size and comparing the customers. It also lists what every run
fingerprint (*.meta.jsonl) recorded about its input. Read-only: it writes nothing but temp files.

Usage (from 1b_dithering/):
    python3 audit_input_populations.py --root . --generator generate_dithered_data.py
    python3 audit_input_populations.py --root . --seeds 777 42 20261016 43
"""
import argparse, hashlib, json, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

STRONG = ("record_id", "name", "customer_segment", "acquisition_channel")
WEAK = ("record_id", "customer_segment", "acquisition_channel")
INPUT_NAMES = ("baseline_customers.jsonl", "agent_input.jsonl")
THRESHOLD = 0.90


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def keys(rows, fields):
    """Per-customer keys. record_id is BASE_<hash> in a baseline file and COND_<hash> in a condition file for the
    same customer, so only the part after the prefix is compared. NOTE: that part is position-based and identical
    across seeds, so it cannot identify a seed by itself; the name, segment and channel do the discriminating."""
    def one(r, f):
        v = r.get(f)
        return v.split("_", 1)[1] if f == "record_id" and isinstance(v, str) and "_" in v else v
    return [tuple(one(r, f) for f in fields) for r in rows]


def reference(generator, n, seed, cache):
    """Customers a clean generation with this seed and size would produce (None if it cannot be generated)."""
    if (n, seed) not in cache:
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(generator), "--n", str(n), "--seed", str(seed), "--baseline-only", "--out", tmp],
                                 capture_output=True, text=True)
            path = Path(tmp) / "baseline" / "agent_input" / "baseline_customers.jsonl"
            cache[(n, seed)] = read_rows(path) if run.returncode == 0 and path.exists() else None
    return cache[(n, seed)]


def display_hash(record):
    """Exactly the agent's own input_hash: md5 of the record as the model sees it (record_id excluded)."""
    shown = {k: v for k, v in record.items() if k != "record_id"}
    return hashlib.md5(json.dumps(shown, sort_keys=True, default=str).encode()).hexdigest()


def hashes(rows):
    return {display_hash(r) for r in rows}


def classify(rows, generator, seeds, cache):
    """-> (best seed or None, strong match fraction, weak match fraction)"""
    best = (None, 0.0, 0.0)
    for seed in seeds:
        ref = reference(generator, len(rows), seed, cache)
        if ref is None:
            continue
        strong = len(set(keys(rows, STRONG)) & set(keys(ref, STRONG))) / len(rows)
        weak = len(set(keys(rows, WEAK)) & set(keys(ref, WEAK))) / len(rows)
        # a condition that corrupted names or emails still shares the customer hash, so the weaker key is a valid fallback
        if max(strong, weak) > max(best[1], best[2]):
            best = (seed, strong, weak)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--generator", default="generate_dithered_data.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 42, 20261016])
    a = ap.parse_args(argv)
    root, generator = Path(a.root), Path(a.generator)
    if not generator.exists():
        print(f"generator not found: {generator}"); return 2

    cache, found, populations = {}, [], Counter()
    for p in sorted(root.rglob("*.jsonl")):
        if p.name in INPUT_NAMES and "_contract" not in p.parts:
            found.append(p)
    print(f"Found {len(found)} agent input files under {root}\n")
    by_dir = defaultdict(Counter)
    inputs = []   # (path, customer hashes, label) for tying run outputs back to a population
    for p in found:
        rows = read_rows(p)
        if not rows:
            continue
        seed, strong, weak = classify(rows, generator, a.seeds, cache)
        ok = seed is not None and max(strong, weak) >= THRESHOLD
        label = f"seed {seed}" if ok else "NO MATCH"
        populations[(len(rows), label)] += 1
        inputs.append((p, hashes(rows), label))
        by_dir[str(p.parent.parent.parent)][(len(rows), label)] += 1
        if not ok:
            print(f"  ?  {p}  n={len(rows)}  best guess seed {seed}  strong {strong:.0%} weak {weak:.0%}")

    print("\nBy top-level directory (n, source):")
    for d, c in sorted(by_dir.items()):
        print(f"  {d:<58} " + ", ".join(f"n={n} {lab} x{k}" for (n, lab), k in sorted(c.items())))

    print("\nRun fingerprints (*.meta.jsonl): what each run recorded about its input")
    metas = sorted(p for p in root.rglob("*.meta.jsonl") if "_contract" not in p.parts)
    if not metas:
        print("  none found (runs made before the fingerprint existed leave no record)")
    for m in metas:
        seen = set()
        for line in open(m, encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("input_file"):
                seen.add((e["input_file"], e.get("max_records"), (e.get("input_file_sha256") or "")[:10]))
        for f, mx, sha in sorted(seen, key=str):
            src = "input file no longer present"
            if Path(f).exists() or (root / f).exists():
                rows = read_rows(Path(f) if Path(f).exists() else root / f)
                seed, strong, weak = classify(rows, generator, a.seeds, cache) if rows else (None, 0, 0)
                src = f"n={len(rows)} " + (f"seed {seed}" if seed is not None and max(strong, weak) >= THRESHOLD else "NO MATCH")
            print(f"  {m.name[:46]:<46} max_records={mx}  sha {sha}  -> {src}")

    print("\nRun outputs: real agent runs versus fixtures, and which population each saw")
    print("  (a FIXTURE is a results file with placeholder rows or no model field: plumbing tests that no model ever saw)")
    skip = set(INPUT_NAMES)
    runs = Counter()
    for p in sorted(root.rglob("*.jsonl")):
        if p.name in skip or p.name.endswith(".meta.jsonl") or "_contract" in p.parts:
            continue
        try:
            head = json.loads(next(l for l in open(p, encoding="utf-8") if l.strip()))
        except (StopIteration, ValueError):
            continue
        if not (isinstance(head, dict) and "record_id" in head and ("business_decision" in head or "agent_confidence" in head)):
            continue
        rows = read_rows(p)
        placeholder = sum("placeholder" in str(r.get("decision_reasoning", "")).lower() for r in rows)
        with_model = sum(1 for r in rows if r.get("model"))
        kind = "FIXTURE" if placeholder / len(rows) >= 0.5 or with_model == 0 else "REAL"
        ids = {r["input_hash"] for r in rows if r.get("input_hash")}
        if not ids:
            runs[f"{kind}, cannot be tied to an input (no input_hash)"] += 1
            print(f"  {p}  ({len(rows)} rows) -> {kind}, cannot be tied to an input (rows carry no input_hash)")
            continue
        labs = sorted({lab for (_, hs, lab) in inputs if len(ids & hs) / len(ids) >= THRESHOLD})
        verdict = "/".join(labs) if labs else "UNRESOLVED (the exact input the run read is not on disk, or was regenerated since)"
        runs[f"{kind} {verdict}"] += 1
        if kind == "FIXTURE" or labs != ["seed 777"]:
            print(f"  {p}  ({len(rows)} rows, {placeholder} placeholder, {with_model} with a model) -> {kind}: {verdict}")
    if not runs:
        print("  no results files recognised (rows with record_id and business_decision)")

    print("\nVERDICT")
    labels = {lab for (_, lab) in populations}
    print("  inputs : " + ("every input file matches seed 777: " + ", ".join(f"n={n} x{k}" for (n, _), k in sorted(populations.items()))
                           if labels <= {"seed 777"} and populations else "NOT all from seed 777: " + ", ".join(f"n={n} {lab} x{k}" for (n, lab), k in sorted(populations.items()))))
    print("  runs   : " + (", ".join(f"{v} x{k}" for v, k in sorted(runs.items())) if runs else "nothing to report"))
    real = {v.split(" ", 1)[1] for v in runs if v.startswith("REAL ")}
    print("  REAL agent runs saw: " + ("nothing found" if not real else ("only seed 777" if real <= {"seed 777"} else "NOT only seed 777: " + ", ".join(sorted(real)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
