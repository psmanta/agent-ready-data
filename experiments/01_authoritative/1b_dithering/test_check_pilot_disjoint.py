#!/usr/bin/env python3
"""
Tests for check_pilot_disjoint.py and the generator's generation_manifest.json. No API calls; small datasets generated in
a temporary directory with the real generator. Run from 1b_dithering/:

    python3 test_check_pilot_disjoint.py
"""
import atexit, contextlib, io, json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_pilot_disjoint as C

WORK = Path(tempfile.mkdtemp(prefix="disjoint_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")

def gen(name, *args):
    out = WORK / name
    r = subprocess.run([sys.executable, str(HERE / "generate_dithered_data.py"), "--out", str(out), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    return out, r.stdout

def failed(results):
    return [label for label, passed, _ in results if not passed]

def cli(eo, *extra):
    return subprocess.run([sys.executable, str(HERE / "check_pilot_disjoint.py"), "--experiments_output", str(eo), *map(str, extra)], capture_output=True, text=True)

print("M1  the generator writes its manifest")
A, out = gen("a", "--n", 20, "--seed", 5000)
m = json.load(open(A / C.MANIFEST_NAME)); r = m["runs"][0]
assert r["seed"] == 5000 and r["n"] == 20 and r["mode"] == "full" and r["seed_was_explicit"] is True; ok("one run recorded: seed, n, mode, and that the seed was explicit")
assert "ground_truth/canonical_customers.json" in r["files"] and any(f.endswith("agent_input.jsonl") for f in r["files"]) and not any("decisions" in f for f in r["files"]); ok(f"{len(r['files'])} generated files hashed (ground truth, baseline, every condition); no agent output")
assert all(v for v in r["code_sha256"].values()) and set(r["code_sha256"]) >= {"generate_dithered_data.py", "base_customer_generator.py", "dither_engine.py"}; ok("the generator and engine code hashes are recorded")
_, out2 = gen("a", "--n", 20, "--seed", 5000, "--baseline-only")
assert len(json.load(open(A / C.MANIFEST_NAME))["runs"]) == 2; ok("a second run is appended, never overwriting the history")

print("D1  a clean dataset passes")
B, _ = gen("b", "--n", 20, "--seed", 5000)
res = C.run_checks(B, 5000, 20, C.PILOTS)
assert not failed(res), failed(res); ok(f"all {len(res)} checks pass for seed 5000, n 20 against the default pilots (777/30, 42/100)")
p = cli(B, "--expected_seed", 5000, "--expected_n", 20)
assert p.returncode == 0 and "checks passed" in p.stdout; ok("the command line exits 0 and says so")

print("D2  each failure is caught")
res = C.run_checks(A, 5000, 20, C.PILOTS)
assert any("partial regeneration" in f for f in failed(res)); ok("a partial regeneration after the last full run is flagged")
assert any("seed 20261016" in f for f in failed(C.run_checks(B, 20261016, 20, C.PILOTS))); ok("data generated with a different seed than expected fails the seed check")
D, out = gen("d", "--n", 20)                                             # forgot --seed: the default, 42
assert "WARNING: no --seed given" in out; ok("the generator warns out loud when --seed is forgotten")
fd = failed(C.run_checks(D, 42, 20, C.PILOTS))
assert any("explicitly" in f for f in fd) and any("seed 42, n 100" in f for f in fd); ok("a forgotten --seed fails twice: not explicit, and it shares customers with the seed-42 pilot")
E, _ = gen("e", "--n", 20, "--seed", 760)                                # per-customer seeds 760..779 overlap 777..806
res = C.run_checks(E, 760, 20, C.PILOTS)
detail = next(d for l, p_, d in res if "seed 777" in l)
assert any("seed 777" in f for f in failed(res)) and detail.startswith("3 customer"); ok("a NEARBY seed is caught: seed 760 with n 20 shares exactly 3 customers with the seed-777 pilot (seeds 777, 778, 779)")
F = WORK / "f"; shutil.copytree(B, F)
cond = next(F.glob("conditions/*/agent_input.jsonl")); cond.write_text(cond.read_text().replace("MEDIUM", "MEDIUN", 1) + "\n")
assert any("unchanged" in f for f in failed(C.run_checks(F, 5000, 20, C.PILOTS))); ok("a condition file edited after generation fails the integrity check")
G = WORK / "g"; shutil.copytree(B, G)
other, _ = gen("other", "--n", 20, "--seed", 5001)
shutil.copy(other / "ground_truth" / "canonical_customers.json", G / "ground_truth" / "canonical_customers.json")
mg = json.load(open(G / C.MANIFEST_NAME)); mg["runs"][-1]["files"]["ground_truth/canonical_customers.json"] = C.sha256_file(G / "ground_truth" / "canonical_customers.json")
(G / C.MANIFEST_NAME).write_text(json.dumps(mg))
assert any("reproduces" in f for f in failed(C.run_checks(G, 5000, 20, C.PILOTS))); ok("data from another seed fails provenance even when the manifest is doctored to match it")
H = WORK / "h"; shutil.copytree(B, H); (H / C.MANIFEST_NAME).unlink()
assert failed(C.run_checks(H, 5000, 20, C.PILOTS)) == ["manifest: generation_manifest.json exists"]; ok("no manifest, no pass")
assert cli(E, "--expected_seed", 760, "--expected_n", 20).returncode == 1; ok("the command line exits 1 on any failure")

print("K1  why profiles, not ids")
p42, p43 = C.population(42, 100), C.population(43, 100)
k42 = {C.profile_key(c) for c in p42}
assert sum(C.profile_key(c) in k42 for c in p43) == 99; ok("seed 43 repeats 99 of seed 42's 100 customers (customer k comes from seed + k): the comparison sees it")
assert len({c["customer_id"] for c in p42} & {c["customer_id"] for c in C.population(5000, 100)}) == 100; ok("comparing customer_ids would be meaningless: two unrelated populations share all 100 ids")

print(f"\n{n_checks} check groups passed")
