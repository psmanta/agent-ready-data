#!/usr/bin/env python3
"""
Tests for the window-aware collection: plan_collection.py, run_collection.py, collect_results.py, and the clean re-run
control (macro-review decision D25).

No API key and no cost: a fake agent with the real command line stands in for the model. It builds its own small dataset
in a temporary directory with the real generator and never touches your real experiments_output. Run from 1b_dithering/:

    python3 test_collection.py

The last group is a SIMULATION of the thing the design exists for: a window effect (some windows flip many customers,
some few). A condition run in its own window inherits that window's rate (a bias that differs from condition to condition);
spread across many windows it gets the average (noise), and then the clean re-run control can be subtracted.
"""
import atexit, collections, contextlib, hashlib, io, json, os, random, shutil, subprocess, sys, tempfile, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

for _n in ("langchain_anthropic", "langchain_core", "langchain_core.messages"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["langchain_anthropic"].ChatAnthropic = type("ChatAnthropic", (), {"__init__": lambda self, *a, **k: None})
sys.modules["langchain_core.messages"].HumanMessage = sys.modules["langchain_core.messages"].SystemMessage = type("M", (), {"__init__": lambda self, *a, **k: None})
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
import business_decision_agent as A
import collection_common as CC
import plan_collection as PC
import collect_results as CR

WORK = Path(tempfile.mkdtemp(prefix="collection_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def sh(*args, env=None, check=False):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=e)
    if check: assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    return r
def silently(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a, **k)

N, B, R = 40, 12, 5
EO = WORK / "eo"
g = sh("generate_dithered_data.py", "--n", N, "--seed", 7, "--clean-reruns", 2, "--out", EO); assert g.returncode == 0, g.stdout[-400:] + g.stderr[-400:]
COND_NAMES = sorted(p.name for p in (EO / "conditions").iterdir())
N_COND = len(COND_NAMES)
BASE = CC.read_jsonl(EO / "baseline" / "agent_input" / "baseline_customers.jsonl")
SRC = {n: CC.read_jsonl(EO / "conditions" / n / "agent_input.jsonl") for n in COND_NAMES}

FAKE = WORK / "fake_agent.py"
FAKE.write_text(r'''
import argparse, hashlib, json, os, re, sys
from pathlib import Path
VALID = ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY")
def ihash(rec): return hashlib.md5(json.dumps({k: v for k, v in rec.items() if k != "record_id"}, sort_keys=True, default=str).encode()).hexdigest()
ap = argparse.ArgumentParser()
for f in ("--input", "--output", "--model", "--temperature", "--max_records"): ap.add_argument(f)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
inp, out = Path(a.input), Path(a.output)
window = int(re.search(r"block_(\d+)", out.name).group(1))
recs = [json.loads(l) for l in open(inp) if l.strip()]
prior = {}
if a.resume and out.exists():
    for l in open(out):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("business_decision") in VALID: prior[r["record_id"]] = r
fail_first = os.environ.get("FAKE_FAIL_FIRST") == "1" and not out.exists()
rows = []
for rec in recs:
    rid = rec["record_id"]
    if rid in prior: rows.append(prior[rid]); continue
    h = ihash(rec); d = VALID[int(h[:8], 16) % 3]
    if fail_first and int(hashlib.md5(rid.encode()).hexdigest()[:4], 16) % 10 == 0: d = "PARSE_ERROR"
    rows.append({"record_id": rid, "input_hash": h, "business_decision": d, "agent_confidence": 0.8, "decision_reasoning": "ok",
                 "key_factors": [], "cost_usd": 0.0025, "model": "fake", "input_tokens": 1500, "output_tokens": 240,
                 "timestamp": "2026-10-15T12:00:00"})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
if os.environ.get("FAKE_MARKER"): open(os.environ["FAKE_MARKER"], "a").write(f"{out.name} {len(recs) - len(prior)}\n")
''')
def expected_decision(rec): return CC.VALID_DECISIONS[int(CC.input_hash(rec)[:8], 16) % 3]

# ----------------------------------------------------------------------------------------------------------------
print("P1  the plan: every task once, runs spread, nothing altered")
PLAN_DIR = EO / "collection"
plan = PC.write_plan(EO, PLAN_DIR, B, R, 11)
plan, by_block = CC.load_plan(PLAN_DIR)
tasks = [t for b in sorted(by_block) for t in by_block[b]]
assert plan["n_tasks"] == len(tasks) == N * R + N_COND * N; ok(f"{len(tasks)} tasks = {N} customers x {R} baseline runs + {N_COND} conditions (55 dithered + 2 controls) x {N} records")
assert len({t["task_id"] for t in tasks}) == len(tasks) and [t["task_id"] for t in tasks] == sorted(t["task_id"] for t in tasks); ok("task ids are unique and increase in execution order (block, then position)")
base_tasks = collections.defaultdict(list)
for t in tasks:
    if t["stratum"].startswith("baseline_run_"): base_tasks[t["source_line"]].append(t)
assert sorted(base_tasks) == list(range(N)) and all(sorted(int(x["stratum"].rsplit("_", 1)[1]) for x in v) == list(range(1, R + 1)) for v in base_tasks.values()); ok("every customer has exactly one task for each baseline run 1..5")
stride = B // R
def circ_gaps(blocks):
    s = sorted(blocks); return [(s[(i + 1) % len(s)] - s[i]) % B or B for i in range(len(s))]
assert all(len({x["block"] for x in v}) == R and min(circ_gaps([x["block"] for x in v])) >= stride for v in base_tasks.values()); ok(f"each customer's {R} baseline runs fall in {R} DIFFERENT blocks, never closer than {stride} blocks apart (spread across the whole collection)")
for n in COND_NAMES:
    ls = sorted(t["source_line"] for t in tasks if t["stratum"] == f"cond:{n}"); assert ls == list(range(len(SRC[n]))), n
ok(f"each of the {N_COND} conditions has every record exactly once")
sizes = plan["block_sizes"]; assert max(sizes) - min(sizes) <= R + 1 and sum(sizes) == len(tasks); ok(f"blocks are the same size to within a few tasks ({min(sizes)} to {max(sizes)})")
for b in (1, B // 2, B):
    lines = CC.read_jsonl(CC.block_paths(PLAN_DIR, b)["input"]); assert len(lines) == len(by_block[b])
    for t, rec in zip(sorted(by_block[b], key=lambda t: t["pos"]), lines):
        src = SRC[t["stratum"][5:]][t["source_line"]] if t["stratum"].startswith("cond:") else BASE[t["source_line"]]
        assert rec["record_id"] == t["task_id"] and list(rec) == list(src) and {k: v for k, v in rec.items() if k != "record_id"} == {k: v for k, v in src.items() if k != "record_id"}
ok("block files carry each source record unchanged (same fields, same order) with only record_id replaced by the unique task id")
assert all(t["input_hash"] == A.BusinessDecisionAgent._input_hash(SRC[t["stratum"][5:]][t["source_line"]] if t["stratum"].startswith("cond:") else BASE[t["source_line"]]) for t in tasks); ok("the planner's input hash equals the REAL agent's _input_hash for every task")
assert plan["tasks_sha256"] == CC.sha256_file(PLAN_DIR / "tasks.jsonl") and all(plan["block_input_sha256"][b - 1] == CC.sha256_file(CC.block_paths(PLAN_DIR, b)["input"]) for b in range(1, B + 1)); ok("plan.json records the hash of the task list and of every block file")
ctl = [t for t in tasks if t["stratum"].startswith("cond:h0_clean_rerun_1")]; bsrc = {r["record_id"].replace("BASE_", "COND_", 1): CC.input_hash(r) for r in BASE}
assert all(t["input_hash"] == bsrc[t["source_record_id"]] for t in ctl); ok("the clean re-run control's tasks have the SAME input hash as the baseline's (zero dithering, identical input)")

seq = [t["stratum"] for t in sorted(by_block[1], key=lambda t: t["pos"])]; switches = sum(a != b for a, b in zip(seq, seq[1:])); assert switches > 0.9 * (len(seq) - 1); ok(f"inside a block the tasks are shuffled together: the stratum changes {switches} times in {len(seq) - 1} steps (conditions are interleaved, not run one after another)")
for k in (1, 2):
    cj = json.load(open(EO / "conditions" / f"h0_clean_rerun_{k}" / "control.json")); ref = json.load(open(EO / "conditions" / f"h0_clean_rerun_{k}" / "dither_reference.json"))
    assert cj["control"] is True and cj["kind"] == "clean_rerun" and len(ref) == N and all(r["_dither_fields"] == [] and r["_dither_applied"] is False and r["customer_id"].startswith("CUST_") for r in ref)
ok("each clean re-run control is marked control.json (so the hypothesis evaluators skip it) and its dither_reference holds NO dithered field")
assert not any((EO / "conditions" / n / "control.json").exists() for n in COND_NAMES if not n.startswith("h0_")); ok("no ordinary dithered condition is marked as a control")

print("P2  randomization, determinism, refusals")
for n in COND_NAMES:
    c = collections.Counter(t["block"] for t in tasks if t["stratum"] == f"cond:{n}"); assert len(c) == B and max(c.values()) - min(c.values()) <= 1, (n, c)
ok("every condition (and both controls) is spread over ALL blocks to within one task: no condition lives in its own window")
bb = collections.Counter(t["block"] for t in tasks if t["stratum"].startswith("baseline_run_")); assert max(bb.values()) - min(bb.values()) <= R; ok("the baseline runs are spread evenly over the blocks too")
consec = tot = 0
for n in COND_NAMES:
    ln = [t["block"] for t in sorted((t for t in tasks if t["stratum"] == f"cond:{n}"), key=lambda t: t["source_line"])]
    consec += sum((b2 - b1) % B == 1 for b1, b2 in zip(ln, ln[1:])); tot += len(ln) - 1
assert consec / tot < 0.2, consec / tot; ok(f"block assignment follows no pattern in record order: consecutive records land in consecutive blocks only {consec / tot:.0%} of the time (about 1 in {B} by chance; a cyclic dealing would give 100%)")
PLAN2 = WORK / "plan2"; PC.write_plan(EO, PLAN2, B, R, 11)
assert all((PLAN_DIR / f).read_bytes() == (PLAN2 / f).read_bytes() for f in ["plan.json", "tasks.jsonl"]) and all(CC.block_paths(PLAN_DIR, b)["input"].read_bytes() == CC.block_paths(PLAN2, b)["input"].read_bytes() for b in range(1, B + 1)); ok("the same seed gives a byte-identical plan (deterministic, no clock or machine in it)")
PLAN3 = WORK / "plan3"; PC.write_plan(EO, PLAN3, B, R, 12); _, by3 = CC.load_plan(PLAN3)
key = lambda bb_: {(t["stratum"], t["source_line"]): t["block"] for b in bb_ for t in bb_[b]}; k1, k3 = key(by_block), key(by3); assert sum(k1[k] != k3[k] for k in k1) > 0.5 * len(k1); ok("a different seed gives a different assignment")
try: PC.write_plan(EO, PLAN_DIR, B, R, 11); raise SystemExit("should refuse")
except SystemExit as e: assert "already exists" in str(e)
ok("a plan is written once: an existing plan.json is never overwritten")
try: PC.write_plan(EO, WORK / "p4", 4, 5, 1); raise SystemExit("should refuse")
except ValueError as e: assert "blocks >= baseline_runs" in str(e)
ok("fewer blocks than baseline runs is refused (the runs could not land in different windows)")
shutil.copytree(EO / "baseline", WORK / "eo2" / "baseline"); shutil.copytree(EO / "conditions", WORK / "eo2" / "conditions"); (WORK / "eo2/baseline/agent_input/baseline_customers.jsonl").unlink()
try: PC.write_plan(WORK / "eo2", WORK / "p5", B, R, 1); raise SystemExit("should refuse")
except SystemExit as e: assert "not found" in str(e)
ok("a missing baseline input is refused with a clear message")
(WORK / "halfplan").mkdir(); (WORK / "halfplan" / "tasks.jsonl").write_text("")
try: CC.load_plan(WORK / "halfplan"); raise SystemExit("should refuse")
except SystemExit as e: assert "no plan.json" in str(e)
ok("a half-written plan directory (no plan.json) is refused by every tool")
cp = sh("plan_collection.py", "--experiments_output", EO, "--out_dir", WORK / "plan_cli", "--blocks", B, "--seed", 3); assert cp.returncode == 0 and f"tasks in {B} blocks" in cp.stdout and "estimate" in cp.stdout; ok("the command line writes a plan and prints the size, the spread and a cost and time estimate")

# ----------------------------------------------------------------------------------------------------------------
print("R1  the executor: dry run, confirmation, retries, idempotence, shards, tamper detection")
MARK = WORK / "marker.txt"; ENV = {"FAKE_MARKER": str(MARK), "FAKE_FAIL_FIRST": "1"}
# The fake agent stands in for the model through the standard path; batch mode is tested in test_batch_runner.py.
run = lambda *extra, env=ENV: sh("run_collection.py", "--plan_dir", PLAN_DIR, "--agent", FAKE, "--mode", "standard", "--no_caffeinate", *extra, env=env)
r = run("--dry_run"); assert r.returncode == 0 and "nothing was sent" in r.stdout and not MARK.exists(); ok("--dry_run reports blocks, tasks and cost and sends nothing")
r = run(); assert r.returncode == 2 and "--yes" in r.stdout and not MARK.exists(); ok("without --yes it refuses to make API calls")
r = run("--yes", "--blocks", "1-2"); assert r.returncode == 0, r.stdout[-500:]
ev = [json.load(open(CC.block_paths(PLAN_DIR, b)["window"])) for b in (1, 2)]
assert all(CC.block_status(PLAN_DIR, b, by_block[b])["complete"] for b in (1, 2)) and all(e["attempts"] == 2 for e in ev) and all("started_at" in e and "finished_at" in e for e in ev); ok("blocks 1-2 complete after a RETRY (the first attempt left PARSE_ERRORs; the resumed attempt fixed only those), and each records its start and finish")
log = [l.split() for l in MARK.read_text().splitlines()]; first, second = int(log[0][1]), int(log[1][1])
assert first == len(by_block[1]) and 0 < second < 0.3 * first; ok(f"the retry re-sent ONLY the unfinished tasks ({second} of {first}), not the whole block: --resume keeps what is already valid")
calls = MARK.read_text().splitlines(); r = run("--yes", "--blocks", "1-2"); assert "nothing to do" in r.stdout and MARK.read_text().splitlines() == calls; ok("running the same command again does nothing: complete blocks are not re-sent")
complete = lambda: {b for b in range(1, B + 1) if CC.block_status(PLAN_DIR, b, by_block[b])["complete"]}
prev = complete()
for i in (1, 2, 3):
    r = run("--yes", "--shard", f"{i}/3"); assert r.returncode == 0, r.stdout[-400:]
    now = complete(); expected = {b for b in range(1, B + 1) if (b - 1) % 3 == i - 1} - prev
    assert now - prev == expected and expected, (i, now - prev, expected); prev = now
assert prev == set(range(1, B + 1)); ok("three shards (1/3, 2/3, 3/3) together complete every block, and each shard completed EXACTLY the blocks it owns ((b-1) mod 3 + 1)")
b12 = CC.block_paths(PLAN_DIR, B); saved = b12["decisions"].read_text(); b12["decisions"].unlink(); orig = b12["input"].read_text()
rec = json.loads(orig.splitlines()[0]); rec["total_spend"] = 123456.0; b12["input"].write_text(json.dumps(rec) + "\n" + "\n".join(orig.splitlines()[1:]) + "\n")
r = run("--yes", "--blocks", str(B)); assert r.returncode == 1 and "altered" in r.stdout and not b12["decisions"].exists(); ok("an edited block file is refused (its hash no longer matches the plan) and nothing is run on it")
b12["input"].write_text(orig); r = run("--yes", "--blocks", str(B)); assert r.returncode == 0 and CC.block_status(PLAN_DIR, B, by_block[B])["complete"]; ok("restoring the file lets the block run again")

# ----------------------------------------------------------------------------------------------------------------
print("C1  the collector: routing, order, refusals")
r = sh("collect_results.py", "--plan_dir", PLAN_DIR, "--status"); assert r.returncode == 0 and "100.0%" in r.stdout and not (EO / "baseline" / "decisions").exists(); ok("--status reports progress and writes nothing")
r = sh("collect_results.py", "--plan_dir", PLAN_DIR); assert r.returncode == 0, r.stdout[-400:]
assert all((EO / "baseline" / "decisions" / f"run{k}.decisions.jsonl").exists() for k in range(1, R + 1)) and all((EO / "conditions" / n / "decisions.jsonl").exists() for n in COND_NAMES); ok(f"{R} baseline run files and {N_COND} condition files were written, where the evaluators already look (controls included)")
bad = 0
for n in COND_NAMES:
    rows = CC.read_jsonl(EO / "conditions" / n / "decisions.jsonl"); src = SRC[n]
    assert [x["record_id"] for x in rows] == [s["record_id"] for s in src]
    bad += sum(x["business_decision"] != expected_decision(s) or x["input_hash"] != CC.input_hash(s) for x, s in zip(rows, src))
assert bad == 0; ok("every condition's rows are in source order, carry their ORIGINAL record_id back, and hold the right decision on the right input")
rows = CC.read_jsonl(EO / "conditions" / COND_NAMES[0] / "decisions.jsonl"); assert all(isinstance(x["collection_block"], int) and x["collection_task_id"].startswith("T") for x in rows); ok("each row also records its block and task id, so any result can be analysed by window")
perc = collections.defaultdict(set)
for k in range(1, R + 1):
    rr = CC.read_jsonl(EO / "baseline" / "decisions" / f"run{k}.decisions.jsonl"); assert [x["record_id"] for x in rr] == [b["record_id"] for b in BASE]
    for x in rr: perc[x["record_id"]].add(x["collection_block"])
assert all(len(v) == R for v in perc.values()); ok("each baseline run file has one row per customer in baseline order, and each customer's five runs came from five different blocks")
assert len({x["collection_block"] for x in CC.read_jsonl(EO / "conditions" / COND_NAMES[3] / "decisions.jsonl")}) >= B - 1; ok("a condition's rows come from (nearly) every block: its results span the whole collection")
r = sh("collect_results.py", "--plan_dir", PLAN_DIR); assert r.returncode != 0 and "already exist" in (r.stdout + r.stderr); ok("a second collection refuses to overwrite existing decisions files")
r = sh("collect_results.py", "--plan_dir", PLAN_DIR, "--overwrite"); assert r.returncode == 0; ok("--overwrite replaces them when asked")
EO3 = WORK / "eo3"; shutil.copytree(EO, EO3, ignore=shutil.ignore_patterns("decisions", "decisions.jsonl", "collection_report.json"))
P3 = EO3 / "collection"
cut = [t for t in tasks if t["stratum"] == f"cond:{COND_NAMES[5]}"][:3]
for t in cut:
    pth = CC.block_paths(P3, t["block"])["decisions"]; keep = [l for l in pth.read_text().splitlines() if json.loads(l)["record_id"] != t["task_id"]]; pth.write_text("".join(l + "\n" for l in keep))
r = sh("collect_results.py", "--plan_dir", P3); assert r.returncode == 1 and "not writing" in r.stdout and not (EO3 / "baseline" / "decisions").exists(); ok("with 3 tasks missing from ONE condition, the collector writes nothing at all (no half-finished results)")
r = sh("collect_results.py", "--plan_dir", P3, "--allow_incomplete"); assert r.returncode == 0 and not (EO3 / "conditions" / COND_NAMES[5] / "decisions.jsonl").exists() and (EO3 / "conditions" / COND_NAMES[4] / "decisions.jsonl").exists(); ok("--allow_incomplete writes the complete strata and withholds only the damaged one")
pth = CC.block_paths(P3, cut[0]["block"] if False else 3)["decisions"]; ls = pth.read_text().splitlines(); r0 = json.loads(ls[0]); r1 = json.loads(ls[1]); r0["input_hash"] = "0" * 32; r1["business_decision"] = "PARSE_ERROR"
pth.write_text("".join(json.dumps(x) + "\n" for x in [r0, r1] + [json.loads(l) for l in ls[2:]]))
st = CC.block_status(P3, 3, by_block[3]); assert len(st["bad_hash"]) == 1 and len(st["invalid"]) == 1 and not st["complete"]; ok("a row on the wrong input and a PARSE_ERROR row are each detected and counted, never accepted")
pth9 = CC.block_paths(P3, 9)["decisions"]; ls9 = [json.loads(l) for l in pth9.read_text().splitlines()]; first_t = ls9[0]["record_id"]
bad_first = dict(ls9[0]); bad_first["business_decision"] = "PARSE_ERROR"; pth9.write_text("".join(json.dumps(x) + "\n" for x in [bad_first] + ls9))
st9 = CC.block_status(P3, 9, by_block[9]); assert first_t in st9["valid"] and st9["complete"]; ok("when a task has two rows (an early PARSE_ERROR, a later valid one), the LATER row counts")
sm = CR.collect(P3, EO3, status_only=True); assert sm["wrong_input"] == 1 and sm["invalid"] == 1 and sm["missing"] == 3; ok("the status counts missing, invalid and wrong-input tasks separately")

# ----------------------------------------------------------------------------------------------------------------
shutil.copytree(PLAN_DIR, WORK / "tamper"); tl = (WORK / "tamper" / "tasks.jsonl").read_text().splitlines(); (WORK / "tamper" / "tasks.jsonl").write_text("\n".join(tl[1:] + tl[:1]) + "\n")
try: CC.load_plan(WORK / "tamper"); raise SystemExit("should refuse")
except SystemExit as e: assert "altered after it was written" in str(e)
ok("a task list edited after the plan was written is detected and refused by every tool")

print("S1  SIMULATION: does randomizing turn a window effect into noise, and can the clean re-run then be subtracted?")
NC, NCON, BS, RS = 400, 10, 24, 5
rng = random.Random(2026)
p_w = [rng.uniform(0.02, 0.25) for _ in range(BS)]                    # each window flips its own share of customers
def decide(cid, w):
    base = int(hashlib.md5(f"b{cid}".encode()).hexdigest()[:6], 16) % 3
    flip = int(hashlib.md5(f"f{cid}-{w}".encode()).hexdigest()[:6], 16) / 0xFFFFFF < p_w[w]
    return (base + 1) % 3 if flip else base
cust = [{"record_id": f"c{i}", "cid": i} for i in range(NC)]
conds = {f"x{j}": [dict(c) for c in cust] for j in range(NCON)}; conds["clean"] = [dict(c) for c in cust]
def drifts(task_list):
    bl = collections.defaultdict(list); got = collections.defaultdict(dict)
    for t in task_list:
        d = decide(t["rec"]["cid"], t["block"])
        if t["stratum"].startswith("baseline_run_"): bl[t["rec"]["cid"]].append(d)
        else: got[t["stratum"]][t["rec"]["cid"]] = d
    maj = {c: collections.Counter(v).most_common(1)[0][0] for c, v in bl.items()}
    return {s: sum(d != maj[c] for c, d in m.items()) / NC for s, m in got.items()}
rand = drifts(PC.build_plan(cust, conds, BS, RS, 5))
naive_tasks = [{"stratum": f"baseline_run_{k}", "rec": c, "block": k - 1, "source_line": i} for k in range(1, RS + 1) for i, c in enumerate(cust)]
for j, (name, recs) in enumerate(sorted(conds.items())):
    naive_tasks += [{"stratum": f"cond:{name}", "rec": c, "block": (RS + j) % BS, "source_line": i} for i, c in enumerate(recs)]
naive = drifts(naive_tasks)
sd = lambda d: (lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5)([x for s, x in d.items()])
print(f"      drift of {NCON + 1} conditions with NO real effect: randomized sd {sd(rand):.3f} (range {min(rand.values()):.2f} to {max(rand.values()):.2f}) | naive one-window-each sd {sd(naive):.3f} (range {min(naive.values()):.2f} to {max(naive.values()):.2f})")
assert sd(rand) < 0.5 * sd(naive); ok("with identical, effect-free conditions, randomized drift varies far less between conditions than one-window-per-condition drift")
assert max(abs(v - rand["cond:clean"]) for v in rand.values()) < 0.06; ok("randomized: every condition's drift is within a few points of the clean re-run's, so SUBTRACTING the control recovers 'no effect'")
assert max(abs(v - naive["cond:clean"]) for v in naive.values()) > 0.08; ok("naive: the same subtraction is badly wrong for some conditions (they sit in windows unlike the control's)")

print(f"\n{n_checks} check groups passed")
