#!/usr/bin/env python3
"""
Tests for batch mode: batch_runner.py and run_collection.py --mode batch.

No API key and no cost: a fake Message Batches client stands in for Anthropic's servers (it returns results out of order,
fails some requests, and can be "interrupted" mid-wait). The one test that needs the real LangChain library checks that a
batch request is the same request the standard path sends; it is skipped, loudly, only if LangChain is not installed.
Run from 1b_dithering/:

    python3 test_batch_runner.py
"""
import atexit, contextlib, hashlib, io, json, os, random, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

import business_decision_agent as A
import batch_runner as BR
import collection_common as CC
import plan_collection as PC

WORK = Path(tempfile.mkdtemp(prefix="batch_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a, **k)

VALID = CC.VALID_DECISIONS
MODEL = "claude-haiku-4-5-20251001"


class FakeBatches:
    """Enough of client.messages.batches to exercise the runner: create, retrieve, results."""
    def __init__(self, polls_until_end=2, fail_once=(), expire_once=(), interrupt_on_retrieve=False, fail_always=()):
        self.batches, self.creates, self.polls_until_end = {}, 0, polls_until_end
        self.fail_always = set(fail_always)
        self.fail_once, self.expire_once = set(fail_once), set(expire_once)
        self.interrupt_on_retrieve = interrupt_on_retrieve
    def create(self, requests):
        self.creates += 1
        bid = f"msgbatch_fake{self.creates:03d}"
        self.batches[bid] = {"requests": requests, "polls": 0, "created_at": f"2026-10-22T10:{self.creates:02d}:00+00:00"}
        return {"id": bid, "created_at": self.batches[bid]["created_at"], "processing_status": "in_progress"}
    def retrieve(self, bid):
        if self.interrupt_on_retrieve:
            self.interrupt_on_retrieve = False
            raise ConnectionError("simulated: the laptop went to sleep")
        b = self.batches[bid]; b["polls"] += 1
        ended = b["polls"] > self.polls_until_end
        return {"id": bid, "processing_status": "ended" if ended else "in_progress", "created_at": b["created_at"],
                "ended_at": "2026-10-22T11:00:00+00:00" if ended else None,
                "request_counts": {"succeeded": 0, "processing": len(b["requests"])}}
    def results(self, bid):
        reqs = list(self.batches[bid]["requests"]); random.Random(bid).shuffle(reqs)     # any order, as documented
        for r in reqs:
            cid = r["custom_id"]
            if cid in self.fail_always:
                yield {"custom_id": cid, "result": {"type": "errored"}}; continue
            if cid in self.fail_once:
                self.fail_once.discard(cid); yield {"custom_id": cid, "result": {"type": "errored"}}; continue
            if cid in self.expire_once:
                self.expire_once.discard(cid); yield {"custom_id": cid, "result": {"type": "expired"}}; continue
            text = r["params"]["messages"][0]["content"]
            d = VALID[int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 3]
            body = json.dumps({"business_decision": d, "agent_confidence": 0.8, "decision_reasoning": "fake", "key_factors": ["nps_score"]})
            yield {"custom_id": cid, "result": {"type": "succeeded", "message": {
                "model": MODEL, "stop_reason": "end_turn", "content": [{"type": "text", "text": body}],
                "usage": {"input_tokens": 1500, "output_tokens": 200}}}}


class FakeClient:
    def __init__(self, **kw):
        self.messages = type("M", (), {})(); self.messages.batches = FakeBatches(**kw)


def expected(rec):
    return VALID[int(hashlib.md5(A.build_user_prompt(rec).encode()).hexdigest()[:8], 16) % 3]


# ----------------------------------------------------------------------------------------------------------------------
print("Q1  the batch request is the standard request")
recs = CC.read_jsonl(HERE / "experiments_output" / "baseline" / "agent_input" / "baseline_customers.jsonl")[:25]
if not recs:
    raise SystemExit("no baseline records found under experiments_output/baseline to test against")
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = ChatAnthropic(model=MODEL, temperature=0.0, max_tokens=A.DEFAULT_MAX_TOKENS, anthropic_api_key="dummy")
    have_lc = hasattr(llm, "_get_request_payload")
except Exception:
    have_lc = False
if have_lc:
    for r in recs:
        lc = llm._get_request_payload([SystemMessage(content=A.SYSTEM_PROMPT), HumanMessage(content=A.build_user_prompt(r))])
        lc = {**{k: v for k, v in lc.items() if k != "extra_body"}, **lc.get("extra_body", {})}   # extra_body is merged into the HTTP body
        assert A.messages_api_params(r, MODEL, 0.0) == lc, (lc.keys(), A.messages_api_params(r, MODEL, 0.0).keys())
    ok(f"for {len(recs)} real records the batch request body equals the body LangChain's ChatAnthropic sends "
       f"(model, max_tokens, temperature, system, messages; nothing more, nothing less)")
else:
    print("  ⚠️  SKIPPED: langchain_anthropic is not installed here, so request parity was NOT checked. Run this on the machine that runs the agent.")
assert all(r["record_id"] not in A.build_user_prompt(r) for r in recs); ok("the record_id never reaches the model, in either mode")

print("Q2  custom ids")
reqs = BR.build_requests([{**r, "record_id": f"T{i:06d}"} for i, r in enumerate(recs)], MODEL, 0.0)
assert [q["custom_id"] for q in reqs] == [f"T{i:06d}" for i in range(len(recs))]; ok("task ids (T000123) are used as custom_ids, one request per record")
for bad in ("BASE_7015D7FD2A5F!", "x" * 65, ""):
    try: BR.build_requests([{**recs[0], "record_id": bad}], MODEL, 0.0); raise SystemExit("should refuse")
    except ValueError: pass
ok("an id the API would reject (bad characters, over 64 characters, empty) is refused before anything is sent")
try: BR.build_requests([{**recs[0], "record_id": "T1"}, {**recs[1], "record_id": "T1"}], MODEL, 0.0); raise SystemExit("should refuse")
except ValueError as e: assert "duplicate" in str(e)
ok("duplicate ids are refused (results are matched by custom_id)")

print("Q3  one batch: out-of-order results, failures, rows, cost")
block = [{**r, "record_id": f"T{i:06d}"} for i, r in enumerate(recs)]
d = WORK / "q3"; d.mkdir()
paths = {"decisions": d / "b.decisions.jsonl", "state": d / "b.batches.json", "meta": d / "b.decisions.meta.jsonl", "input": d / "b.input.jsonl"}
paths["input"].write_text("".join(json.dumps(r) + "\n" for r in block))
fc = FakeClient(fail_once={"T000003"}, expire_once={"T000007"})
res = quiet(BR.run_batch, block, decisions_path=paths["decisions"], state_path=paths["state"], meta_path=paths["meta"],
            input_path=paths["input"], model=MODEL, temperature=0.0, client=fc, poll_seconds=0)
rows = CC.read_jsonl(paths["decisions"])
assert fc.messages.batches.creates == 1 and res["counts"] == {"succeeded": len(block) - 2, "errored": 1, "canceled": 0, "expired": 1}; ok("one batch created; errored and expired requests are counted, not written")
assert sorted(r["record_id"] for r in rows) == sorted(r["record_id"] for r in block if r["record_id"] not in ("T000003", "T000007")); ok("results that came back in shuffled order are matched to the right records by custom_id")
byid = {r["record_id"]: r for r in block}
assert all(r["business_decision"] == expected(byid[r["record_id"]]) and r["input_hash"] == CC.input_hash(byid[r["record_id"]]) for r in rows); ok("each row's decision and input hash belong to its own record")
std = A.build_result_row(block[0], {"business_decision": "LOW_PRIORITY"}, model=MODEL, cost_usd=0.0, input_tokens=1, output_tokens=1, latency_ms=5, timestamp="t")
extras = ["api_mode", "batch_id", "batch_created_at", "batch_ended_at", "served_model", "stop_reason"]
assert all(list(r)[:len(std)] == list(std) and list(r)[len(std):] == extras for r in rows); ok("rows have exactly the standard schema, in the same order, plus six batch fields")
std_cost = 1500 / 1000 * 0.001 + 200 / 1000 * 0.005
assert all(abs(r["cost_usd"] - 0.5 * std_cost) < 1e-12 and r["processing_time_ms"] is None and r["served_model"] == MODEL and r["timestamp"] == r["batch_ended_at"] for r in rows)
ok("cost is half the standard price from model_pricing.json; latency is null (not invented); the served model and the batch end time are recorded")
st = json.load(open(paths["state"])); meta = CC.read_jsonl(paths["meta"])
assert st["batches"][0]["collected"] and "_records" not in st["batches"][0] and st["batches"][0]["requests_sha256"] == BR.requests_sha256(BR.build_requests(block, MODEL, 0.0))
assert [m["event"] for m in meta] == ["batch_submitted", "batch_collected"] and meta[0]["api_mode"] == "batch" and meta[0]["system_prompt_sha256"] == hashlib.sha256(A.SYSTEM_PROMPT.encode()).hexdigest() and meta[0]["model"] == MODEL and meta[0]["temperature"] == 0.0
ok("the state file records the batch and a hash of exactly what was sent; the fingerprint records mode, model, settings and the prompt hash")

print("Q4  interrupted: resume the same batch, never pay twice")
d = WORK / "q4"; d.mkdir()
p4 = {k: d / v.name for k, v in paths.items()}; p4["input"].write_text(paths["input"].read_text())
fc = FakeClient(interrupt_on_retrieve=True)
try: quiet(BR.run_batch, block, decisions_path=p4["decisions"], state_path=p4["state"], meta_path=p4["meta"], input_path=p4["input"], model=MODEL, temperature=0.0, client=fc, poll_seconds=0); raise SystemExit("should have been interrupted")
except ConnectionError: pass
st = json.load(open(p4["state"]))
assert len(st["batches"]) == 1 and not st["batches"][0]["collected"] and "submitting" not in st and not p4["decisions"].exists(); ok("after an interruption the batch id is on disk, marked not yet collected, and no rows were written")
res = quiet(BR.run_batch, block, decisions_path=p4["decisions"], state_path=p4["state"], meta_path=p4["meta"], input_path=p4["input"], model=MODEL, temperature=0.0, client=fc, poll_seconds=0)
assert fc.messages.batches.creates == 1 and res["resumed"] and len(CC.read_jsonl(p4["decisions"])) == len(block); ok("re-running waits for and collects the SAME batch: still one batch created, every record decided")
st = json.load(open(p4["state"])); st["submitting"] = "2026-10-22T09:00:00+00:00"; p4["state"].write_text(json.dumps(st))
try: quiet(BR.run_batch, block, decisions_path=p4["decisions"], state_path=p4["state"], meta_path=p4["meta"], input_path=p4["input"], model=MODEL, temperature=0.0, client=fc, poll_seconds=0); raise SystemExit("should refuse")
except RuntimeError as e: assert "Console" in str(e)
assert fc.messages.batches.creates == 1; ok("a submission that started but saved no id stops the runner with instructions; nothing is sent blind")

print("Q5  run_collection.py --mode batch, end to end on a small plan")
EO = WORK / "eo"
g = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "12", "--seed", "7", "--clean-reruns", "1", "--out", str(EO)], capture_output=True, text=True)
assert g.returncode == 0, g.stdout[-400:] + g.stderr[-400:]
PLAN = EO / "collection"; B = 6
quiet(PC.write_plan, EO, PLAN, B, 5, 11)
plan, by_block = CC.load_plan(PLAN)
all_tasks = [t["task_id"] for b in by_block for t in by_block[b]]
fail = set(random.Random(3).sample(all_tasks, 4))
fc = FakeClient(fail_once=fail)
import run_collection as RC
BR.default_client = lambda: fc
def rc(*args):
    old = sys.argv; sys.argv = ["run_collection.py", "--plan_dir", str(PLAN), "--no_caffeinate", "--poll_seconds", "0", "--allow_version_mismatch", *args]
    try: return quiet(RC.main)
    finally: sys.argv = old
assert rc("--dry_run") == 0 and fc.messages.batches.creates == 0; ok("a dry run sends nothing")
assert rc() == 2 and fc.messages.batches.creates == 0; ok("without --yes nothing is sent")
assert rc("--yes") == 0
statuses = [CC.block_status(PLAN, b, by_block[b]) for b in range(1, B + 1)]
assert all(s["complete"] for s in statuses); ok(f"every block completes; the {len(fail)} requests that failed were resent and decided")
assert fc.messages.batches.creates == B + len({t["block"] for t in (t for b in by_block for t in by_block[b]) if t["task_id"] in fail}); ok("one batch per block, plus one retry batch for each block that had a failure (only the failed requests are resent)")
created = [fc.messages.batches.batches[f"msgbatch_fake{i:03d}"] for i in range(1, fc.messages.batches.creates + 1)]
blocks_in_order = []
for c in created:
    ids = {q["custom_id"] for q in c["requests"]}
    blocks_in_order.append(next(b for b in by_block if ids <= {t["task_id"] for t in by_block[b]}))
assert blocks_in_order == sorted(blocks_in_order) and len(set(blocks_in_order)) == B; ok("blocks are submitted strictly in order, each in its own batch(es), never mixed: a block is still one window")
w = json.load(open(CC.block_paths(PLAN, 1)["window"]))
assert w["mode"] == "batch" and w["batches"] and all(x["batch_id"].startswith("msgbatch_") and x["ended_at"] for x in w["batches"]); ok("each block's window file records the mode and its batch ids with their start and end times")
assert all(r["business_decision"] == expected(rec) for b in range(1, B + 1) for rec in CC.read_jsonl(CC.block_paths(PLAN, b)["input"]) for r in [statuses[b - 1]["valid"][rec["record_id"]]]); ok("every task's accepted decision belongs to its own input")
n_before = fc.messages.batches.creates
assert rc("--yes") == 0 and fc.messages.batches.creates == n_before; ok("running again does nothing and sends nothing (idempotent)")
CR = subprocess.run([sys.executable, "collect_results.py", "--plan_dir", str(PLAN)], capture_output=True, text=True)
assert CR.returncode == 0, CR.stdout[-500:] + CR.stderr[-500:]; ok("collect_results.py accepts batch-mode output unchanged")

print("Q6  the request stack is pinned (temperature was removed in API version 2026-10-21)")
import httpx, anthropic
seen = {}
def _h(req):
    seen.update(dict(req.headers))
    return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": MODEL, "content": [{"type": "text", "text": "{}"}],
                                     "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
import importlib; importlib.reload(BR)
bc = BR.default_client()
bc = bc.with_options(http_client=httpx.Client(transport=httpx.MockTransport(_h)))
bc.messages.create(**A.messages_api_params(recs[0], MODEL, 0.0))
assert seen.get("anthropic-version") == A.ANTHROPIC_API_VERSION == "2023-06-01"; ok("the batch path's client sends the pinned anthropic-version header on the wire (checked with a mock transport)")
agent = A.BusinessDecisionAgent(model=MODEL, temperature=0.0)
assert agent.llm.default_headers.get("anthropic-version") == A.ANTHROPIC_API_VERSION and agent.llm._client.default_headers.get("anthropic-version") == A.ANTHROPIC_API_VERSION
ok("the standard path's LangChain client carries the same pinned header")
fp = BR.batch_fingerprint(paths["input"], paths["decisions"], 1, MODEL, 0.0, 1024)
assert fp["anthropic_api_version"] == "2023-06-01" and isinstance(fp["request_stack_mismatches"], list); ok("the run fingerprint records the API version and any library mismatch")
saved = dict(A.PINNED_PACKAGES); A.PINNED_PACKAGES["anthropic"] = "0.0.1"
old = sys.argv; sys.argv = ["run_collection.py", "--plan_dir", str(PLAN), "--no_caffeinate", "--poll_seconds", "0", "--yes", "--blocks", "1"]
try:
    n_before = fc.messages.batches.creates
    for p_ in CC.block_paths(PLAN, 1).values():
        if p_.name.endswith((".decisions.jsonl", ".batches.json", ".window.json")) and p_.exists(): p_.unlink()
    BR.default_client = lambda: fc
    rc_ = quiet(RC.main)
finally:
    sys.argv = old; A.PINNED_PACKAGES.clear(); A.PINNED_PACKAGES.update(saved)
assert rc_ == 2 and fc.messages.batches.creates == n_before; ok("a real run with request libraries other than the pinned ones is refused before anything is sent")

print("Q7  pre-mortem guards: budget pause (G1), failed-call ceiling (G4), era check (G1)")
def fresh_plan(name):
    eo = WORK / name
    g = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "12", "--seed", "7", "--clean-reruns", "1", "--out", str(eo)], capture_output=True, text=True)
    assert g.returncode == 0, g.stdout[-400:] + g.stderr[-400:]
    quiet(PC.write_plan, eo, eo / "collection", B, 5, 11)
    return eo / "collection", CC.load_plan(eo / "collection")[1]
def rc_in(plan_dir, *args):
    old = sys.argv; sys.argv = ["run_collection.py", "--plan_dir", str(plan_dir), "--no_caffeinate", "--poll_seconds", "0", "--allow_version_mismatch", "--yes", *args]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = RC.main()
        return code, buf.getvalue()
    finally: sys.argv = old
def collect_in(plan_dir, *args):
    return subprocess.run([sys.executable, "collect_results.py", "--plan_dir", str(plan_dir), *args], capture_output=True, text=True)

# G1 budget: allow block 1 only (the estimate before 100 priced rows is the --cost_per_call fallback at half price)
P7, bb7 = fresh_plan("q7_budget")
fc = FakeClient(); BR.default_client = lambda: fc
est1 = len(bb7[1]) * 0.00256 * 0.5
code, out = rc_in(P7, "--budget_usd", f"{est1 * 1.5:.6f}")
assert code == 3 and "PAUSED before block 2" in out; ok("the run pauses before the block that would pass the budget, with a distinct exit code (3) and a message saying what to do")
assert CC.block_status(P7, 1, bb7[1])["complete"] and not CC.block_paths(P7, 2)["decisions"].exists() and fc.messages.batches.creates == 1
ok("the finished block is kept and nothing was sent for the paused block")
sp = CC.spend_so_far(P7, B)
assert sp["rows"] == len(bb7[1]) and abs(sp["usd"] - len(bb7[1]) * 0.5 * std_cost) < 1e-9; ok("spend so far is the sum of recorded cost_usd across every decisions file")
code, out = rc_in(P7, "--budget_usd", "1000")
assert code == 0 and all(CC.block_status(P7, b, bb7[b])["complete"] for b in range(1, B + 1)) and fc.messages.batches.creates == B
ok("raising the budget and re-running continues from block 2; block 1 is never re-sent")
code, out = rc_in(P7, "--budget_usd", "0")
assert code == 0 and "nothing to do" in out; ok("a budget below the spend already recorded does not block a run that has nothing left to send")

# G4 failure ceiling: permanently failing requests, above and below 2%
P8, bb8 = fresh_plan("q7_fail")
n8 = sum(len(v) for v in bb8.values())
tasks8 = [t["task_id"] for b in sorted(bb8) for t in bb8[b]]
k_over = int(n8 * 0.02) + 1
fc = FakeClient(fail_always=set(random.Random(5).sample(tasks8, k_over))); BR.default_client = lambda: fc
code, out = rc_in(P8, "--retries", "1")
assert code == 4 and "STOP" in out and f"{k_over:,} of {n8:,}" in out; ok(f"{k_over} of {n8} tasks still failed after retries ({k_over / n8:.1%}) is over 2%: the run stops with exit code 4 and says to investigate")
r = collect_in(P8, "--allow_incomplete")
assert r.returncode == 1 and "not writing" in r.stdout and not (P8 / "collection_report.json").exists(), r.stdout[-300:]
ok("collect_results.py refuses to write while the ceiling is exceeded, even with --allow_incomplete")
r = collect_in(P8, "--allow_incomplete", "--override_failure_ceiling")
assert r.returncode == 0, r.stdout[-400:]; rep8 = json.load(open(P8 / "collection_report.json"))
assert rep8["failed_calls"]["failed"] == k_over and rep8["failed_calls"]["over_ceiling"]; ok("the explicit override writes, and the report still records the failed-call rate and that it was over the ceiling")
fs = CC.failure_summary(P8, bb8, list(range(1, B + 1)))
assert fs["tasks"] == n8 and abs(fs["rate"] - k_over / n8) < 1e-12; ok("the rate is failed tasks over attempted tasks")
# under the ceiling: drop enough failures (by marking them answered) to fall to 2% or below
P9, bb9 = fresh_plan("q7_under")
tasks9 = [t["task_id"] for b in sorted(bb9) for t in bb9[b]]
k_under = max(0, int(len(tasks9) * 0.02))
fc = FakeClient(fail_always=set(random.Random(6).sample(tasks9, k_under))); BR.default_client = lambda: fc
code, out = rc_in(P9, "--retries", "0")
assert code == (1 if k_under else 0) and "STOP" not in out; ok(f"{k_under} of {len(tasks9)} failed ({k_under / len(tasks9):.1%}, at or under 2%) is reported but does not stop the run")
fs = CC.failure_summary(P9, bb9, [1])
assert fs["attempted_blocks"] == 1 and fs["tasks"] == len(bb9[1]); ok("only blocks that have been attempted count toward the rate")
# G4 definition: an unparseable or off-label answer is a failure; odd but well-formed reasoning is not
st = CC.block_status(P7, 1, bb7[1]); tid = next(iter(st["valid"]))
dpath = CC.block_paths(P7, 1)["decisions"]; rows1 = CC.read_jsonl(dpath)
odd = [dict(x, decision_reasoning="banana " * 3) if x["record_id"] == tid else x for x in rows1]
dpath.write_text("".join(json.dumps(x) + "\n" for x in odd))
assert CC.failure_summary(P7, bb7, [1])["failed"] == 0; ok("odd but well-formed reasoning is data, not a failed call")
bad = [dict(x, business_decision="MAYBE") if x["record_id"] == tid else x for x in rows1]
dpath.write_text("".join(json.dumps(x) + "\n" for x in bad))
assert CC.failure_summary(P7, bb7, [1])["failed"] == 1; ok("a decision other than HIGH, MEDIUM or LOW priority is a failed call")
dpath.write_text("".join(json.dumps(x) + "\n" for x in rows1))

# G1 era check: rewrite the recorded window times
def set_windows(plan_dir, starts_h, dur_h=1.0):
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)
    for b, h in enumerate(starts_h, start=1):
        wp = CC.block_paths(plan_dir, b)["window"]; w = json.load(open(wp))
        w["started_at"] = (t0 + timedelta(hours=h)).isoformat(); w["finished_at"] = (t0 + timedelta(hours=h + dur_h)).isoformat()
        wp.write_text(json.dumps(w))
set_windows(P7, [0, 1, 2, 9, 10, 17])          # gaps after blocks 3 and 5: 6 h exactly? no: 9-3=6 h (not over), 17-11=6 h
e = CC.compute_eras(P7, B)
assert e["n_eras"] == 1 and not e["gaps_over_threshold"]; ok("a gap of exactly 6 hours does not start a new era (the rule is more than 6)")
set_windows(P7, [0, 1, 2, 9.5, 10.5, 30])
e = CC.compute_eras(P7, B)
assert e["n_eras"] == 3 and [g["after_block"] for g in e["gaps_over_threshold"]] == [3, 5] and e["block_era"] == {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
ok("gaps of 6.5 and 18.5 hours split the run into three eras, and every block is assigned to one")
set_windows(P7, [20, 0, 1, 2, 3, 4])            # block 1 re-run last: eras follow time, not block number
e = CC.compute_eras(P7, B)
assert e["block_era"][1] == 2 and e["n_eras"] == 2 and e["gaps_over_threshold"][0]["before_block"] == 1; ok("eras follow the recorded times, not block numbers (a block re-run later lands in the later era)")
set_windows(P7, [0, 1, 2, 9.5, 10.5, 30])
r = collect_in(P7)
assert r.returncode == 0 and "eras: 3" in r.stdout, r.stdout[-400:]
rep7 = json.load(open(P7 / "collection_report.json")); eras7 = json.load(open(P7 / "collection_eras.json"))
assert rep7["eras"]["n_eras"] == 3 and eras7["n_eras"] == 3; ok("collect_results.py writes collection_eras.json and puts the eras in collection_report.json")
plan7 = CC.load_plan(P7)[0]
era_of = {t["task_id"]: t["block"] for b in bb7 for t in bb7[b]}
rows_out = [x for f in sorted((P7.parent).rglob("*.jsonl")) if "collection" not in f.parts for x in CC.read_jsonl(f, tolerant=True) if "collection_era" in x]
assert rows_out and all(x["collection_era"] == eras7["block_era"][str(x["collection_block"])] for x in rows_out)
ok(f"every collected row ({len(rows_out)}) carries collection_era, matching its block's era")

print(f"\n{n_checks} check groups passed")
