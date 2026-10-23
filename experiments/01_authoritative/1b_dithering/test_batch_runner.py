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
    def __init__(self, polls_until_end=2, fail_once=(), expire_once=(), interrupt_on_retrieve=False):
        self.batches, self.creates, self.polls_until_end = {}, 0, polls_until_end
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
    old = sys.argv; sys.argv = ["run_collection.py", "--plan_dir", str(PLAN), "--no_caffeinate", "--poll_seconds", "0", *args]
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

print(f"\n{n_checks} check groups passed")
