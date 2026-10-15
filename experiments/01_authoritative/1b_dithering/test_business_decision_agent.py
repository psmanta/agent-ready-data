#!/usr/bin/env python3
"""
Regression test for business_decision_agent.py: the frozen system prompt, --max_records, and --resume.

The REAL base agent's code runs (JSON parsing, record handling); only the model call is faked, so this
costs nothing and needs no API key. It builds its own small dataset in a temporary directory and never
touches your real experiments_output. Run from 1b_dithering/:

    python3 test_business_decision_agent.py

If you deliberately change the system prompt, the T0 check fails on purpose: update PROMPT_SHA256 in a
disclosed, dated amendment, never silently.
"""
import atexit, contextlib, hashlib, io, json, os, shutil, subprocess, sys, tempfile, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

# The agent file lives in the repo at <repo>/experiments/01_authoritative/1b_dithering and imports
# agents.base_agent from <repo>/shared/agents. Only the LangChain packages (not installed everywhere)
# are stubbed; the real base_agent.py and llm_factory.py are used.
for _n in ("langchain_anthropic", "langchain_core", "langchain_core.messages"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["langchain_anthropic"].ChatAnthropic = type("ChatAnthropic", (), {"__init__": lambda self, *a, **k: None})
sys.modules["langchain_core.messages"].HumanMessage = sys.modules["langchain_core.messages"].SystemMessage = type("M", (), {"__init__": lambda self, content="": None})
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
import business_decision_agent as A

PROMPT_SHA256 = "94618d826d5e8114115cbb16ae9939f10749318064674f365019d93d5e5b8fdd"
WORK = Path(tempfile.mkdtemp(prefix="agent_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
_g = subprocess.run([sys.executable, "generate_dithered_data.py", "--n", "30", "--seed", "7", "--out", str(WORK / "out")], capture_output=True, text=True)
assert _g.returncode == 0, _g.stdout[-400:] + _g.stderr[-400:]
recs = [json.loads(l) for l in open(WORK / "out/baseline/agent_input/baseline_customers.jsonl")][:12]
IN = WORK / "in.jsonl"; IN.write_text("".join(json.dumps(r) + "\n" for r in recs))
display = lambda r: json.dumps({k: v for k, v in r.items() if k != "record_id"}, indent=2, default=str)
LOOKUP = {display(r): r["record_id"] for r in recs}
n_checks = 0
def ok(label):
    global n_checks; n_checks += 1; print(f"  ✅ {label}")

class Fake:
    calls = []; fail_on_call = None; garbage_once = set(); seen = set()
    @classmethod
    def reset(cls): cls.calls, cls.fail_on_call, cls.garbage_once, cls.seen = [], None, set(), set()

def fake_invoke(self, system_prompt, user_prompt):
    rid = next(v for k, v in LOOKUP.items() if k in user_prompt)
    Fake.calls.append(rid)
    if Fake.fail_on_call is not None and len(Fake.calls) == Fake.fail_on_call:
        raise Exception("LLM invocation failed after 3 attempts: simulated outage")
    if rid in Fake.garbage_once and rid not in Fake.seen:
        Fake.seen.add(rid); content = "I cannot produce JSON for this"
    else:
        h = int(hashlib.md5(rid.encode()).hexdigest(), 16)
        content = json.dumps({"business_decision": ["HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY"][h % 3], "agent_confidence": round(0.5 + (h % 50) / 100, 2),
                              "decision_reasoning": f"reasoning for {rid}", "key_factors": ["churn_risk_score"]})
    return {"content": content, "latency_ms": 12.0, "cost": 0.001, "input_tokens": 100, "output_tokens": 20, "tokens": 120}
A.BusinessDecisionAgent._invoke_llm = fake_invoke

def agent(): return A.BusinessDecisionAgent(api_key="dummy", log_dir=WORK / "logs")
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def strip(rows): return [{k: v for k, v in r.items() if k not in ("timestamp", "processing_time_ms")} for r in rows]
def silently(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a, **k)

print("T0  the system prompt is frozen")
assert hashlib.sha256(A.SYSTEM_PROMPT.encode()).hexdigest() == PROMPT_SHA256, "the system prompt changed"; ok("SYSTEM_PROMPT matches its frozen sha256")
assert "temperature: float = 0.0" in open(HERE / "business_decision_agent.py").read() and "claude-haiku-4-5-20251001" in open(HERE / "business_decision_agent.py").read(); ok("temperature 0.0 and claude-haiku-4-5-20251001 are the defaults")

print("T1  a plain run: schema, hashes, summary")
Fake.reset(); s = silently(agent().process_file, IN, WORK / "plain.jsonl"); rows = load(WORK / "plain.jsonl")
assert [r["record_id"] for r in rows] == [r["record_id"] for r in recs] and len(Fake.calls) == 12; ok("12 records in, 12 decisions out, in input order, one model call each")
assert all(r["input_hash"] == hashlib.md5(json.dumps({k: v for k, v in rec.items() if k != "record_id"}, sort_keys=True, default=str).encode()).hexdigest() for r, rec in zip(rows, recs)); ok("input_hash is the md5 of the record exactly as the agent sees it (record_id excluded)")
assert all({"record_id", "customer_segment", "input_hash", "business_decision", "agent_confidence", "decision_reasoning", "key_factors", "model", "timestamp"} <= set(r) for r in rows); ok("every output row carries the keys the evaluators read")
assert s["total_records"] == 12 and s["errors"] == 0 and s["resumed_records"] == 0 and s["processed_this_session"] == 12 and abs(s["total_cost_usd"] - 0.012) < 1e-9; ok("summary: 12 records, 0 errors, cost $0.012")

print("T2  --max_records")
Fake.reset(); s = silently(agent().process_file, IN, WORK / "m3.jsonl", max_records=3)
assert [r["record_id"] for r in load(WORK / "m3.jsonl")] == [r["record_id"] for r in recs[:3]] and len(Fake.calls) == 3 and s["total_records"] == 3; ok("exactly the first 3 records are processed, with 3 model calls")

print("T3  a crash mid-run, then --resume")
Fake.reset(); Fake.fail_on_call = 6
try: silently(agent().process_file, IN, WORK / "crash.jsonl"); raise SystemExit("expected a crash")
except Exception as e: assert "simulated outage" in str(e)
assert len(load(WORK / "crash.jsonl")) == 5; ok("a crash leaves the 5 completed decisions on disk")
Fake.reset(); s = silently(agent().process_file, IN, WORK / "crash.jsonl", resume=True)
assert Fake.calls == [r["record_id"] for r in recs[5:]]; ok("resume sends only the 7 missing records to the model")
assert strip(load(WORK / "crash.jsonl")) == strip(rows); ok("the resumed file equals the uninterrupted run, record for record")
assert s["resumed_records"] == 5 and s["processed_this_session"] == 7 and s["total_records"] == 12 and abs(s["total_cost_usd"] - 0.012) < 1e-9; ok("summary: 5 kept + 7 new; total cost covers the whole file")

print("T4  PARSE_ERROR lines and truncated lines")
Fake.reset(); Fake.garbage_once = {recs[3]["record_id"]}; silently(agent().process_file, IN, WORK / "pe.jsonl")
assert sum(r["business_decision"] == "PARSE_ERROR" for r in load(WORK / "pe.jsonl")) == 1
Fake.reset(); silently(agent().process_file, IN, WORK / "pe.jsonl", resume=True); got = load(WORK / "pe.jsonl"); ids = [r["record_id"] for r in got]
assert Fake.calls == [recs[3]["record_id"]] and len(ids) == 12 == len(set(ids)) and not any(r["business_decision"] == "PARSE_ERROR" for r in got); ok("only the failed record is re-sent; 12 unique ids; no PARSE_ERROR left")
txt = (WORK / "plain.jsonl").read_text().splitlines(keepends=True); (WORK / "trunc.jsonl").write_text("".join(txt[:7]) + txt[7][:40])
Fake.reset(); silently(agent().process_file, IN, WORK / "trunc.jsonl", resume=True)
assert Fake.calls == [r["record_id"] for r in recs[7:]] and strip(load(WORK / "trunc.jsonl")) == strip(rows); ok("a line cut off mid-write is ignored and its record re-sent")

print("T5  resume refusals leave the file untouched")
before = (WORK / "plain.jsonl").read_bytes(); Fake.reset(); silently(agent().process_file, IN, WORK / "plain.jsonl", resume=True)
assert Fake.calls == [] and (WORK / "plain.jsonl").read_bytes() == before; ok("resuming a finished run makes zero calls and changes nothing (byte-identical)")
changed = [dict(r) for r in recs]; changed[1]["total_spend"] = (changed[1].get("total_spend") or 0) + 1.0; IN2 = WORK / "in2.jsonl"; IN2.write_text("".join(json.dumps(r) + "\n" for r in changed))
try: silently(agent().process_file, IN2, WORK / "plain.jsonl", resume=True); raise SystemExit("should refuse")
except ValueError as e: assert "different input" in str(e) and recs[1]["record_id"] in str(e)
assert (WORK / "plain.jsonl").read_bytes() == before; ok("decisions made on DIFFERENT input are refused (the record is named) and the file is untouched")
try: silently(agent().process_file, IN, WORK / "plain.jsonl", resume=True, max_records=5); raise SystemExit("should refuse")
except ValueError as e: assert "outside the current input window" in str(e)
assert (WORK / "plain.jsonl").read_bytes() == before; ok("a smaller --max_records window cannot silently drop completed work")
IN3 = WORK / "dup.jsonl"; IN3.write_text("".join(json.dumps(r) + "\n" for r in recs + [recs[0]]))
try: silently(agent().process_file, IN3, WORK / "plain.jsonl", resume=True); raise SystemExit("should refuse")
except ValueError as e: assert "unique record_ids" in str(e)
ok("duplicate record_ids are refused under resume")

print("T6  overwriting, and the command line")
buf = io.StringIO(); Fake.reset()
with contextlib.redirect_stdout(buf): agent().process_file(IN, WORK / "plain.jsonl", max_records=2)
assert "WARNING: overwriting" in buf.getvalue() and len(load(WORK / "plain.jsonl")) == 2; ok("overwriting without --resume still works, now with a warning")
(WORK / "cli.jsonl").unlink(missing_ok=True); Fake.reset(); sys.argv = ["x", "--input", str(IN), "--output", str(WORK / "cli.jsonl"), "--max_records", "4"]
assert silently(A.main) == 0 and len(load(WORK / "cli.jsonl")) == 4 and len(Fake.calls) == 4; ok("the command line honors --max_records (4 records, 4 model calls)")
Fake.reset(); sys.argv = ["x", "--input", str(IN), "--output", str(WORK / "cli.jsonl"), "--resume"]
assert silently(A.main) == 0 and len(load(WORK / "cli.jsonl")) == 12 and len(Fake.calls) == 8; ok("the command line honors --resume (8 missing records sent, 12 in the file)")
print("T7  the run fingerprint (a record of what produced each output file)")
Fake.reset(); (WORK / "fp.jsonl").unlink(missing_ok=True); (WORK / "fp.meta.jsonl").unlink(missing_ok=True)
silently(agent().process_file, IN, WORK / "fp.jsonl", max_records=5)
meta = [json.loads(l) for l in open(WORK / "fp.meta.jsonl")]; st, fin = meta[0], meta[1]
assert len(meta) == 2 and st["event"] == "started" and fin["event"] == "finished"; ok("one 'started' line and one 'finished' line are written beside the decisions file")
assert st["system_prompt_sha256"] == PROMPT_SHA256; ok("the fingerprint carries the system-prompt hash, equal to the frozen one")
assert st["model"] == "claude-haiku-4-5-20251001" and st["temperature"] == 0.0 and st["max_tokens"] == 1024; ok("the model snapshot, temperature and max tokens are recorded")
assert st["input_file_sha256"] == hashlib.sha256(IN.read_bytes()).hexdigest() and st["input_records"] == 5 and st["max_records"] == 5 and st["resume"] is False; ok("the input file hash, the record count and the flags are recorded")
assert st["agent_file_sha256"] == hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest(); ok("the agent file's own hash is recorded")
assert set(st["packages"]) == {"anthropic", "langchain-anthropic", "langchain-core", "python"} and st["packages"]["python"]; assert "argv" in st and "platform" in st and "started_at" in st; ok("library and python versions, the command line, the platform and a UTC start time are recorded")
assert fin["processed_this_session"] == 5 and fin["errors"] == 0 and "finished_at" in fin; ok("the 'finished' line records what the session processed")
Fake.reset(); silently(agent().process_file, IN, WORK / "fp.jsonl", resume=True)
meta = [json.loads(l) for l in open(WORK / "fp.meta.jsonl")]
assert len(meta) == 4 and meta[2]["resume"] is True and meta[2]["input_records"] == 12 and meta[3]["processed_this_session"] == 7; ok("a resumed run APPENDS its own fingerprint: every session stays on record")
rows_a = load(WORK / "fp.jsonl"); assert len(rows_a) == 12 and all("system_prompt_sha256" not in r for r in rows_a); ok("the decision rows themselves are unchanged (the fingerprint lives beside them, not in them)")
_real_run = subprocess.run
def _boom(*a, **k): raise RuntimeError("git is not available")
A.subprocess.run = _boom
try:
    Fake.reset(); (WORK / "fp2.jsonl").unlink(missing_ok=True); (WORK / "fp2.meta.jsonl").unlink(missing_ok=True)
    silently(agent().process_file, IN, WORK / "fp2.jsonl", max_records=3)
finally:
    A.subprocess.run = _real_run
m2 = [json.loads(l) for l in open(WORK / "fp2.meta.jsonl")]; assert len(load(WORK / "fp2.jsonl")) == 3 and m2[0]["git_commit"] is None and m2[0]["git_dirty"] is None and m2[0]["system_prompt_sha256"] == PROMPT_SHA256; ok("with git unavailable the run still completes and the fingerprint records git as null")
(WORK / "ro").mkdir(exist_ok=True); (WORK / "ro" / "fp4.meta.jsonl").mkdir(exist_ok=True)
buf = io.StringIO(); Fake.reset()
with contextlib.redirect_stdout(buf): agent().process_file(IN, WORK / "ro" / "fp4.jsonl", max_records=2)
assert len(load(WORK / "ro" / "fp4.jsonl")) == 2 and "WARNING: could not write the run fingerprint" in buf.getvalue(); ok("a fingerprint that cannot be written (a directory is in the way) only warns; the experiment and its decisions are unaffected")

Fake.reset(); (WORK / "fp5.jsonl").unlink(missing_ok=True); (WORK / "fp5.meta.jsonl").unlink(missing_ok=True)
silently(A.BusinessDecisionAgent(api_key="dummy", log_dir=WORK / "logs", model="claude-test-model", temperature=0.3, max_tokens=777).process_file, IN, WORK / "fp5.jsonl", max_records=2)
m5 = json.loads(open(WORK / "fp5.meta.jsonl").readline()); assert m5["model"] == "claude-test-model" and m5["temperature"] == 0.3 and m5["max_tokens"] == 777
ok("NON-default settings are recorded as they were (model, temperature 0.3, max tokens 777): the fingerprint reports what was used, not what is usual")
_orig_fp = A.BusinessDecisionAgent.run_fingerprint
def _raise(self, *a, **k): raise RuntimeError("fingerprint exploded")
A.BusinessDecisionAgent.run_fingerprint = _raise
try:
    Fake.reset(); (WORK / "fp6.jsonl").unlink(missing_ok=True); (WORK / "fp6.meta.jsonl").unlink(missing_ok=True)
    silently(agent().process_file, IN, WORK / "fp6.jsonl", max_records=3)
finally:
    A.BusinessDecisionAgent.run_fingerprint = _orig_fp
m6 = json.loads(open(WORK / "fp6.meta.jsonl").readline()); assert len(load(WORK / "fp6.jsonl")) == 3 and "exploded" in m6["fingerprint_error"]
ok("even if building the fingerprint blows up, the run completes and the failure is itself recorded")

print(f"\n{n_checks} check groups passed")
