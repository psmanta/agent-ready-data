#!/usr/bin/env python3
"""
Regression test for freeze_manifest.py, run against real temporary git repositories.
Run from 1b_dithering/:   python3 test_freeze_manifest.py
"""
import atexit, contextlib, io, json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze_manifest as fm

WORK = Path(tempfile.mkdtemp(prefix="freeze_test_"))
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
n = 0
def ok(label):
    global n; n += 1; print(f"  ✅ {label}")
def run(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): rc = fn(*a, **k)
    return rc, buf.getvalue()
def git(base, *args): return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=base, capture_output=True, text=True, check=True).stdout.strip()

ARTS = ["agent.py", "sub/engine.py", "AMENDMENT.md"]
def repo(name, commit=True):
    d = WORK / name; (d / "sub").mkdir(parents=True)
    for i, p in enumerate(ARTS): (d / p).write_text(f"content {i}\n")
    git(d, "init", "-q")
    if commit: git(d, "add", "-A"); git(d, "commit", "-q", "-m", "init")
    return d

print("F1  writing a freeze")
d = repo("r1"); out = d / "freeze.json"
rc, text = run(fm.write, d, "collection", out, artifacts=ARTS); assert rc == 0 and out.exists(), text
m = json.load(open(out)); head = git(d, "rev-parse", "HEAD")
assert m["git_commit"] == head and set(m["artifacts"]) == set(ARTS) and not any(k.startswith("_") for k in m) and m["stage"] == "collection"; ok("records the commit, a sha256 per artifact and the stage; no scratch keys leak into the file")
import hashlib
assert all(m["artifacts"][p] == hashlib.sha256((d / p).read_bytes()).hexdigest() for p in ARTS); ok("each recorded hash is the true sha256 of the file")

print("F2  verifying")
rc, text = run(fm.verify, d, "collection", out, artifacts=ARTS); assert rc == 0 and "all 3 frozen artifacts match" in text; ok("an untouched freeze verifies")
(d / "sub/engine.py").write_text("tampered\n")
rc, text = run(fm.verify, d, "collection", out); assert rc == 1 and "sub/engine.py: CHANGED" in text and "agent.py" not in text; ok("a changed artifact is detected and NAMED (and unchanged ones are not)")
(d / "sub/engine.py").write_text("content 1\n"); (d / "AMENDMENT.md").unlink()
rc, text = run(fm.verify, d, "collection", out); assert rc == 1 and "AMENDMENT.md: MISSING" in text; ok("a deleted artifact is reported MISSING")
(d / "AMENDMENT.md").write_text("content 2\n")
rc, text = run(fm.verify, d, "analysis", out); assert rc == 1 and "stage" in text; ok("verifying against the wrong stage fails")
rc, text = run(fm.verify, d, "collection", d / "nope.json"); assert rc == 1; ok("verifying with no manifest fails")

print("F3  write-once")
rc, text = run(fm.write, d, "collection", out, artifacts=ARTS); assert rc == 0 and "IDENTICAL" in text; ok("re-writing an identical freeze is a no-op")
(d / "agent.py").write_text("a deliberate, COMMITTED change\n"); git(d, "add", "-A"); git(d, "commit", "-q", "-m", "change")
before = out.read_bytes(); rc, text = run(fm.write, d, "collection", out, artifacts=ARTS)
assert rc == 1 and "DIFFERENT content" in text and out.read_bytes() == before; ok("a committed change after the freeze is refused, and the frozen file is untouched")

print("F4  refusals at write time")
d2 = repo("r2"); (d2 / "agent.py").write_text("uncommitted edit\n")
rc, text = run(fm.write, d2, "collection", d2 / "f.json", artifacts=ARTS); assert rc == 1 and "uncommitted" in text and "agent.py" in text and not (d2 / "f.json").exists(); ok("an artifact with uncommitted changes is refused (the commit would not contain what was hashed)")
rc, text = run(fm.write, d2, "collection", d2 / "f.json", allow_dirty=True, artifacts=ARTS); assert rc == 0 and (d2 / "f.json").exists(); ok("--allow_dirty permits a rehearsal")
d3 = repo("r3"); (d3 / "sub/engine.py").unlink()
rc, text = run(fm.write, d3, "collection", d3 / "f.json", artifacts=ARTS); assert rc == 1 and "missing" in text and "sub/engine.py" in text; ok("a missing artifact is refused, by name")
d4 = repo("r4"); (d4 / "untracked_artifact.py").write_text("x\n")
rc, text = run(fm.write, d4, "collection", d4 / "f.json", artifacts=ARTS + ["untracked_artifact.py"]); assert rc == 1 and "untracked_artifact.py" in text; ok("an artifact never committed (untracked) is refused too")

print("F5  no git repository")
d5 = WORK / "nogit"; d5.mkdir(); (d5 / "a.txt").write_text("x\n")
rc, text = run(fm.write, d5, "collection", d5 / "f.json", artifacts=["a.txt"]); assert rc == 0 and json.load(open(d5 / "f.json"))["git_commit"] is None and "not a git repository" in text; ok("outside a repository it still freezes the hashes, records commit=null, and says so")

print("F6  the real artifact lists")
assert set(fm.ARTIFACTS) == {"collection", "analysis"} and len(set(fm.ARTIFACTS["collection"])) == len(fm.ARTIFACTS["collection"]); ok("two stages; no duplicate artifacts")
assert "1b_DESIGN_AMENDMENT_1.md" in fm.ARTIFACTS["collection"] and "business_decision_agent.py" in fm.ARTIFACTS["collection"] and "freeze_manifest.py" in fm.ARTIFACTS["collection"] and "test_freeze_manifest.py" in fm.ARTIFACTS["collection"]; ok("the amendment, the agent, and the freeze tooling itself are frozen")
assert fm.ARTIFACTS["analysis"] == ["evaluate_h5.py", "evaluate_h6.py", "evaluate_h7.py", "evaluate_h8.py"]; ok("the analysis stage holds exactly the evaluators not yet built")
print(f"\n{n} check groups passed")
