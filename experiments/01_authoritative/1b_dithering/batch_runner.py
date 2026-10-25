#!/usr/bin/env python3
"""
Batch runner: send a set of records through Anthropic's Message Batches API as ONE batch, and write the results in the
same decisions.jsonl schema as the standard (one-call-at-a-time) agent.
=====================================================================================================================
Why: the Batch API costs half as much. The research is self-funded, and the model, prompt and settings are unchanged.

What a "window" means in batch mode. The collection is a randomized block design built to stop window effects (the
same input getting a different decision depending on when it runs; the October 2026 smoke check) from aliasing with
conditions. In standard mode a block is the hour it took to run. In batch mode a block is ONE batch, and the next block
is submitted only after the previous one has ended, so blocks stay separate in time and in order. Anthropic decides
when each request inside a batch runs; that is inside the block, and the randomized design already treats within-block
timing as noise.

Single source of truth. The request body, the response parser and the output row come from business_decision_agent.py
(messages_api_params, parse_decision_text, build_result_row), the same functions the standard path uses.
test_batch_runner.py checks that the request equals what LangChain sends for the same record.

Resumable. The batch id is written to <block>.batches.json as soon as the batch is created, before any waiting. If the
process stops (laptop sleep, closed terminal), re-running picks up the same batch and waits for it; it never pays for a
second submission of a batch that is still running. A marker written just BEFORE creation catches the one gap (a crash
between "create" and "write id"): the runner then stops and asks you to check the Console rather than guess.

Rows added in batch mode: api_mode, batch_id, batch_created_at, batch_ended_at, served_model (the model string the API
reports), stop_reason. processing_time_ms is null (the API reports no per-request latency). cost_usd is at the batch
price (50% of the standard price in model_pricing.json).
"""
import hashlib
import json
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import business_decision_agent as A  # noqa: E402  (loads .env, defines the prompt and the shared builders)

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
BATCH_PRICE_FACTOR = 0.5            # must equal model_pricing.json's batch_discount (checked at load)
API_MODE = "batch"


# ----------------------------------------------------------------------------------------------------------------------
# Building requests
# ----------------------------------------------------------------------------------------------------------------------

def build_requests(records: List[Dict[str, Any]], model: str, temperature: float,
                   max_tokens: int = A.DEFAULT_MAX_TOKENS) -> List[Dict[str, Any]]:
    """One batch request per record; custom_id is the record_id (a task id in the collection)."""
    seen = set()
    out = []
    for r in records:
        cid = r.get("record_id")
        if not isinstance(cid, str) or not CUSTOM_ID_RE.match(cid):
            raise ValueError(f"record_id {cid!r} cannot be a batch custom_id (1-64 of A-Z a-z 0-9 _ -)")
        if cid in seen:
            raise ValueError(f"duplicate record_id {cid!r}: batch custom_ids must be unique")
        seen.add(cid)
        out.append({"custom_id": cid, "params": A.messages_api_params(r, model, temperature, max_tokens)})
    return out


def requests_sha256(requests: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(requests, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------------------------------------------------

_PRICING = None


def batch_cost(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Cost at the batch price, from the same model_pricing.json the standard path uses (None if no price is known)."""
    global _PRICING
    if _PRICING is None:
        path = HERE.parent.parent.parent / "shared" / "agents" / "model_pricing.json"
        cfg = json.load(open(path)) if path.exists() else {}
        if cfg.get("batch_discount", BATCH_PRICE_FACTOR) != BATCH_PRICE_FACTOR:
            raise ValueError("model_pricing.json's batch_discount disagrees with batch_runner.BATCH_PRICE_FACTOR")
        _PRICING = cfg.get("pricing", {})
    keys = [k for k in _PRICING if k in model.lower()]
    price = _PRICING[max(keys, key=len)] if keys else None          # longest matching key wins
    if price is None or input_tokens is None or output_tokens is None:
        return None
    per = price.get("per_tokens", 1000)
    return BATCH_PRICE_FACTOR * (input_tokens / per * price["input"] + output_tokens / per * price["output"])


# ----------------------------------------------------------------------------------------------------------------------
# State file (<block>.batches.json)
# ----------------------------------------------------------------------------------------------------------------------

def load_state(path: Path) -> Dict[str, Any]:
    if Path(path).exists():
        return json.load(open(path))
    return {"batches": []}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)                                     # atomic: a crash never leaves a half-written state file


def _iso(x) -> Optional[str]:
    if x is None:
        return None
    return x.isoformat() if hasattr(x, "isoformat") else str(x)


def _get(obj, name, default=None):
    """Attribute or key access, so the SDK's objects and plain dicts (tests) both work."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ----------------------------------------------------------------------------------------------------------------------
# Fingerprint (same fields as the standard path, plus batch facts)
# ----------------------------------------------------------------------------------------------------------------------

def batch_fingerprint(input_path: Path, output_path: Path, n_records: int, model: str, temperature: float,
                      max_tokens: int) -> Dict[str, Any]:
    stand_in = types.SimpleNamespace(model=model, temperature=temperature, configured_max_tokens=max_tokens)
    try:
        fp = A.BusinessDecisionAgent.run_fingerprint(stand_in, input_path, output_path, n_records, None, True)
    except Exception as e:                               # best effort, never stops a run
        fp = {"event": "started", "fingerprint_error": str(e)}
    fp["api_mode"] = API_MODE
    try:
        fp["batch_runner_file_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        fp["batch_runner_file_sha256"] = None
    return fp


def append_meta(meta_path: Path, entry: Dict[str, Any]) -> None:
    A.BusinessDecisionAgent._append_meta(meta_path, entry)


# ----------------------------------------------------------------------------------------------------------------------
# Running one batch
# ----------------------------------------------------------------------------------------------------------------------

def default_client():
    import anthropic
    return anthropic.Anthropic(default_headers=A.PINNED_HEADERS)   # the same pinned API version as the standard path


def _message_text(message) -> str:
    parts = []
    for block in _get(message, "content", []) or []:
        if _get(block, "type") == "text":
            parts.append(_get(block, "text", ""))
    return "".join(parts)


def collect_results(client, batch_id: str, records_by_id: Dict[str, Dict[str, Any]], model: str,
                    created_at: Optional[str], ended_at: Optional[str]) -> Dict[str, Any]:
    """Turn a finished batch's results into decision rows. Results come back in any order; custom_id matches them."""
    rows, counts, unknown = [], {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}, 0
    for res in client.messages.batches.results(batch_id):
        cid = _get(res, "custom_id")
        outcome = _get(res, "result")
        kind = _get(outcome, "type")
        counts[kind] = counts.get(kind, 0) + 1
        if cid not in records_by_id:
            unknown += 1
            continue
        if kind != "succeeded":
            continue                                      # not billed; the caller resends it in the next batch
        message = _get(outcome, "message")
        usage = _get(message, "usage")
        in_tok, out_tok = _get(usage, "input_tokens"), _get(usage, "output_tokens")
        decision = A.parse_decision_text(_message_text(message))
        row = A.build_result_row(records_by_id[cid], decision, model=model, cost_usd=batch_cost(model, in_tok, out_tok),
                                 input_tokens=in_tok, output_tokens=out_tok, latency_ms=None,
                                 timestamp=ended_at or datetime.now(timezone.utc).isoformat())
        row.update({"api_mode": API_MODE, "batch_id": batch_id, "batch_created_at": created_at,
                    "batch_ended_at": ended_at, "served_model": _get(message, "model"),
                    "stop_reason": _get(message, "stop_reason")})
        rows.append(row)
    if unknown:
        raise RuntimeError(f"batch {batch_id} returned {unknown} result(s) for records this block did not send")
    return {"rows": rows, "counts": counts}


def run_batch(records: List[Dict[str, Any]], *, decisions_path: Path, state_path: Path, meta_path: Path,
              input_path: Path, model: str, temperature: float, max_tokens: int = A.DEFAULT_MAX_TOKENS,
              client=None, poll_seconds: float = 60.0, sleep: Callable[[float], None] = time.sleep,
              log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Send `records` as one batch (or finish waiting for this block's unfinished batch), append the decisions to
    decisions_path and return {"batch_id", "counts", "n_rows", "created_at", "ended_at", "resumed"}.

    Only records that still need a decision should be passed; the caller decides that from the decisions file."""
    client = client or default_client()
    state = load_state(state_path)
    records_by_id = {r["record_id"]: r for r in records}

    if state.get("submitting"):
        raise RuntimeError(
            f"{state_path.name} says a batch submission started at {state['submitting']} but no batch id was saved. "
            f"Check the Console (Batches) for a batch created then: if it exists, add its id to this file as an "
            f"uncollected batch; if not, delete the 'submitting' key. Nothing was sent.")

    open_batches = [b for b in state["batches"] if not b.get("collected")]
    resumed = bool(open_batches)
    if open_batches:
        entry = open_batches[0]
        # On resume, results are matched against what that batch sent, not what is pending now.
        records_by_id = {**records_by_id, **{cid: rec for cid, rec in entry.get("_records", {}).items()}}
        log(f"    resuming batch {entry['batch_id']} (submitted {entry['submitted_at']}); not resubmitting")
    else:
        if not records:
            return {"batch_id": None, "counts": {}, "n_rows": 0, "created_at": None, "ended_at": None, "resumed": False}
        requests = build_requests(records, model, temperature, max_tokens)
        fp = batch_fingerprint(input_path, decisions_path, len(records), model, temperature, max_tokens)
        state["submitting"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
        batch = client.messages.batches.create(requests=requests)
        entry = {"batch_id": _get(batch, "id"), "submitted_at": state["submitting"],
                 "created_at": _iso(_get(batch, "created_at")), "n_requests": len(requests),
                 "requests_sha256": requests_sha256(requests), "collected": False,
                 "_records": records_by_id}
        state["batches"].append(entry)
        del state["submitting"]
        save_state(state_path, state)
        fp.update({"event": "batch_submitted", "batch_id": entry["batch_id"], "n_requests": len(requests),
                   "requests_sha256": entry["requests_sha256"]})
        append_meta(meta_path, fp)
        log(f"    submitted batch {entry['batch_id']} ({len(requests)} requests)")

    batch_id = entry["batch_id"]
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = _get(batch, "processing_status")
        if status == "ended":
            break
        rc = _get(batch, "request_counts")
        log(f"    batch {batch_id}: {status}; {_get(rc, 'succeeded', 0)} succeeded, {_get(rc, 'processing', 0)} processing")
        sleep(poll_seconds)

    created_at, ended_at = _iso(_get(batch, "created_at")), _iso(_get(batch, "ended_at"))
    got = collect_results(client, batch_id, records_by_id, model, created_at, ended_at)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_path, "a") as f:                  # block_status keeps the last row per task, so appends are safe
        for row in got["rows"]:
            f.write(json.dumps(row, default=str) + "\n")

    entry.update({"collected": True, "created_at": created_at, "ended_at": ended_at, "counts": got["counts"]})
    entry.pop("_records", None)                           # the decisions file now holds what matters
    save_state(state_path, state)
    append_meta(meta_path, {"event": "batch_collected", "api_mode": API_MODE, "batch_id": batch_id,
                            "created_at": created_at, "ended_at": ended_at, "counts": got["counts"],
                            "rows_written": len(got["rows"]), "collected_at": datetime.now(timezone.utc).isoformat()})
    log(f"    batch {batch_id} ended: {got['counts']}")
    return {"batch_id": batch_id, "counts": got["counts"], "n_rows": len(got["rows"]),
            "created_at": created_at, "ended_at": ended_at, "resumed": resumed}
