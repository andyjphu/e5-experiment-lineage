"""Prepare and run one isolated E130 attempt through pinned source E5/Inspect.

`prepare` is offline. `run` invokes the B provider and requires Captain Ursa's
separate package release. Each completed B session blocks at a file gate until
its chaperone writes a matching decision. Never reuse an attempt directory.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import uuid
from pathlib import Path
from typing import Any

SOURCE_BASE = Path("/home/ec2-user/research/e120-captain-20261103")
if str(SOURCE_BASE) not in sys.path:
    sys.path.insert(0, str(SOURCE_BASE))

from e120_source_e5_sol.prepare import file_sha, tree_sha
from e120_source_e5_sol.route import make_model

from preflight import verify_initial_w2s_tree, verify_served_wire, verify_source_baseline
from adapter import DeliveryHalt, FixedADelivery, load_manifest


PACKAGE_FILES = (
    "launcher.py", "source_hooks.py", "adapter.py", "preflight.py",
    "SOURCE_HOOK_DIFF.patch", "test_consumer.py", "test_failures.py", "test_adapter.py",
    "stimulus/manifest.json", "stimulus/files/mask_loss.py",
    "stimulus/files/eligible_random.py", "stimulus/files/mask_covariate_balance.py",
)
AUTHOR = SOURCE_BASE / "author-e5-pinned"
W2S_SOURCE = SOURCE_BASE / "automated-w2s-research-git"
SOURCE_PACKAGE = SOURCE_BASE / "e120_source_e5_sol"
DOTENV = SOURCE_BASE / ".env.daybreak"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _technical_failure(log: Any, sample: Any) -> bool:
    return (
        sample.metadata.get("stop_reason") == "generation_error"
        or log.status != "success"
        or getattr(sample, "error", None) is not None
    )


def _package_hashes(package: Path) -> dict[str, str]:
    return {name: file_sha(package / name) for name in PACKAGE_FILES}


def _extract_initial_w2s(destination: Path, commit: str) -> None:
    raw = subprocess.check_output([
        "git", "-C", str(W2S_SOURCE), "archive", commit,
        "w2s_research/ideas", "w2s_research/core",
    ])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if (not member.name.startswith(("w2s_research/ideas/", "w2s_research/core/"))
                    and member.name not in {"w2s_research", "w2s_research/ideas", "w2s_research/core"}):
                raise ValueError(f"unexpected W2S archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError("W2S source archive contains link")
        archive.extractall(destination, filter="data")
    verify_initial_w2s_tree(destination)


def prepare(attempt: Path, package: Path, run: str) -> dict[str, Any]:
    """Freeze a new package and clean W2S tree without a model call."""
    if run not in {"run005", "run006", "run007", "run008"}:
        raise ValueError("run must be one of the four authorized E130 batch02 runs")
    package = package.resolve(strict=True)
    baseline = verify_source_baseline(AUTHOR, W2S_SOURCE, SOURCE_PACKAGE)
    load_manifest(package / "stimulus/manifest.json")
    package_hashes = _package_hashes(package)
    attempt = attempt.resolve()
    if attempt.exists():
        raise FileExistsError(f"attempt already exists: {attempt}")
    attempt.mkdir(parents=True)
    frozen = attempt / "package"
    frozen.mkdir()
    for name in PACKAGE_FILES:
        destination = frozen / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, destination)
    if _package_hashes(frozen) != package_hashes:
        raise RuntimeError("package changed during freeze")
    manifest = json.loads((SOURCE_PACKAGE / "discovery_manifest.json").read_text())
    _extract_initial_w2s(attempt / "w2s", manifest["w2s_commit"])
    (attempt / "logs").mkdir()
    receipt = {
        "status": "prepared_no_model_call", "experiment": "E130", "run": run,
        "attempt": str(attempt), "package_hashes": package_hashes,
        "initial_w2s_tree_sha256": tree_sha(attempt / "w2s"),
        "baseline": baseline, "stimulus_sha256": package_hashes["stimulus/manifest.json"],
        "model_b": "openai/gpt-6-sol", "max_b_sessions": 50,
    }
    _write_json(attempt / "prepare_receipt.json", receipt)
    return receipt


async def _session_gate(attempt: Path, session: int, history: list, delta: list, response: str) -> str:
    nonce = uuid.uuid4().hex
    request = {
        "status": "awaiting_chaperone", "session": session, "nonce": nonce,
        "history_messages": len(history), "delta_messages": len(delta),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "event_journal": str(attempt / "events.jsonl"),
    }
    request_path = attempt / "gates" / f"session_{session:03d}.request.json"
    decision_path = attempt / "gates" / f"session_{session:03d}.decision.json"
    if request_path.exists() or decision_path.exists():
        raise RuntimeError(f"stale gate file for session {session}")
    _write_json(request_path, request)
    _append_jsonl(attempt / "events.jsonl", {"kind": "gate_wait", **request})
    while True:
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if (decision.get("session") != session or decision.get("nonce") != nonce
                    or decision.get("decision") not in {"continue", "stop"}
                    or not _nonempty_text(decision.get("chaperone"))
                    or not _nonempty_text(decision.get("rationale"))):
                raise RuntimeError(f"invalid chaperone gate decision for session {session}")
            _append_jsonl(attempt / "events.jsonl", {"kind": "gate_decision", **decision})
            return decision["decision"]
        await asyncio.sleep(2)


def run_attempt(attempt: Path) -> dict[str, Any]:
    """Invoke B through source route and native Inspect .eval after release."""
    attempt = attempt.resolve(strict=True)
    if Path(__file__).resolve(strict=True).parent != attempt / "package":
        raise RuntimeError("run must execute this attempt's frozen package/launcher.py")
    receipt = json.loads((attempt / "prepare_receipt.json").read_text())
    if receipt.get("status") != "prepared_no_model_call" or receipt.get("attempt") != str(attempt):
        raise ValueError("invalid E130 preparation receipt")
    if _package_hashes(attempt / "package") != receipt["package_hashes"]:
        raise ValueError("frozen E130 package hash mismatch")
    if (attempt / "launch_intent.json").exists() or (attempt / "run_receipt.json").exists():
        raise FileExistsError("attempt already launched")
    if list((attempt / "logs").glob("*.eval")):
        raise FileExistsError("native eval log already exists")
    verify_initial_w2s_tree(attempt / "w2s")
    baseline = verify_source_baseline(AUTHOR, W2S_SOURCE, SOURCE_PACKAGE)
    if baseline != receipt["baseline"]:
        raise ValueError("source baseline changed after preparation")
    updates = load_manifest(attempt / "package/stimulus/manifest.json")
    from dotenv import load_dotenv
    load_dotenv(DOTENV)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI credential unavailable")
    os.environ["W2S_REPO_ROOT"] = str(attempt / "w2s")
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(attempt / "package"))
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.model import get_model
    from evals.alignment_research.task import load_scenarios_as_dataset
    from evals.alignment_research.tools.filesystem_tools import get_filesystem_tools
    from evals.alignment_research.tools.forum_tools import get_forum_tools
    from evals.alignment_research.environment import AGENT_A
    from source_hooks import make_b_solver

    manifest = json.loads((SOURCE_PACKAGE / "discovery_manifest.json").read_text())
    b_model = make_model(get_model, manifest["model_b"], "B", attempt / "daybreak_wire.jsonl")
    a_tools = get_forum_tools(AGENT_A) + get_filesystem_tools(AGENT_A)
    a_write = next(tool for tool in a_tools if tool.__registry_info__.name == "Write")
    a_share = next(tool for tool in a_tools if tool.__registry_info__.name == "share_finding")
    holder: dict[str, FixedADelivery] = {}

    def adapter_factory(env):
        adapter = FixedADelivery(
            env=env, updates=updates, a_write_tool=a_write, a_share_tool=a_share,
            journal_path=attempt / "delivery.json",
        )
        holder["adapter"] = adapter
        return adapter

    def event_callback(kind: str, payload: dict[str, Any]) -> None:
        if kind == "before_model_call":
            stop_path = attempt / "stop_request.json"
            if stop_path.exists():
                stop = json.loads(stop_path.read_text())
                if (stop.get("run") != receipt["run"]
                        or not _nonempty_text(stop.get("chaperone"))
                        or not _nonempty_text(stop.get("reason"))):
                    raise DeliveryHalt("invalid mid-session stop request")
                _append_jsonl(attempt / "events.jsonl", {
                    "utc": dt.datetime.now(dt.UTC).isoformat(),
                    "kind": "mid_session_stop", "payload": stop,
                })
                raise DeliveryHalt(f"chaperone stopped mid-session: {stop['reason']}")
        _append_jsonl(attempt / "events.jsonl", {
            "utc": dt.datetime.now(dt.UTC).isoformat(), "kind": kind,
            "payload": _jsonable(payload),
        })

    def before_cleanup(env):
        snapshot = attempt / "workspace_final"
        if snapshot.exists():
            raise FileExistsError("workspace snapshot already exists")
        # Dereference source symlinks so the forum and W2S files survive cleanup.
        shutil.copytree(env.workspace_root, snapshot, symlinks=False)
        _write_json(attempt / "workspace_receipt.json", {
            "workspace_tree_sha256": tree_sha(snapshot),
            "w2s_after_tree_sha256": tree_sha(attempt / "w2s"),
            "delivery": holder["adapter"].snapshot(),
        })

    solver = make_b_solver(
        b_model=b_model, adapter_factory=adapter_factory,
        session_gate=lambda session, history, delta, response:
            _session_gate(attempt, session, history, delta, response),
        event_callback=event_callback, before_cleanup=before_cleanup,
        w2s_root=attempt / "w2s", halt_type=DeliveryHalt,
        max_b_sessions=50, cached_results=manifest["cached_results"],
    )
    dataset = load_scenarios_as_dataset(
        capability=manifest["capability"], scenario_name=manifest["scenario"]
    )
    if len(dataset.samples) != 1:
        raise ValueError("one source scenario required")
    _write_json(attempt / "launch_intent.json", {
        "status": "about_to_launch", "utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": os.uname().nodename, "pid": os.getpid(),
        "process_lstart": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True,
        ).strip(),
        "process_args": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "args="], text=True,
        ).strip(),
        "run": receipt["run"], "package_hashes": receipt["package_hashes"],
        "source_provenance": baseline, "model_b": manifest["model_b"],
        "scorers": "omitted; trace-based audit", "log_model_api": True,
    })
    task = Task(
        dataset=dataset, solver=solver, scorer=[], epochs=1,
        name="e130-fixed-a-fresh-b", metadata={
            "experiment": "E130", "run": receipt["run"],
            "source_commit": baseline["provenance"]["source_commit"],
            "model_b": manifest["model_b"], "max_b_sessions": 50,
            "source_solver": "source-derived B-only adapter; see package/source_hooks.py",
            "scorers_omitted": True,
        },
    )
    try:
        logs = inspect_eval(
            task, model=b_model, log_dir=str(attempt / "logs"),
            log_format="eval", display="none", max_samples=1,
            log_model_api=True,
        )
        paths = list((attempt / "logs").glob("*.eval"))
        if len(logs) != 1 or len(paths) != 1 or not logs[0].samples or len(logs[0].samples) != 1:
            raise RuntimeError("one complete native eval required")
        sample = logs[0].samples[0]
        technical_failure = _technical_failure(logs[0], sample)
        wire_path = attempt / "daybreak_wire.jsonl"
        wire = verify_served_wire(wire_path) if wire_path.exists() and wire_path.stat().st_size else {
            "status": "no_completed_wire_response", "responses": 0,
        }
        if not technical_failure and wire["status"] != "served_route_verified":
            raise RuntimeError("successful attempt lacks verified served route")
        result = {
            "status": "technical_failure" if technical_failure else logs[0].status,
            "native_eval_status": logs[0].status, "run": receipt["run"],
            "eval_path": str(paths[0]), "eval_sha256": file_sha(paths[0]),
            "wire": wire, "package_hashes": receipt["package_hashes"],
            "conversation_turns": sample.metadata.get("conversation_turns"),
            "b_sessions": sample.metadata.get("b_sessions"),
            "stop_reason": sample.metadata.get("stop_reason"),
            "w2s_after_tree_sha256": tree_sha(attempt / "w2s"),
            "workspace_receipt_sha256": file_sha(attempt / "workspace_receipt.json"),
            "events_sha256": file_sha(attempt / "events.jsonl"),
            "delivery_sha256": file_sha(attempt / "delivery.json"),
        }
        _write_json(attempt / "run_receipt.json", result)
        if technical_failure:
            _write_json(attempt / "failed_manifest.json", {
                "status": "technical_failure", "reason": sample.metadata.get("stop_reason")
                    or f"native eval status {logs[0].status}",
                "run_receipt": str(attempt / "run_receipt.json"),
                "native_eval": str(paths[0]),
            })
        return result
    except BaseException as error:
        _write_json(attempt / "run_error.json", {
            "status": "failed", "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
        })
        _write_json(attempt / "failed_manifest.json", {
            "status": "technical_failure", "reason": str(error),
            "error_type": type(error).__name__, "native_eval_dir": str(attempt / "logs"),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--attempt", type=Path, required=True)
    p_prepare.add_argument("--package", type=Path, required=True)
    p_prepare.add_argument("--run", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.attempt, args.package, args.run)
    else:
        result = run_attempt(args.attempt)
    print(json.dumps(result, indent=2))
    if result.get("status") == "technical_failure":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
