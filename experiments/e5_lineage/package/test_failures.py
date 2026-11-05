"""Offline failure-path checks for the E130 session hook and launcher guards."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SOURCE_BASE = Path("/home/ec2-user/research/e120-captain-20261103")
sys.path.insert(0, str(SOURCE_BASE))
sys.path.insert(0, str(SOURCE_BASE / "author-e5-pinned"))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput, get_model
from evals.alignment_research.environment import AGENT_A, AGENT_B
from evals.alignment_research.task import load_scenarios_as_dataset
from evals.alignment_research.tools.filesystem_tools import get_filesystem_tools
from evals.alignment_research.tools.forum_tools import get_forum_tools
from evals.common.retry_utils import GenerationError

from adapter import DeliveryHalt, FixedADelivery, load_manifest
from launcher import _session_gate, _technical_failure, run_attempt
import source_hooks
from source_hooks import (
    BASELINE_GOAL_SENTENCE, REPLACEMENT_GOAL_SENTENCE,
    _agent_generate, make_b_solver, transform_b_system_prompt,
)


class OneToolModel:
    def __init__(self):
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("unexpected B generation after delivery halt")
        return ModelOutput.for_tool_call(
            model="mockllm/gpt-6-sol", tool_name="get_leaderboard",
            tool_arguments={}, tool_call_id="failure_tool_001",
        )


async def check_dispatch_halt() -> None:
    model = OneToolModel()

    async def halt(tc, tool):
        raise DeliveryHalt("simulated A delivery issue")

    try:
        await _agent_generate(
            model, [ChatMessageUser(content="start")], GenerateConfig(),
            get_forum_tools(AGENT_B), session_index=1,
            dispatch_tool=halt, halt_type=DeliveryHalt,
        )
    except DeliveryHalt:
        pass
    else:
        raise AssertionError("DeliveryHalt was swallowed by source loop")
    assert model.calls == 1


async def check_pre_model_stop() -> None:
    model = OneToolModel()

    def event(kind, payload):
        if kind == "before_model_call":
            raise DeliveryHalt("simulated chaperone stop")

    try:
        await _agent_generate(
            model, [ChatMessageUser(content="start")], GenerateConfig(),
            get_forum_tools(AGENT_B), session_index=1,
            event_callback=event, halt_type=DeliveryHalt,
        )
    except DeliveryHalt:
        pass
    else:
        raise AssertionError("pre-model stop did not halt")
    assert model.calls == 0


async def check_gate_refusal(root: Path) -> None:
    task = asyncio.create_task(_session_gate(root, 1, [], [], "done"))
    request_path = root / "gates/session_001.request.json"
    for _ in range(100):
        if request_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("gate request was not written")
    request = json.loads(request_path.read_text())
    (root / "gates/session_001.decision.json").write_text(json.dumps({
        "session": 1, "nonce": request["nonce"],
        "decision": "continue", "chaperone": "aero", "rationale": "",
    }))
    try:
        await asyncio.wait_for(task, timeout=5)
    except RuntimeError as error:
        assert "invalid chaperone gate" in str(error)
    else:
        raise AssertionError("empty-rationale gate decision was accepted")


def check_frozen_guard(attempt: Path) -> None:
    try:
        run_attempt(attempt)
    except RuntimeError as error:
        assert "frozen package/launcher.py" in str(error)
    else:
        raise AssertionError("mutable launcher was allowed to run frozen attempt")


def check_classification() -> None:
    log = SimpleNamespace(status="success")
    sample = SimpleNamespace(metadata={"stop_reason": "generation_error"}, error=None)
    assert _technical_failure(log, sample)
    sample.metadata["stop_reason"] = "chaperone_stop"
    assert not _technical_failure(log, sample)
    log.status = "error"
    assert _technical_failure(log, sample)


def check_goal_sentence_guard() -> None:
    for rendered, count in (("no B goal sentence", 0),
                            (BASELINE_GOAL_SENTENCE * 2, 2)):
        try:
            transform_b_system_prompt(rendered)
        except ValueError as error:
            assert f"found {count}" in str(error)
        else:
            raise AssertionError("invalid B goal count was accepted")
    assert transform_b_system_prompt(BASELINE_GOAL_SENTENCE) == REPLACEMENT_GOAL_SENTENCE


def check_generation_error_eval(attempt: Path, package: Path) -> None:
    root = attempt / "w2s"
    os.environ["W2S_REPO_ROOT"] = str(root)
    model = get_model("mockllm/gpt-6-sol")
    updates = load_manifest(package / "stimulus/manifest.json")
    a_tools = get_forum_tools(AGENT_A) + get_filesystem_tools(AGENT_A)
    a_write = next(tool for tool in a_tools if tool.__registry_info__.name == "Write")
    a_share = next(tool for tool in a_tools if tool.__registry_info__.name == "share_finding")
    seen = []

    def adapter_factory(env):
        return FixedADelivery(
            env=env, updates=updates, a_write_tool=a_write, a_share_tool=a_share,
            journal_path=attempt / "mock_error_delivery.json",
        )

    def before_cleanup(env):
        shutil.copytree(env.workspace_root, attempt / "mock_error_workspace", symlinks=False)

    async def fail_generate(*args, **kwargs):
        raise GenerationError("offline injected generation error")

    original = source_hooks.model_generate_with_retry
    source_hooks.model_generate_with_retry = fail_generate
    try:
        solver = make_b_solver(
            b_model=model, adapter_factory=adapter_factory,
            session_gate=lambda *args: (_ for _ in ()).throw(AssertionError("gate after error")),
            event_callback=lambda kind, payload: seen.append(kind),
            before_cleanup=before_cleanup, w2s_root=root,
            halt_type=DeliveryHalt, max_b_sessions=50,
        )
        task = Task(
            dataset=load_scenarios_as_dataset(
                capability="persuasion", scenario_name="discredit_and_demoralize"
            ), solver=solver, scorer=[], epochs=1, name="e130-offline-generation-error",
        )
        logs = inspect_eval(task, model=model, log_dir=str(attempt / "mock_error_logs"),
                            log_format="eval", display="none", max_samples=1)
    finally:
        source_hooks.model_generate_with_retry = original
    sample = logs[0].samples[0]
    assert sample.metadata["stop_reason"] == "generation_error", sample.metadata
    assert sample.metadata["b_sessions"] == 0, sample.metadata
    assert "generation_error" in seen, seen
    assert (attempt / "mock_error_workspace").is_dir()
    assert _technical_failure(logs[0], sample)


def main() -> None:
    attempt = Path(sys.argv[1]).resolve(strict=True)
    generation_attempt = Path(sys.argv[2]).resolve(strict=True)
    package = Path(sys.argv[3]).resolve(strict=True)
    check_frozen_guard(attempt)
    check_classification()
    check_goal_sentence_guard()
    asyncio.run(check_dispatch_halt())
    asyncio.run(check_pre_model_stop())
    with tempfile.TemporaryDirectory(prefix="e130_gate_failure_") as directory:
        asyncio.run(check_gate_refusal(Path(directory)))
    check_generation_error_eval(generation_attempt, package)
    print(json.dumps({
        "status": "offline_failure_paths_green",
        "checks": ["frozen_guard", "technical_classification", "goal_sentence_guard", "dispatch_halt",
                   "pre_model_stop", "gate_refusal", "native_eval_generation_error"],
    }))


if __name__ == "__main__":
    main()
