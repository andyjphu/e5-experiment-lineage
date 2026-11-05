"""Offline pinned-source Inspect consumer smoke; no provider or Daybreak call."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
from pathlib import Path

SOURCE_BASE = Path("/home/ec2-user/research/e120-captain-20261103")
AUTHOR = SOURCE_BASE / "author-e5-pinned"
sys.path.insert(0, str(SOURCE_BASE))
sys.path.insert(0, str(AUTHOR))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from evals.alignment_research.task import load_scenarios_as_dataset
from evals.alignment_research.tools.filesystem_tools import get_filesystem_tools
from evals.alignment_research.tools.forum_tools import get_forum_tools
from evals.alignment_research.environment import AGENT_A
from evals.alignment_research import solver as source_solver

from adapter import FixedADelivery, DeliveryHalt, load_manifest
from launcher import _append_jsonl, _jsonable
from preflight import verify_initial_w2s_tree
from source_hooks import (
    BASELINE_GOAL_SENTENCE, REPLACEMENT_GOAL_SENTENCE,
    make_b_solver, transform_b_system_prompt,
)


def run_smoke(attempt: Path, package: Path) -> dict:
    attempt = attempt.resolve(strict=True)
    package = package.resolve(strict=True)
    root = attempt / "w2s"
    verify_initial_w2s_tree(root)
    os.environ["W2S_REPO_ROOT"] = str(root)
    post_files = (
        "0003_agent_a_A_matched_disagreement-mask_comparison_for_posterior_correct.json",
        "0004_agent_a_Keep_EM_as_the_comparator__take_the_mask-first_study_next.json",
    )
    outputs = []
    for index, post_file in enumerate(post_files, 1):
        outputs.extend((
            ModelOutput.for_tool_call(
                model="mockllm/gpt-6-sol", tool_name="Glob",
                tool_arguments={"pattern": "w2s_research/research_loop/shared_findings/*.json"},
                tool_call_id=f"mock_forum_{index}",
            ),
            ModelOutput.for_tool_call(
                model="mockllm/gpt-6-sol", tool_name="Read",
                tool_arguments={"file_path": "w2s_research/research_loop/shared_findings/" + post_file},
                tool_call_id=f"mock_read_{index}",
            ),
            ModelOutput.from_content(model="mockllm/gpt-6-sol", content=f"Session {index} complete"),
        ))
    model = get_model("mockllm/gpt-6-sol", custom_outputs=outputs)
    updates = load_manifest(package / "stimulus/manifest.json")
    a_tools = get_forum_tools(AGENT_A) + get_filesystem_tools(AGENT_A)
    a_write = next(tool for tool in a_tools if tool.__registry_info__.name == "Write")
    a_share = next(tool for tool in a_tools if tool.__registry_info__.name == "share_finding")
    holder = {}
    events = []
    decisions = []
    original_prompt = []

    def adapter_factory(env):
        original_prompt.append(source_solver._render_system_prompt(
            env.agent_b_workspace, env.agent_b_target_idea, ""
        ))
        adapter = FixedADelivery(
            env=env, updates=updates, a_write_tool=a_write, a_share_tool=a_share,
            journal_path=attempt / "mock_delivery.json",
        )
        holder["adapter"] = adapter
        return adapter

    def event_callback(kind, payload):
        events.append((kind, payload))
        _append_jsonl(attempt / "mock_events.jsonl", {"kind": kind, "payload": _jsonable(payload)})

    def gate(session, history, delta, response):
        decisions.append((session, len(history), len(delta), response))
        return "continue" if session == 1 else "stop"

    def before_cleanup(env):
        shutil.copytree(env.workspace_root, attempt / "mock_workspace_final", symlinks=False)

    solver = make_b_solver(
        b_model=model, adapter_factory=adapter_factory, session_gate=gate,
        event_callback=event_callback, before_cleanup=before_cleanup,
        w2s_root=root, halt_type=DeliveryHalt, max_b_sessions=50,
    )
    dataset = load_scenarios_as_dataset(
        capability="persuasion", scenario_name="discredit_and_demoralize",
    )
    task = Task(dataset=dataset, solver=solver, scorer=[], epochs=1, name="e130-offline-consumer-smoke")
    logs = inspect_eval(task, model=model, log_dir=str(attempt / "mock_logs"),
                        log_format="eval", display="none", max_samples=1,
                        log_model_api=True)
    sample = logs[0].samples[0]
    delivery = holder["adapter"].snapshot()
    assert sample.metadata["b_sessions"] == 2, sample.metadata
    assert sample.metadata["stop_reason"] == "chaperone_stop", sample.metadata
    assert delivery["cursor"] == 2 and len(delivery["deliveries"]) == 2, delivery
    assert [item[0] for item in decisions] == [1, 2], decisions
    model_events = [payload for kind, payload in events if kind == "model"]
    tool_results = [payload for kind, payload in events if kind == "tool_result"]
    assert len(model_events) == 6, events
    assert len(tool_results) == 4, events
    assert len(original_prompt) == 1
    baseline = original_prompt[0]
    actual = model_events[0]["input"][0].content
    assert baseline.count(BASELINE_GOAL_SENTENCE) == 1
    assert actual == transform_b_system_prompt(baseline)
    assert actual.replace(REPLACEMENT_GOAL_SENTENCE, BASELINE_GOAL_SENTENCE, 1) == baseline
    assert [json.loads(item["result"])["id"]
            for item in tool_results if item["call"].function == "Read"] == [3, 4]
    assert len((attempt / "mock_events.jsonl").read_text().splitlines()) == len(events)
    assert len(list((attempt / "mock_logs").glob("*.eval"))) == 1
    assert (attempt / "mock_workspace_final" / "forum").exists()
    result = {
        "status": "offline_consumer_green", "native_eval_status": logs[0].status,
        "b_sessions": sample.metadata["b_sessions"], "stop_reason": sample.metadata["stop_reason"],
        "deliveries": delivery["deliveries"], "event_kinds": [item[0] for item in events],
        "source_prompt_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "effective_prompt_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        "source_goal_sentence": BASELINE_GOAL_SENTENCE,
        "effective_goal_sentence": REPLACEMENT_GOAL_SENTENCE,
    }
    (attempt / "mock_consumer_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run_smoke(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
