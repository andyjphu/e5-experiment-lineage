"""Offline consumer-path checks using the pinned source E5 tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from functools import wraps
from pathlib import Path

from adapter import DeliveryHalt, FixedADelivery, load_manifest
from evals.alignment_research.environment import reset_environment_state
from evals.alignment_research.tools.filesystem_tools import get_filesystem_tools
from evals.alignment_research.tools.forum_tools import get_forum_tools


FORUM = "w2s_research/research_loop/shared_findings"
CODE = "w2s_research/ideas/gradient_masking/plan.py"


def named(tools, name):
    return next(tool for tool in tools if tool.__registry_info__.name == name)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e130_a_delivery_test_")
        self.root = Path(self.temporary.name)
        self.forum = self.root / "forum"
        self.forum.mkdir()
        self.code_root = self.root / "w2s/w2s_research"
        (self.code_root / "ideas").mkdir(parents=True)
        (self.code_root / "core").mkdir()
        self.env = reset_environment_state()
        self.env.workspace_root = self.root
        self.env.forum_dir = self.forum
        for actor in ("agent_a", "agent_b"):
            workspace = self.root / actor
            research = workspace / "w2s_research"
            (research / "research_loop").mkdir(parents=True)
            (research / "ideas").symlink_to(self.code_root / "ideas")
            (research / "core").symlink_to(self.code_root / "core")
            (research / "research_loop/shared_findings").symlink_to(self.forum)
            setattr(self.env, f"{actor}_workspace", workspace)
        self.env.share_finding_to_disk("system", "seed one", "baseline", "observation")
        self.env.share_finding_to_disk("system", "seed two", "baseline", "observation")
        self.a_write = named(get_filesystem_tools("agent_a"), "Write")
        self.a_share = named(get_forum_tools("agent_a"), "share_finding")
        self.b_ls = named(get_filesystem_tools("agent_b"), "LS")
        self.b_glob = named(get_filesystem_tools("agent_b"), "Glob")
        self.b_read = named(get_filesystem_tools("agent_b"), "Read")
        self.b_share = named(get_forum_tools("agent_b"), "share_finding")

    async def asyncTearDown(self):
        self.temporary.cleanup()

    def adapter(self, updates, write_tool=None):
        return FixedADelivery(env=self.env, updates=updates,
                              a_write_tool=write_tool or self.a_write,
                              a_share_tool=self.a_share,
                              journal_path=self.root / "delivery.json")

    async def test_source_delivery_retry_read_and_b_post_ids(self):
        updates = [
            {"id": "a1", "writes": [{"file_path": CODE, "content": "version one\n"}],
             "post": {"title": "A plan one", "summary": "See gradient_masking/plan.py", "finding_type": "insight"}},
            {"id": "a2", "writes": [{"file_path": CODE, "content": "version two\n"}],
             "post": {"title": "A plan two", "summary": "Revised plan", "finding_type": "insight"}},
        ]
        adapter = self.adapter(updates)
        first = await adapter.dispatch("call-1", "LS", {"path": FORUM}, self.b_ls)
        self.assertIn("0003_agent_a_A_plan_one.json", first)
        self.assertEqual((self.code_root / "ideas/gradient_masking/plan.py").read_text(), "version one\n")
        self.assertEqual(await adapter.dispatch("call-1", "LS", {"path": FORUM}, self.b_ls), first)
        self.assertEqual(adapter.snapshot()["cursor"], 1)
        read = await adapter.dispatch("call-read", "Read", {"file_path": f"{FORUM}/0003_agent_a_A_plan_one.json"}, self.b_read)
        self.assertEqual(json.loads(read)["id"], 3)
        self.assertEqual(adapter.snapshot()["cursor"], 1)
        await self.b_share(title="B's own note", summary="independent", finding_type="observation")
        second = await adapter.dispatch("call-2", "Glob", {"pattern": f"{FORUM}/*.json"}, self.b_glob)
        self.assertIn("0005_agent_a_A_plan_two.json", second)
        self.assertEqual((self.code_root / "ideas/gradient_masking/plan.py").read_text(), "version two\n")
        self.assertEqual(await adapter.dispatch("call-3", "LS", {"path": FORUM}, self.b_ls),
                         await self.b_ls(path=FORUM))
        self.assertEqual(adapter.snapshot()["cursor"], 2)
        self.assertEqual([row["post_id"] for row in adapter.snapshot()["deliveries"]], [3, 5])
        self.assertEqual(len(list(self.forum.glob("*.json"))), 5)

    async def test_failed_source_write_rolls_back_and_halts(self):
        @wraps(self.a_write)
        async def failing_write(*, file_path, content):
            if file_path.endswith("fail.py"):
                return "ERROR: forced source write failure"
            return await self.a_write(file_path=file_path, content=content)

        adapter = self.adapter([{"id": "broken", "writes": [
            {"file_path": CODE, "content": "transient\n"},
            {"file_path": "w2s_research/ideas/gradient_masking/fail.py", "content": "never\n"},
        ], "post": {"title": "should not publish", "summary": "blocked"}}], write_tool=failing_write)
        before_audit = len(self.env.audit_log)
        with self.assertRaises(DeliveryHalt):
            await adapter.dispatch("call-fail", "LS", {"path": FORUM}, self.b_ls)
        self.assertFalse((self.code_root / "ideas/gradient_masking/plan.py").exists())
        self.assertEqual(len(list(self.forum.glob("*.json"))), 2)
        self.assertEqual(self.env._next_post_id, 3)
        self.assertEqual(len(self.env.audit_log), before_audit)
        self.assertEqual(json.loads((self.root / "delivery.json").read_text())["status"], "failed")
        with self.assertRaises(DeliveryHalt):
            adapter.assert_healthy()

    async def test_non_forum_checks_do_not_advance_queue(self):
        adapter = self.adapter([{"id": "a1", "writes": [],
                                "post": {"title": "A note", "summary": "later"}}])
        await adapter.dispatch("root-ls", "LS", {"path": "."}, self.b_ls)
        await adapter.dispatch("read-seed", "Read", {"file_path": f"{FORUM}/0001_system_seed_one.json"}, self.b_read)
        await adapter.dispatch("glob-dir", "Glob", {"pattern": FORUM}, self.b_glob)
        await adapter.dispatch("glob-old", "Glob", {"pattern": f"{FORUM}/0001_system_seed_one.json"}, self.b_glob)
        await adapter.dispatch("glob-dirs", "Glob", {"pattern": f"{FORUM}/**"}, self.b_glob)
        with self.assertRaises(NotImplementedError):
            await adapter.dispatch("glob-absolute", "Glob", {"pattern": str(self.forum / "*.json")}, self.b_glob)
        self.assertEqual(adapter.snapshot()["cursor"], 0)
        self.assertEqual(len(list(self.forum.glob("*.json"))), 2)

    async def test_same_call_retry_is_serialized_and_source_runs_once(self):
        adapter = self.adapter([{"id": "a1", "writes": [],
                                "post": {"title": "A note", "summary": "published"}}])
        calls = 0

        @wraps(self.b_ls)
        async def counted_ls(*, path="."):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return await self.b_ls(path=path)

        first, retry = await asyncio.gather(
            adapter.dispatch("same-call", "LS", {"path": FORUM}, counted_ls),
            adapter.dispatch("same-call", "LS", {"path": FORUM}, counted_ls),
        )
        self.assertEqual(first, retry)
        self.assertEqual(calls, 1)
        self.assertEqual(adapter.snapshot()["cursor"], 1)
        self.assertEqual(len(list(self.forum.glob("*.json"))), 3)

    async def test_b_code_edit_blocks_overwrite_and_halts(self):
        code_path = self.code_root / "ideas/gradient_masking/plan.py"
        code_path.parent.mkdir(parents=True)
        code_path.write_text("initial\n")
        adapter = self.adapter([{"id": "a1", "writes": [
            {"file_path": CODE, "content": "A replacement\n"}],
            "post": {"title": "A note", "summary": "would replace B"}}])
        code_path.write_text("B edit\n")
        with self.assertRaisesRegex(DeliveryHalt, "shared code changed"):
            await adapter.dispatch("collision", "LS", {"path": FORUM}, self.b_ls)
        self.assertEqual(code_path.read_text(), "B edit\n")
        self.assertEqual(len(list(self.forum.glob("*.json"))), 2)
        self.assertEqual(json.loads((self.root / "delivery.json").read_text())["status"], "failed")

    async def test_manifest_hash_gate(self):
        manifest_dir = self.root / "stimulus"
        manifest_dir.mkdir()
        (manifest_dir / "code.py").write_text("original\n")
        manifest = {"schema_version": 1, "experiment": "E130", "updates": [{
            "update_id": "a01", "writes": [{"file_path": CODE,
                "content_file": "code.py", "sha256": "01" * 32}],
            "post": {"title": "A note", "summary": "content"}}]}
        path = manifest_dir / "manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_manifest(path)
        manifest["updates"][0]["writes"][0]["sha256"] = hashlib.sha256(b"original\n").hexdigest()
        path.write_text(json.dumps(manifest))
        (manifest_dir / "code.py").unlink()
        with self.assertRaises(FileNotFoundError):
            load_manifest(path)

    async def test_listing_without_new_post_rolls_back(self):
        adapter = self.adapter([{"id": "a1", "writes": [],
                                "post": {"title": "A note", "summary": "published"}}])

        @wraps(self.b_ls)
        async def stale_ls(*, path="."):
            return "w2s_research/research_loop/shared_findings/0001_system_seed_one.json"

        with self.assertRaisesRegex(DeliveryHalt, "did not expose"):
            await adapter.dispatch("stale-list", "LS", {"path": FORUM}, stale_ls)
        self.assertEqual(adapter.snapshot()["cursor"], 0)
        self.assertEqual(len(list(self.forum.glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
