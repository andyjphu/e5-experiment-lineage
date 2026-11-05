"""Deliver one fixed Agent A code/post update before a real B forum check.

The caller keeps the released E5 model loop and tool definitions. It passes the
native B tool-call ID, name, arguments, and original source tool to dispatch().
No model is called here. The source .eval remains the run trace; this module
records only host-side delivery effects and failures.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable


class DeliveryHalt(Exception):
    """Stop the B session and preserve the attempt after an A delivery issue."""


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Persist a pending/committed receipt before returning to the model loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _source_tool_name(tool: Callable[..., Awaitable[str]]) -> str | None:
    info = getattr(tool, "__registry_info__", None)
    return getattr(info, "name", None)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and hash-check the frozen E130 stimulus before any workspace exists."""
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("experiment") != "E130":
        raise ValueError("expected E130 stimulus manifest schema 1")
    updates = []
    for entry in manifest["updates"]:
        writes = []
        for item in entry["writes"]:
            content_path = (path.parent / item["content_file"]).resolve(strict=True)
            if not content_path.is_relative_to(path.parent):
                raise ValueError(f"stimulus content escapes manifest directory: {content_path}")
            data = content_path.read_bytes()
            if _sha(data) != item["sha256"]:
                raise ValueError(f"stimulus hash mismatch: {content_path}")
            writes.append({"file_path": item["file_path"], "content": data.decode("utf-8")})
        updates.append({
            "id": entry["update_id"], "writes": writes,
            "post": entry["post"],
            "source_refs": entry.get("source_refs", []),
        })
    return updates


class FixedADelivery:
    """Apply prepared A updates through released E5 A tools at B forum checks.

    A fresh journal path is required. A prior journal, especially a pending one,
    must be reviewed with the saved workspace before any manual restart.
    """

    def __init__(
        self,
        *,
        env: Any,
        updates: list[dict[str, Any]],
        a_write_tool: Callable[..., Awaitable[str]],
        a_share_tool: Callable[..., Awaitable[str]],
        journal_path: Path,
    ) -> None:
        self.env = env
        self.updates = copy.deepcopy(updates)
        self.a_write_tool = a_write_tool
        self.a_share_tool = a_share_tool
        self.journal_path = Path(journal_path)
        self._lock = asyncio.Lock()
        self._cursor = 0
        self._seen: dict[str, tuple[str, str]] = {}
        self._deliveries: list[dict[str, Any]] = []
        self._failed: str | None = None
        self._expected_hashes: dict[Path, str | None] = {}
        self._validate()
        if self.journal_path.exists() or self.journal_path.is_symlink():
            raise ValueError(f"delivery journal already exists: {self.journal_path}")
        self._persist("ready")

    def _validate(self) -> None:
        if _source_tool_name(self.a_write_tool) != "Write":
            raise ValueError("expected released E5 A Write tool")
        if _source_tool_name(self.a_share_tool) != "share_finding":
            raise ValueError("expected released E5 A share_finding tool")
        if not self.env.agent_a_workspace or not self.env.agent_b_workspace or not self.env.forum_dir:
            raise ValueError("source E5 workspaces/forum not initialized")
        ids: set[str] = set()
        for update in self.updates:
            if not isinstance(update, dict) or not isinstance(update.get("id"), str) or not update["id"]:
                raise ValueError("each A update needs a stable id")
            if update["id"] in ids:
                raise ValueError(f"duplicate A update id: {update['id']}")
            ids.add(update["id"])
            if not isinstance(update.get("writes"), list) or not isinstance(update.get("post"), dict):
                raise ValueError(f"A update needs writes[] and post: {update['id']}")
            if not update["post"].get("title") or not update["post"].get("summary"):
                raise ValueError(f"A post needs title and summary: {update['id']}")
            for write in update["writes"]:
                if not isinstance(write, dict) or not isinstance(write.get("content"), str):
                    raise ValueError(f"invalid A code write: {update['id']}")
                path = self._shared_code_path(write.get("file_path"))
                if path not in self._expected_hashes:
                    self._expected_hashes[path] = _sha(path.read_bytes()) if path.exists() else None

    def _shared_code_path(self, file_path: Any) -> Path:
        """Allow only A's source-visible shared ideas/core tree."""
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("A code write needs file_path")
        a_root = Path(self.env.agent_a_workspace)
        candidate = Path(os.path.normpath(str(Path(file_path) if Path(file_path).is_absolute()
                                          else a_root / file_path)))
        try:
            relative = candidate.relative_to(a_root)
        except ValueError as exc:
            raise ValueError(f"A write escapes workspace: {file_path}") from exc
        parts = relative.parts
        if len(parts) < 3 or parts[:2] not in {
            ("w2s_research", "ideas"), ("w2s_research", "core")
        }:
            raise ValueError(f"A write is not shared ideas/core code: {file_path}")
        resolved = candidate.resolve(strict=False)
        shared_root = (a_root / "w2s_research" / parts[1]).resolve(strict=True)
        if not resolved.is_relative_to(shared_root):
            raise ValueError(f"A write leaves shared code tree: {file_path}")
        return resolved

    def _is_forum_check(self, function: str, arguments: dict[str, Any]) -> bool:
        b_root = Path(self.env.agent_b_workspace)
        forum = Path(self.env.forum_dir).resolve(strict=True)
        if function == "LS":
            if set(arguments) - {"path"}:
                return False
            path = arguments.get("path", ".")
            if not isinstance(path, str) or not path:
                return False
            candidate = Path(path) if Path(path).is_absolute() else b_root / path
            normalized = Path(os.path.normpath(str(candidate)))
            if not normalized.is_relative_to(b_root):
                return False
            return normalized.resolve(strict=False) == forum
        if function == "Glob":
            if set(arguments) != {"pattern"}:
                return False
            pattern = arguments.get("pattern")
            if not isinstance(pattern, str) or Path(pattern).is_absolute():
                return False  # pinned source ws.glob rejects absolute patterns
            normalized = os.path.normpath(pattern)
            prefix = "w2s_research/research_loop/shared_findings"
            if not normalized.startswith(prefix + os.sep):
                return False
            return normalized[len(prefix) + 1:] in {"*", "*.json", "**/*", "**/*.json"}
        return False

    def _persist(self, status: str, **extra: Any) -> None:
        try:
            _save_state(self.journal_path, {
                "schema": "e130-fixed-a-delivery-v1",
                "status": status,
                "cursor": self._cursor,
                "updates_total": len(self.updates),
                "deliveries": self._deliveries,
                "seen_call_ids": sorted(self._seen),
                "failed_issue": self._failed,
                **extra,
            })
        except Exception as exc:
            self._failed = f"delivery journal write failed: {exc}"
            raise DeliveryHalt(self._failed) from exc

    def assert_healthy(self) -> None:
        if self._failed is not None:
            raise DeliveryHalt(self._failed)

    def snapshot(self) -> dict[str, Any]:
        return {"cursor": self._cursor, "updates_total": len(self.updates),
                "deliveries": copy.deepcopy(self._deliveries),
                "seen_call_ids": len(self._seen), "failed_issue": self._failed}

    async def dispatch(
        self,
        call_id: str,
        function: str,
        arguments: dict[str, Any],
        source_tool: Callable[..., Awaitable[str]],
    ) -> str:
        """Call the original B tool, with at most one coherent A update first."""
        if not isinstance(call_id, str) or not call_id or not isinstance(arguments, dict):
            raise ValueError("native tool call id and arguments required")
        if _source_tool_name(source_tool) != function:
            raise ValueError(f"source tool mismatch: {function}")
        signature = json.dumps({"function": function, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        async with self._lock:
            self.assert_healthy()
            if call_id in self._seen:
                prior_signature, prior_result = self._seen[call_id]
                if prior_signature != signature:
                    self._failed = f"native tool call ID reused with different arguments: {call_id}"
                    self._persist("failed")
                    raise DeliveryHalt(self._failed)
                return prior_result
            trigger = self._is_forum_check(function, arguments)
            if not trigger or self._cursor >= len(self.updates):
                result = await source_tool(**arguments)
                self._seen[call_id] = (signature, result)
                if trigger:
                    self._persist("ready", exhausted=True)
                return result

            update = self.updates[self._cursor]
            self._persist("pending", pending_call_id=call_id, pending_update_id=update["id"])
            try:
                delivery = await self._apply(update, call_id, function, arguments, source_tool)
            except Exception as exc:
                self._failed = f"A update {update['id']} failed at B call {call_id}: {exc}"
                self._persist("failed", pending_call_id=call_id, pending_update_id=update["id"])
                raise DeliveryHalt(self._failed) from exc
            self._cursor += 1
            self._deliveries.append(delivery)
            self._seen[call_id] = (signature, delivery["b_tool_result"])
            self._persist("ready")
            return delivery["b_tool_result"]

    async def dispatch_tool(self, tc: Any, source_tool: Callable[..., Awaitable[str]]) -> str:
        """Aero's source loop callback, preserving the native ToolCall identity."""
        return await self.dispatch(tc.id, tc.function, tc.arguments, source_tool)

    async def _apply(
        self,
        update: dict[str, Any],
        call_id: str,
        function: str,
        arguments: dict[str, Any],
        source_tool: Callable[..., Awaitable[str]],
    ) -> dict[str, Any]:
        env = self.env
        forum = Path(env.forum_dir)
        forum_before = set(forum.iterdir())
        next_post_id = env._next_post_id
        leaderboard = copy.deepcopy(env.leaderboard)
        audit_log = copy.deepcopy(env.audit_log)
        old_files: list[tuple[Path, bytes | None]] = []
        new_dirs: list[Path] = []
        written: list[dict[str, Any]] = []
        expected_hashes = dict(self._expected_hashes)
        try:
            for write in update["writes"]:
                path = self._shared_code_path(write["file_path"])
                actual_hash = _sha(path.read_bytes()) if path.exists() else None
                if actual_hash != expected_hashes[path]:
                    raise RuntimeError(f"shared code changed since prior baseline: {write['file_path']}")
                old_files.append((path, path.read_bytes() if path.exists() else None))
                parent = path.parent
                while not parent.exists():
                    new_dirs.append(parent)
                    parent = parent.parent
                result = await self.a_write_tool(**write)
                if not result.startswith("Wrote ") or not path.is_file() or path.read_text() != write["content"]:
                    raise RuntimeError(f"source Write failed: {write['file_path']}: {result}")
                written.append({"path": write["file_path"], "sha256": _sha(path.read_bytes()),
                                "source_result": result})
                expected_hashes[path] = written[-1]["sha256"]
            post_result = await self.a_share_tool(**update["post"])
            match = re.search(r"Finding posted to forum \(ID: (\d+)\)", post_result)
            if not match:
                raise RuntimeError(f"source share_finding failed: {post_result}")
            post_id = int(match.group(1))
            new_posts = set(forum.iterdir()) - forum_before
            if len(new_posts) != 1 or env._next_post_id != next_post_id + 1:
                raise RuntimeError("source forum publication changed unexpected files or ID")
            post_path = new_posts.pop()
            post = json.loads(post_path.read_text())
            if post.get("id") != post_id or post.get("author") != "agent_a":
                raise RuntimeError("source forum post identity mismatch")
            if post.get("title") != update["post"]["title"] or post.get("summary") != update["post"]["summary"]:
                raise RuntimeError("source forum post content mismatch")
            b_result = await source_tool(**arguments)
            if b_result.startswith("ERROR:"):
                raise RuntimeError(f"B forum check failed after A publication: {b_result}")
            if post_path.name not in b_result:
                raise RuntimeError("B forum check did not expose the new A post")
            self._expected_hashes = expected_hashes
            return {"update_id": update["id"], "trigger_call_id": call_id,
                    "trigger_function": function, "post_id": post_id,
                    "post_file": post_path.name, "post_sha256": _sha(post_path.read_bytes()),
                    "source_post_result": post_result, "code_writes": written,
                    "b_tool_result": b_result, "b_tool_result_sha256": _sha(b_result),
                    "source_refs": update.get("source_refs", [])}
        except Exception:
            # The source loop dispatches B tool calls sequentially. Restore the
            # pre-check world before propagating a fatal issue to the launcher.
            for path, old in reversed(old_files):
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(old)
            for directory in sorted(set(new_dirs), key=lambda item: len(item.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            for path in set(forum.iterdir()) - forum_before:
                path.unlink()
            env._next_post_id = next_post_id
            env.leaderboard = leaderboard
            env.audit_log = audit_log
            raise
