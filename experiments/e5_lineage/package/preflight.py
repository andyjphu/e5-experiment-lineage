"""Read-only provenance and served-route checks for E130 reconstruction attempts.

Run this on ADB with the pinned author venv. It never invokes a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e120_source_e5_sol.prepare import file_sha, tree_sha, verify


SOURCE_MANIFEST_SHA256 = "83465e93b1f985db89ac3112ee9faaefc03e1f4a221d4fd7640abffb501bf08e"
SOURCE_PREPARE_SHA256 = "f9090b6157244157403314236853d92867fc4c22726c17f989eabc0794312b83"
SOURCE_ROUTE_SHA256 = "a85b73b6f08914d4253a2498721a481790606a2aef79cf9591d194c6d4cc7c2c"
SOURCE_RUN_SHA256 = "76f83e7bffb58fa0cfc3edbf2671f7c2f1ae388080daadf81d840dbc7e9fe8f6"
INITIAL_W2S_TREE_SHA256 = "c3d3be8fee0ff7d28b082dc1a2cd239927bf6279ad4117e28a6c3623b643de25"
MODEL = "gpt-6-sol"
PROGRAM = {"cyber": "daybreak_blue"}


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def verify_source_baseline(author: Path, w2s_source: Path, source_package: Path) -> dict:
    """Match the proven 009 source wrapper and runtime without a provider call."""
    author = author.resolve(strict=True)
    w2s_source = w2s_source.resolve(strict=True)
    source_package = source_package.resolve(strict=True)
    hashes = {
        "discovery_manifest.json": SOURCE_MANIFEST_SHA256,
        "prepare.py": SOURCE_PREPARE_SHA256,
        "route.py": SOURCE_ROUTE_SHA256,
        "run.py": SOURCE_RUN_SHA256,
    }
    for name, expected in hashes.items():
        _require_equal(file_sha(source_package / name), expected, name)

    manifest = json.loads((source_package / "discovery_manifest.json").read_text())
    _require_equal(manifest["source_author_commit"], "2e70dcd171584217fc0cba45662ff7547b369619", "source commit")
    _require_equal(manifest["w2s_commit"], "79a0562fa1a2c246048ed7c009f3684907987b05", "W2S commit")
    _require_equal(manifest["scenario"], "discredit_and_demoralize", "scenario")
    _require_equal(manifest["capability"], "persuasion", "capability")
    _require_equal(manifest["instruction_level"], 1, "instruction level")
    _require_equal(manifest["cached_results"], True, "cached results")
    _require_equal(manifest["model_b"], "openai/gpt-6-sol", "B model")
    _require_equal(manifest["access_programs"], PROGRAM, "Daybreak program")
    provenance = verify(author, w2s_source, manifest)
    return {
        "status": "source_baseline_verified_no_model_call",
        "source_package": str(source_package),
        "source_hashes": hashes,
        "provenance": provenance,
        "scenario": manifest["scenario"],
        "capability": manifest["capability"],
        "instruction_level": manifest["instruction_level"],
        "cached_results": manifest["cached_results"],
        "model_b": manifest["model_b"],
        "access_programs": manifest["access_programs"],
    }


def verify_initial_w2s_tree(checkout: Path) -> str:
    """Check an extracted per-attempt tree before any A update is applied."""
    actual = tree_sha(checkout.resolve(strict=True))
    _require_equal(actual, INITIAL_W2S_TREE_SHA256, "initial W2S tree")
    return actual


def verify_served_wire(journal: Path) -> dict:
    """Verify every completed B response from the actual Daybreak wire journal."""
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Daybreak wire journal is empty")
    ids = set()
    for index, row in enumerate(rows, 1):
        _require_equal(row.get("actor"), "B", f"wire row {index} actor")
        _require_equal(row.get("request_model"), MODEL, f"wire row {index} request model")
        _require_equal(row.get("response_model"), MODEL, f"wire row {index} served model")
        _require_equal(row.get("request_program"), PROGRAM, f"wire row {index} request program")
        _require_equal(row.get("response_program"), PROGRAM, f"wire row {index} served program")
        _require_equal(row.get("response_status"), "completed", f"wire row {index} status")
        response_id = row.get("response_id")
        if not response_id or response_id in ids:
            raise ValueError(f"wire row {index} missing or duplicate response ID")
        ids.add(response_id)
    return {"status": "served_route_verified", "responses": len(rows), "journal_sha256": file_sha(journal)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--w2s-source", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_source_baseline(args.author, args.w2s_source, args.source_package), indent=2))


if __name__ == "__main__":
    main()
