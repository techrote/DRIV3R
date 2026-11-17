#!/usr/bin/env python3
"""Pinned official-Q4, real System-One GREYHOUND executed-layer ablation.

No synthetic policy stands in for inference. W0 is an external historical
legal-action host; every actual /v1/systemone choice is immediately applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import urllib.request

LLAMA_SHA = "8642541b255bfb5dc39cd351a178be09013a55d6"
Q4_SHA = "adca5789ad30ae6027e8e936749bb9f19673097492ab26fa1c5b69643ca50b4b"
Q4_BYTES = 1099629888
Q4_REPO_REV = "4a181094d1b6addfaa3a0956f78e7ab2aa406a0c"
W0_SHA_BLOB = "8979b6f0296e81051194bacc907992d38e006605"
STEPS = 24
FOLDS = ((7703, "pipework"), (8807, "greebles"))
BRANCH = "research/import-consolidated-worm-labs"
ORIGINAL_GREYHOUND = (0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 20, 22, 23)

# Exact nested GREYHOUND ancestry. The 8-step and 4-step candidates reuse
# frozen, previously authored flow IDs rather than rewriting historical modes.
PROGRESSION = {
    "greyhound": ORIGINAL_GREYHOUND,
    "grey10_cut": (0, 1, 2, 4, 6, 8, 10, 14, 20, 23),
    "grey8_cut": (0, 1, 2, 4, 8, 10, 20, 23),
    "grey6_cut": (0, 2, 4, 8, 10, 20),
    "aero4_no2": (0, 4, 10, 20),
    "grey3_cut": (0, 4, 10),
    "grey2_cut": (0, 10),
}
EXTRA_FLOWS = {
    name: layers for name, layers in PROGRESSION.items()
    if name in ("grey10_cut", "grey6_cut", "grey3_cut", "grey2_cut")
}
TIERS = {
    "upper": ("grey10_cut", "grey8_cut"),
    "middle": ("grey6_cut", "aero4_no2"),
    "extreme": ("grey3_cut", "grey2_cut"),
}
CONTROLS = ("stock", "greyhound")
LEGACY_FILES = {
    "experiments/driv3r/backend.py": (
        "67bc90e685f3311f73b6dbb3b0f6773d10e64623",
        "baad34f4aaa2bb3e14c05985b93c00922f1fe3f9"),
    "experiments/driv3r/pins.json": (
        "67bc90e685f3311f73b6dbb3b0f6773d10e64623",
        "cf4c7f9f164e6ad2d8ca368d610d65ef3d6e2767"),
    "experiments/agent_painter/process.py": (
        "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61",
        "8e0f0b42e513e10102924b42f6dddedb5cc36930"),
    "experiments/agent_painter/worm/common/worm.py": (
        "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61",
        W0_SHA_BLOB),
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for part in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def validate_lineage():
    from experiments.driv3r.frankenlayers.patch_runtime import FLOWS as base
    from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
    validate_flows()
    assert base["greyhound"] == ORIGINAL_GREYHOUND
    assert LAB_FLOWS["grey8_cut"] == PROGRESSION["grey8_cut"]
    assert LAB_FLOWS["aero4_no2"] == PROGRESSION["aero4_no2"]
    lengths = [13, 10, 8, 6, 4, 3, 2]
    assert list(map(len, PROGRESSION.values())) == lengths
    previous = None
    for name, layers in PROGRESSION.items():
        assert len(layers) == len(set(layers)) and layers[0] == 0
        assert all(0 <= i < 24 for i in layers)
        assert 10 in layers and list(layers) == sorted(layers)
        if previous is not None:
            assert set(layers) < set(previous), "Not a nested deletion"
        previous = layers
    assert not set(EXTRA_FLOWS) & (set(base) | set(LAB_FLOWS))
    return True


def validate_request(req):
    if type(req) is not dict or req.get("schema") != "driv3r/greyhound-q4-request/1":
        raise ValueError("Wrong experiment request schema")
    if req.get("tier") not in TIERS:
        raise ValueError("Unsupported progression tier")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{3,79}", str(req.get("nonce", ""))):
        raise ValueError("Invalid unique request nonce")
    if req.get("steps_per_episode") != STEPS or req.get("folds") != [list(x) for x in FOLDS]:
        raise ValueError("Frozen host budget/folds changed")
    if req.get("q4_sha256") != Q4_SHA or req.get("q4_bytes") != Q4_BYTES:
        raise ValueError("Q4 checkpoint identity changed")
    if req.get("llama_commit") != LLAMA_SHA or req.get("branch") != BRANCH:
        raise ValueError("Wrong runtime or branch")
    if req.get("max_parallel") != 1 or req.get("cpu_only") is not True:
        raise ValueError("CPU concurrency/budget mismatch")
    if any(req.get(k) is not False for k in ("paid_gpu", "paid_api", "merge", "ready")):
        raise ValueError("Unsafe experiment request")
    if req.get("native_game") is not False or req.get("automatic_on_routine_push") is not False:
        raise ValueError("Wrong experiment scope")
    return req


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _download_bounded(url, target, expected_bytes, expected_sha):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_bytes and sha256_file(target) == expected_sha:
        return "verified_cache"
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    h = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=120) as response, part.open("xb") as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise RuntimeError("Download exceeded pinned model bytes")
                out.write(chunk)
                h.update(chunk)
        if total != expected_bytes or h.hexdigest() != expected_sha:
            raise RuntimeError("Model Q4 SHA256/length mismatch")
        part.rename(target)
    finally:
        if part.exists():
            part.unlink()
    return "downloaded_verified"


def bootstrap_legacy(root, evidence):
    root, evidence = Path(root), Path(evidence)
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for path, (commit, expected_blob) in LEGACY_FILES.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/techrote/cybersand/{commit}/{path}"
        if destination.exists() and git_blob_sha(destination.read_bytes()) == expected_blob:
            data = destination.read_bytes()
            state = "verified_cache"
        else:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(200_001)
            if len(data) > 200_000:
                raise ValueError(f"Oversized legacy module: {path}")
            if git_blob_sha(data) != expected_blob:
                raise ValueError(f"Wrong historical Git blob: {path}")
            destination.write_bytes(data)
            state = "downloaded_verified"
        receipts[path] = {"source_commit": commit, "git_blob_sha1": expected_blob,
                          "sha256": hashlib.sha256(data).hexdigest(), "state": state}
    write_json(evidence / "legacy-source-receipts.json", receipts)
    return root / "experiments/agent_painter/worm/common/worm.py"


def acquire_q4(root, evidence):
    root, evidence = Path(root), Path(evidence)
    url = ("https://huggingface.co/LiquidAI/d1-2B-GGUF/resolve/"
           + Q4_REPO_REV + "/d1-2B-Q4_K_M.gguf")
    target = root / "d1-2B-Q4_K_M.gguf"
    try:
        state = _download_bounded(url, target, Q4_BYTES, Q4_SHA)
    except Exception as exc:
        write_json(evidence / "q4-acquisition-failure.json",
                   {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                    "expected_sha256": Q4_SHA, "expected_bytes": Q4_BYTES, "source": url})
        raise
    receipt = {"schema": "driv3r/published-q4/1", "repository": "LiquidAI/d1-2B-GGUF",
               "revision": Q4_REPO_REV, "filename": target.name,
               "format": "official Q4_K_M, NOT Q8-requantized",
               "expected_bytes": Q4_BYTES, "sha256": sha256_file(target),
               "source_url": url, "state": state, "license": "LFM Open License v1.0"}
    write_json(evidence / "published-q4-receipt.json", receipt)
    return target


def patch_compiled_graph(llama_root, evidence):
    from experiments.driv3r.frankenlayers_lab.patch_lab import patch as lab_patch, UNKNOWN, TARGET
    validate_lineage()
    base = lab_patch(Path(llama_root))
    target = Path(llama_root) / TARGET
    old = target.read_text()
    if old.count(UNKNOWN) != 1:
        raise RuntimeError("Unknown-case patch anchor changed: refuse fuzzy patch")
    extras = "".join(
        '        } else if (mode == "' + name + '") {\n'
        '            d1_flow = {' + ",".join(map(str, layers)) + '};\n'
        for name, layers in EXTRA_FLOWS.items())
    target.write_text(old.replace(UNKNOWN, extras + UNKNOWN))
    final = target.read_bytes()
    receipt = {"schema": "driv3r/q4-greyhound-graph-patch/1",
               "base_patch": base, "llama_commit": LLAMA_SHA,
               "patched_lfm3_sha256": hashlib.sha256(final).hexdigest(),
               "extra_flows": {k: list(v) for k, v in EXTRA_FLOWS.items()},
               "full_lineage": {k: list(v) for k, v in PROGRESSION.items()},
               "decision_graph_only": True, "base_model_weights_modified": False,
               "stock_completion_graph_unchanged": True}
    write_json(Path(evidence) / "graph-patch-receipt.json", receipt)
    return receipt


def _add_legacy_to_import_paths(root):
    import experiments
    import experiments.driv3r
    import experiments.agent_painter
    root = Path(root).resolve()
    for module, directory in ((experiments, "experiments"),
                              (experiments.driv3r, "experiments/driv3r"),
                              (experiments.agent_painter, "experiments/agent_painter")):
        p = str(root / directory)
        if p not in module.__path__:
            module.__path__.append(p)


def source_guard(commit):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != commit:
        raise RuntimeError("Checked out commit does not equal stated exact-head source")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--",
         "experiments/driv3r/frankenlayers_lab", ".github/workflows/driv3r-q4-greyhound.yml"],
        text=True).strip()
    if dirty:
        raise RuntimeError("Research source not clean; refusing attribution")


def evaluate(binary, model, w0, legacy_root, patch_receipt, request, evidence, commit):
    from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
    from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS
    from experiments.driv3r.frankenlayers_lab import wave20_structural as summary
    from experiments.driv3r.frankenlayers import infer as paired
    validate_lineage()
    if sha256_file(model) != Q4_SHA or Path(model).stat().st_size != Q4_BYTES:
        raise ValueError("Published Q4 model identity mismatch")
    receipt = json.loads(Path(patch_receipt).read_text())
    if (receipt["llama_commit"] != LLAMA_SHA
        or receipt["extra_flows"] != {k:list(v) for k,v in EXTRA_FLOWS.items()}
        or receipt["base_patch"]["lab_flows"] != {k:list(v) for k,v in LAB_FLOWS.items()}):
        raise ValueError("Compiled runtime patch receipt mismatch")
    if any(name in FLOWS for name in LAB_FLOWS) or any(name in FLOWS for name in EXTRA_FLOWS):
        raise RuntimeError("Historical mode registry collision")
    FLOWS.update(LAB_FLOWS)
    FLOWS.update(EXTRA_FLOWS)
    host = paired.load_w0(w0)
    tier = request["tier"]
    modes = CONTROLS + TIERS[tier]
    evidence = Path(evidence)
    header = {"schema": "driv3r/greyhound-q4-real-inference/1",
              "source_commit": commit, "tier": tier, "nonce": request["nonce"],
              "official_q4_sha256": Q4_SHA, "q4_bytes": Q4_BYTES,
              "binary_sha256": sha256_file(binary), "w0_sha256": sha256_file(w0),
              "w0_git_blob_sha1": W0_SHA_BLOB, "patch_receipt_sha256": sha256_file(patch_receipt),
              "model_quant": "vendor Q4_K_M; NOT Q8 derived",
              "synthetic_host": True, "native_paint": False,
              "no_smoothing": True, "post_action_endpoint_unknown": True,
              "same_initial_host_seed": True, "later_states_diverge": True,
              "tiers": {k:list(v) for k,v in TIERS.items()},
              "source_flows": {k:list(PROGRESSION[k]) for k in modes if k in PROGRESSION},
              "folds": [list(f) for f in FOLDS], "steps_per_episode": STEPS,
              "maximum_model_decisions": len(FOLDS)*len(modes)*STEPS}
    write_json(evidence/"provenance.json", header)
    ledger = []
    with summary.rss_instrument(paired):
        for seed, family in FOLDS:
            old = paired.SEED, paired.STEPS, paired.FAMILIES
            paired.SEED, paired.STEPS, paired.FAMILIES = seed, STEPS, (family,)
            stock_p50 = None
            try:
                for mode in modes:
                    out = evidence / f"{seed}-{family}" / mode
                    entry = {"seed": seed, "family": family, "mode": mode,
                             "executed_source_layers": list(FLOWS[mode]),
                             "skipped_source_blocks": 24-len(set(FLOWS[mode])),
                             "status": "FAILED"}
                    try:
                        report = paired.run_arm(binary, model, host, mode, out)
                        decisions = json.loads((out/f"{mode}-{family}-choices.json").read_text())
                        measures = summary.trace_metrics(decisions, report, paired)
                        entry["metrics"] = measures
                        entry["status"] = "COMPLETED"
                        entry["nonterminal_budget"] = (
                            measures["decisions"] == STEPS
                            and measures["first_STOP_tick"] is None)
                        svg = summary.svg_from_calls(decisions, f"{tier} {mode} {seed}/{family}")
                        if svg:
                            (out/"observed-path.svg").write_text(svg)
                        write_json(out/"observed-path.json", [
                            {"tick": i, "position": [c["observation"]["x"], c["observation"]["y"]],
                             "legal": c["observation"]["legal"],
                             "raw_choice": c["parsed"]["candidate_id"]}
                            for i,c in enumerate(decisions)])
                        mem = out/"process"/"rss.json"
                        entry["sampled_post_ready_rss"] = json.loads(mem.read_text()) if mem.exists() else None
                        if mode == "stock":
                            stock_p50 = measures.get("p50_decision_seconds")
                        if stock_p50 and measures.get("p50_decision_seconds"):
                            entry["same_runner_stock_speed_ratio"] = (
                                stock_p50 / measures["p50_decision_seconds"])
                    except Exception:
                        entry["error"] = traceback.format_exc()
                    out.mkdir(parents=True, exist_ok=True)
                    write_json(out/"result.json", entry)
                    ledger.append(entry)
                    write_json(evidence/"catalogue.json", {
                        "schema": "driv3r/greyhound-q4-candidate-gallery/1",
                        "source_commit": commit, "tier": tier, "cases": ledger,
                        "artistic_niches": ["Meanderer","Curler","Brancher","Chaotic Artist"],
                        "niche_assessments": "UNRATED: do not infer personalities from scalar statistics"})
            finally:
                paired.SEED, paired.STEPS, paired.FAMILIES = old
    status = "SOURCE_AND_Q4_VERIFIED"
    missing_controls = [x for x in ledger if x["mode"] in CONTROLS and x["status"] != "COMPLETED"]
    any_mutant_completed = any(x["mode"] in TIERS[tier] and x["status"] == "COMPLETED" for x in ledger)
    if missing_controls or not any_mutant_completed:
        status = "INFRASTRUCTURE_OR_MODEL_FAILURE"
    elif any(x["status"] == "FAILED" for x in ledger):
        status = "PARTIAL_WITH_FAILURES"
    else:
        status = "COMPLETE_RESEARCH_ONLY"
    write_json(evidence/"result-summary.json", {
        "status": status, "cases": len(ledger),
        "completed": sum(x["status"] == "COMPLETED" for x in ledger),
        "failed": sum(x["status"] == "FAILED" for x in ledger),
        "nonterminal_candidates": [f'{x["mode"]}:{x["seed"]}:{x["family"]}'
             for x in ledger if x.get("nonterminal_budget") and x["mode"] in TIERS[tier]],
        "niche_rankings": "NOT YET OWNER REVIEWED",
        "progression_gate": ("REVIEW_NEGATIVE_AND_POSITIVE_ARTIFACTS_BEFORE_NEXT_TIER"),
        "native_qualified": False})
    print(json.dumps({"status":status,"cases":len(ledger)},sort_keys=True),flush=True)
    if status == "INFRASTRUCTURE_OR_MODEL_FAILURE":
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--evidence", type=Path, required=True)
    m = sub.add_parser("model")
    m.add_argument("--root", type=Path, required=True)
    m.add_argument("--evidence", type=Path, required=True)
    k = sub.add_parser("patch")
    k.add_argument("--llama-root", type=Path, required=True)
    k.add_argument("--evidence", type=Path, required=True)
    r = sub.add_parser("run")
    for arg in ("binary", "model", "w0", "legacy-root", "patch-receipt", "request", "evidence"):
        r.add_argument("--"+arg, type=Path, required=True)
    r.add_argument("--source-commit", required=True)
    opts = p.parse_args()
    if opts.command == "bootstrap":
        bootstrap_legacy(opts.root, opts.evidence)
    elif opts.command == "model":
        acquire_q4(opts.root, opts.evidence)
    elif opts.command == "patch":
        patch_compiled_graph(opts.llama_root, opts.evidence)
    else:
        source_guard(opts.source_commit)
        req = validate_request(json.loads(opts.request.read_text()))
        _add_legacy_to_import_paths(opts.legacy_root)
        evaluate(opts.binary, opts.model, opts.w0, opts.legacy_root, opts.patch_receipt,
                 req, opts.evidence, opts.source_commit)

if __name__ == "__main__":
    main()
