#!/usr/bin/env python3
"""GP350-016 independent GP350-015 original-receipt audit.

Requires only Python stdlib and the immutable Actions artifact ZIP.
NO additional LFM forward pass, optimizer update, or causal W0 rollout.
The score-penalty comparison is deliberately OBSERVED-STATE-ONLY.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
SOURCE_SHA = "a09470a10f4dfc7cbdb83753ea3d8eded533a444"
SOURCE_ZIP_SHA = "a4a83e6b1bfcbdb05e35b96e3adc9b218dafe59b0c6b76c416c51efd0162469f"
CHECKPOINTS = {
    18: "205f6356c25766a5f0664e5e4faf5a5135c22f092c55aa2d50f81293e230335f",
    36: "e37999047767861cfb7c05d64310d503505eeff3795c755d547b5e8aaa81cdf7",
    54: "dc302a1a610b5301ae94d4b8304cf9ae4292cfbd9a209dbb98db3252bf4c9d64",
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def trapped(obs):
    vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    return "RETRACE" in obs["legal"] and not any(
        action in obs["legal"] and
        obs["local_5x5"][2 + vectors[(obs["heading"] + offset) % 4][1]]
                         [2 + vectors[(obs["heading"] + offset) % 4][0]] == "."
        for action, offset in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1))
    )

def choose(obs, scores, penalty=0.0):
    return max(obs["legal"], key=lambda a:
        scores[ACTIONS.index(a)] - (penalty if a == "RETRACE" else 0.0))

def inspect(path):
    raw = path.read_bytes()
    assert sha(raw) == SOURCE_ZIP_SHA, "Not the original immutable GP350-015 archive"
    out = {
        "schema": "driv3r.lfm350.gp350016-original-receipt-audit/1",
        "source_run": 43943992509,
        "source_artifact": 14266902004,
        "source_sha": SOURCE_SHA,
        "original_zip_sha256": sha(raw),
        "new_model_forwards": 0,
        "new_optimizer_updates": 0,
        "counterfactuals_are_observed_state_only": True,
        "stages": {}
    }
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "Broken ZIP entry"
        out["original_zip_members_checked"] = len(archive.namelist())
        summary = json.loads(archive.read("original-experiment/summary.json"))
        assert summary["status"] == "complete"
        assert summary["source_sha"] == SOURCE_SHA
        assert summary["new_model_decisions"] == 2304
        assert summary["new_heldout_probes"] == 360
        for step, adapter_sha in CHECKPOINTS.items():
            name = "gp350014-step" + str(step)
            stage = summary["fresh_checkpoint_stages"][name]
            assert stage["adapter_sha256"] == adapter_sha
            matched = {}
            for kind, n in (("general", 72), ("recovery", 48)):
                p = json.loads(archive.read(
                    "original-experiment/" + name + "-" + kind + "-probe.json"))
                assert p["n"] == len(p["rows"]) == n
                assert p["teacher_matches"] == sum(bool(x["match"]) for x in p["rows"])
                matched[kind] = p["teacher_matches"]
                if kind == "recovery":
                    assert sum(x["target"] == "RETRACE" for x in p["rows"]) == 24
                    assert sum(x["target"] != "RETRACE" for x in p["rows"]) == 24
                    matched["trapped_retrace"] = sum(
                        x["target"] == x["legal"] == "RETRACE" for x in p["rows"])
                    matched["false_retrace"] = sum(
                        x["target"] != "RETRACE" and x["legal"] == "RETRACE"
                        for x in p["rows"])
            counts = Counter()
            by_seed = defaultdict(set)
            traps_by_seed = defaultdict(set)
            receipts = []
            for episode in stage["episodes"]:
                name_png = episode["png"]
                png = archive.read("original-experiment/" + name_png)
                assert png.startswith(b"\x89PNG\r\n\x1a\n")
                assert sha(png) == episode["png_sha256"]
                trace = json.loads(archive.read(
                    "original-experiment/" + name_png[:-4] + ".json"))
                assert len(trace["scores"]) == len(trace["record"]["events"]) == 64
                for decision, event in zip(trace["scores"], trace["record"]["events"]):
                    obs = decision["obs"]
                    scores = decision["six_logits"]
                    assert len(scores) == 6 and all(math.isfinite(x) for x in scores)
                    raw_choice = ACTIONS[max(range(6), key=lambda i: scores[i])]
                    legal_choice = choose(obs, scores)
                    assert raw_choice == decision["raw"]
                    assert legal_choice == decision["applied"] == event["applied"]
                    assert legal_choice == event["raw_choice"] and event["raw_legal"]
                    assert decision["raw_legal"] == (raw_choice in obs["legal"])
                    is_trapped = trapped(obs)
                    key = json.dumps(obs, sort_keys=True, separators=(",", ":"))
                    by_seed[episode["seed"]].add(key)
                    if is_trapped:
                        traps_by_seed[episode["seed"]].add(key)
                    counts["decisions"] += 1
                    counts["trapped"] += int(is_trapped)
                    counts["trapped_retrace"] += int(
                        is_trapped and legal_choice == "RETRACE")
                    counts["premature_retrace"] += int(
                        not is_trapped and legal_choice == "RETRACE")
                    counts["branches"] += int(legal_choice == "BRANCH")
                    counts["raw_illegal"] += int(raw_choice not in obs["legal"])
                    receipts.append((obs, scores, legal_choice, is_trapped))
            assert counts["decisions"] == 768
            reference = stage["on_policy"]["total"]
            assert (counts["trapped"], counts["trapped_retrace"],
                    counts["premature_retrace"], counts["branches"],
                    counts["raw_illegal"]) == (
                        reference["trapped"], reference["trapped_retraces"],
                        reference["retrace_with_open_arm"], reference["branches"],
                        reference["raw_illegal"])
            assert set(by_seed) == {9503, 9521}
            overlap = len(by_seed[9503] & by_seed[9521])
            unique_count = len(by_seed[9503] | by_seed[9521])
            assert unique_count + overlap == 768
            trap_overlap = len(traps_by_seed[9503] & traps_by_seed[9521])
            sweep = {}
            for penalty in (0, 0.5, 1):
                alternatives = [(choose(obs, scores, penalty), action, t)
                                for obs, scores, action, t in receipts]
                sweep[str(penalty)] = {
                    "trapped_retrace": sum(t and changed == "RETRACE"
                                           for changed, _, t in alternatives),
                    "premature_retrace": sum(not t and changed == "RETRACE"
                                             for changed, _, t in alternatives),
                    "changed_decisions": sum(changed != old
                                             for changed, old, _ in alternatives),
                }
            out["stages"]["step" + str(step)] = {
                "adapter_sha256": adapter_sha,
                "heldout": matched,
                "on_policy": dict(counts),
                "mean_unique_cells_original": stage["on_policy"]["mean_unique_cells"],
                "original_png_hashes_checked": len(stage["episodes"]),
                "unique_exact_observations": unique_count,
                "duplicate_observations_across_seeds": overlap,
                "unique_trapped_observations": len(
                    traps_by_seed[9503] | traps_by_seed[9521]),
                "trapped_observations_repeated_across_seeds": trap_overlap,
                "observed_state_penalty_sweep": sweep,
            }
    assert sum(x["original_png_hashes_checked"] for x in out["stages"].values()) == 36
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_zip", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = inspect(args.original_zip)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        assert not args.out.exists(), "Immutable evidence already exists"
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
