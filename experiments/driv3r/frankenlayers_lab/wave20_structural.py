#!/usr/bin/env python3
"""Manual-only Wave 20 Q8 whole-layer deletion in a real compiled System One graph.

Requires pinned, separately available CyberSand historical Python runtime and W0;
DRIV3R import alone does not include those dependencies. No controller/adapters.
All success claims derive only from actual /v1/systemone decisions applied by W0.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import traceback
from xml.sax.saxutils import escape

FOLDS = ((7703, "pipework"), (8807, "greebles"))
STEPS = 24
CONTROLS = ("stock", "greyhound", "branchwitch", "aero4_no2")
CANDIDATES = ("grey8_cut", "witch8_cut", "route6_cut", "tail6_cut",
              "curl4_cut", "branch4_cut")
MODES = CONTROLS + CANDIDATES
Q8_SHA256 = "073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c"
PINNED_LLAMA = "8642541b255bfb5dc39cd351a178be09013a55d6"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkout_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def source_guard(expected):
    if checkout_sha() != expected:
        raise ValueError("Source HEAD does not equal declared experiment SHA")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--",
                                     "experiments/driv3r/frankenlayers_lab"], text=True).strip()
    if dirty:
        raise ValueError("Laboratory source dirty: cannot assign exact-source provenance")


def validation_receipt(receipt, flows):
    if receipt.get("llama_source_commit") != PINNED_LLAMA:
        raise ValueError("Different llama.cpp source revision")
    if receipt.get("decision_only") is not True or receipt.get("weights_modified") is not False:
        raise ValueError("Not the frozen donor decision-only patch")
    for mode in CANDIDATES + ("aero4_no2",):
        if receipt.get("lab_flows", {}).get(mode) != list(flows[mode]):
            raise ValueError("Missing or altered compiled flow mapping: " + mode)


def _rss_kib(pid):
    try:
        line = next(x for x in Path(f"/proc/{pid}/status").read_text().splitlines()
                    if x.startswith("VmRSS:"))
        return int(line.split()[1])
    except (OSError, StopIteration, ValueError):
        return None


@contextmanager
def rss_instrument(paired):
    base = paired.Server
    class Metered(base):
        def __enter__(self):
            result = super().__enter__()
            self._stop = threading.Event()
            self._rss_samples = []
            def poll():
                while not self._stop.wait(0.10):
                    sample = _rss_kib(self.proc.pid)
                    if sample is not None:
                        self._rss_samples.append(sample)
            first = _rss_kib(self.proc.pid)
            if first is not None:
                self._rss_samples.append(first)
            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
            return result

        def __exit__(self, *args):
            if hasattr(self, "_stop"):
                self._stop.set()
                self._thread.join(timeout=2)
                if self.proc is not None:
                    last = _rss_kib(self.proc.pid)
                    if last is not None:
                        self._rss_samples.append(last)
                (self.root / "rss.json").write_text(json.dumps({
                    "peak_post_ready_rss_kib": max(self._rss_samples, default=None),
                    "observations": len(self._rss_samples),
                    "sampling_interval_seconds": 0.1,
                    "caveat": "Post-readiness sampling; process startup peak not measured"
                }, indent=2) + "\n")
            return super().__exit__(*args)
    paired.Server = Metered
    try:
        yield
    finally:
        paired.Server = base


def trace_metrics(calls, arm, paired):
    ok = [x for x in calls if x.get("status") == "ok"]
    xy = [(int(x["observation"]["x"]), int(x["observation"]["y"])) for x in ok]
    raw = [x["parsed"]["candidate_id"] for x in ok]
    legal_count = sum(x["parsed"]["candidate_id"] in x["observation"]["legal"] for x in ok)
    lat = [float(x["latency_seconds"]) for x in ok]
    output_rate = []
    for x in ok:
        usage = x.get("raw_response", {}).get("usage", {})
        tokens = usage.get("output_tokens")
        if isinstance(tokens, (int, float)) and tokens >= 0 and x["latency_seconds"] > 0:
            output_rate.append(tokens / x["latency_seconds"])
    d2 = sum(xy[i] == xy[i-2] and xy[i] != xy[i-1] for i in range(2, len(xy)))
    edges = []
    for i in range(len(xy)-1):
        if raw[i] not in ("RETRACE", "BRANCH") and xy[i] != xy[i+1]:
            edges.append(tuple(sorted((xy[i], xy[i+1]))))
    straight, longest = 0, 0
    for a in raw:
        straight = straight + 1 if a == "STRAIGHT" else 0
        longest = max(longest, straight)
    total = len(ok)
    entropy = -sum((n/total)*math.log2(n/total) for n in Counter(raw).values()) if total else None
    return {
        "decisions": total, "budget": STEPS,
        "host_legal_raw_choices": legal_count, "raw_choice_legality": legal_count/total if total else None,
        "choice_counts": dict(Counter(raw)), "first_STOP_tick": raw.index("STOP") if "STOP" in raw else None,
        "p50_decision_seconds": statistics.median(lat) if lat else None,
        "p95_decision_seconds": paired.percentile(lat, .95),
        "median_end_to_end_output_tokens_per_second": statistics.median(output_rate) if output_rate else None,
        "output_rate_caveat": "Output tokens / wall-call seconds including prefill; null if usage absent",
        "input_token_rate_median": statistics.median(
            [x["input_tokens_per_second"] for x in ok if x.get("input_tokens_per_second")]
        ) if any(x.get("input_tokens_per_second") for x in ok) else None,
        "turn_choice_fraction": sum(a in ("LEFT", "RIGHT", "BRANCH") for a in raw)/total if total else None,
        "longest_STRAIGHT_choice_run": longest, "two_step_position_oscillations": d2,
        "repeated_observed_edges_excluding_RETRACE_BRANCH": len(edges)-len(set(edges)),
        "repeat_predecision_positions": len(xy)-len(set(xy)),
        "branch_choice_count": raw.count("BRANCH"), "retrace_choice_count": raw.count("RETRACE"),
        "observed_unique_positions": len(set(xy)),
        "observed_net_displacement": math.dist(xy[0], xy[-1]) if len(xy)>1 else None,
        "position_caveat": "Predecision host observations; final post-action endpoint omitted",
        "host_episode_metrics": arm["experiments"][0]["metrics"]
    }


def svg_from_calls(calls, label):
    points = [(int(c["observation"]["x"]), int(c["observation"]["y"]), c["choice"])
              for c in calls if c.get("status") == "ok"]
    if not points:
        return None
    xmin = min(p[0] for p in points)-2
    xmax = max(p[0] for p in points)+2
    ymin = min(p[1] for p in points)-2
    ymax = max(p[1] for p in points)+2
    groups, group = [], []
    for i, (x,y,a) in enumerate(points):
        if i and points[i-1][2] == "RETRACE" and group:
            groups.append(group)
            group = []
        group.append((x,y))
    if group: groups.append(group)
    segments = "".join('<polyline points="'+" ".join(f"{x},{y}" for x,y in g)
                       + '" fill="none" stroke="#156c89" stroke-width="0.25" '
                       + 'stroke-linecap="round" stroke-linejoin="round"/>'
                       for g in groups if len(g)>1)
    dots = "".join(f'<circle cx="{x}" cy="{y}" r=".16" fill="#c14f36"/>'
                   for x,y,_ in points)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="'
            + f'{xmin} {ymin} {max(4,xmax-xmin)} {max(4,ymax-ymin)}'
            + '"><title>'+escape(label)
            + ' - unsmoothed predecision coordinates, RETRACE breaks</title>'
            + '<rect x="'+str(xmin)+'" y="'+str(ymin)+'" width="'+str(max(4,xmax-xmin))
            + '" height="'+str(max(4,ymax-ymin))+'" fill="#f8f6f1"/>'
            + segments + dots + '</svg>\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("binary", "model-q8", "w0", "patch-receipt", "legacy-root", "output"):
        p.add_argument("--"+arg, required=True, type=Path)
    p.add_argument("--source-sha", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Immutable output destination already exists")
    if len(args.source_sha) != 40:
        raise ValueError("Expected exact 40-character Git commit SHA")
    source_guard(args.source_sha)
    if sha256(args.model_q8) != Q8_SHA256:
        raise ValueError("Frozen Q8 donor hash mismatch")
    if not args.legacy_root.is_dir():
        raise ValueError("Read-only CyberSand historical runtime checkout required")
    sys.path.append(str(args.legacy_root.resolve()))
    try:
        from experiments.driv3r.frankenlayers import infer as paired
        from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
        from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS, validate_flows
    except ImportError as exc:
        raise RuntimeError("Standalone DRIV3R import omits backend/process dependencies; "
                           "provide pinned external read-only CyberSand Python runtime") from exc
    validate_flows()
    if any(m in FLOWS for m in LAB_FLOWS):
        raise ValueError("Model registry collision")
    receipt = json.loads(args.patch_receipt.read_text())
    validation_receipt(receipt, LAB_FLOWS)
    FLOWS.update(LAB_FLOWS)
    w0 = paired.load_w0(args.w0)
    provenance = {
        "schema": "frankenlayers-wave20-live-structural/1",
        "source_sha": args.source_sha, "immutable_q8_sha256": Q8_SHA256,
        "binary_sha256": sha256(args.binary), "w0_sha256": sha256(args.w0),
        "patch_receipt_sha256": sha256(args.patch_receipt),
        "llama_source_commit": PINNED_LLAMA, "synthetic_host_only": True,
        "native_game_qualified": False, "model_weights_changed": False,
        "executed_graph_changed": True, "controls": CONTROLS,
        "candidate_flows": {m: list(LAB_FLOWS[m]) for m in CANDIDATES},
        "folds": FOLDS, "steps_per_episode": STEPS,
        "maximum_attempted_model_decisions": len(MODES)*len(FOLDS)*STEPS,
        "cpu_only": True, "no_posthoc_action_logic": True,
        "rss_caveat": "Sampled post-ready process RSS; not full launch peak"}
    args.output.mkdir(parents=True)
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    ledger = []
    with rss_instrument(paired):
        for seed, family in FOLDS:
            old = paired.SEED, paired.STEPS, paired.FAMILIES
            paired.SEED, paired.STEPS, paired.FAMILIES = seed, STEPS, (family,)
            try:
                for mode in MODES:
                    out = args.output / f"{seed}-{family}" / mode
                    entry = {"seed": seed, "family": family, "mode": mode,
                             "executed_layers": list(FLOWS[mode]),
                             "skipped_entire_source_layers": 24-len(set(FLOWS[mode])),
                             "status": "failure"}
                    try:
                        arm = paired.run_arm(args.binary, args.model_q8, w0, mode, out)
                        choices_file = out / f"{mode}-{family}-choices.json"
                        calls = json.loads(choices_file.read_text())
                        if len(calls) != arm["model_decisions"]:
                            raise AssertionError("Decision evidence mismatch")
                        entry["measurements"] = trace_metrics(calls, arm, paired)
                        mem = out/"process"/"rss.json"
                        entry["post_ready_rss"] = json.loads(mem.read_text()) if mem.exists() else None
                        svg = svg_from_calls(calls, f"{mode} {seed} {family}")
                        if svg: (out/"raw-predecision-path.svg").write_text(svg)
                        (out/"raw-predecision-path.json").write_text(json.dumps([
                            {"tick": i, "x": c["observation"]["x"], "y": c["observation"]["y"],
                             "raw_choice": c["parsed"]["candidate_id"],
                             "legal": c["observation"]["legal"]}
                            for i,c in enumerate(calls)], indent=2)+"\n")
                        entry["status"] = "completed"
                    except Exception:
                        entry["failure_traceback"] = traceback.format_exc()
                    out.mkdir(parents=True, exist_ok=True)
                    (out/"wave20-case.json").write_text(json.dumps(entry,indent=2,sort_keys=True)+"\n")
                    ledger.append(entry)
                    (args.output/"catalogue.json").write_text(json.dumps({
                        "schema": "frankenlayers-wave20-candidate-catalogue/1",
                        "niche_assessments": "UNREVIEWED - never label from metrics alone",
                        "cases": ledger},indent=2,sort_keys=True)+"\n")
            finally:
                paired.SEED, paired.STEPS, paired.FAMILIES = old
    print(json.dumps({"completed": sum(x["status"]=="completed" for x in ledger),
                      "failed": sum(x["status"]=="failure" for x in ledger),
                      "cases":len(ledger), "output":str(args.output)},sort_keys=True))
    if any(x["status"]=="failure" for x in ledger):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
