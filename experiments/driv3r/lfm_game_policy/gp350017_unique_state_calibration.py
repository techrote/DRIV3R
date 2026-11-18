#!/usr/bin/env python3
"""GP350-017: deduplicated frozen-LFM score calibration, NOT LFM LoRA training.

Read-only retrospective study of genuine GP350-015 model scores.  Two 160-step
Adam fits optimize ONLY an external scalar RETRACE logit offset, with world-size
holdout and exact-observation grouping.  No fresh model forwards, no closed-loop
counterfactual rollouts, and no native CyberSand evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
STAGES = ("gp350014-step18", "gp350014-step36", "gp350014-step54")
ZIP_SHA256 = "a4a83e6b1bfcbdb05e35b96e3adc9b218dafe59b0c6b76c416c51efd0162469f"
EXPERIMENT_SHA = "a09470a10f4dfc7cbdb83753ea3d8eded533a444"
MODEL_REVISION = "9921cd4d82c0e3986887aa4b376f9446729ef8a3"
CHECKPOINTS = {
    "gp350014-step18": "205f6356c25766a5f0664e5e4faf5a5135c22f092c55aa2d50f81293e230335f",
    "gp350014-step36": "e37999047767861cfb7c05d64310d503505eeff3795c755d547b5e8aaa81cdf7",
    "gp350014-step54": "dc302a1a610b5301ae94d4b8304cf9ae4292cfbd9a209dbb98db3252bf4c9d64",
}
VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))

def is_trapped(obs):
    """Synthetic W0 local definition, not authoritative native-game semantics."""
    if "RETRACE" not in obs["legal"]:
        return False
    for action, turn in (("STRAIGHT", 0), ("LEFT", -1), ("RIGHT", 1)):
        if action not in obs["legal"]:
            continue
        dx, dy = VECTORS[(obs["heading"] + turn) % 4]
        if obs["local_5x5"][2 + dy][2 + dx] == ".":
            return False
    return True

def choose(obs, scores, retrace_bias=0.0, geometry_gate=False):
    legal = obs["legal"]
    if geometry_gate:
        # This is a transparent, tautological state-label rule for comparison;
        # it does NOT prove a neural model learned when to retrace.
        if is_trapped(obs):
            return "RETRACE"
        legal = [a for a in legal if a != "RETRACE"]
    return max(legal, key=lambda a: scores[ACTIONS.index(a)] +
               (retrace_bias if a == "RETRACE" else 0.0))

def read_original(path):
    blob = path.read_bytes()
    assert digest(blob) == ZIP_SHA256, "Original GP350-015 artifact digest mismatch"
    archive = zipfile.ZipFile(path)
    assert archive.testzip() is None
    summary = json.loads(archive.read("original-experiment/summary.json"))
    assert summary["status"] == "complete"
    assert summary["source_sha"] == EXPERIMENT_SHA
    assert summary["revision"] == MODEL_REVISION
    assert summary["new_model_decisions"] == 2304
    assert summary["new_heldout_probes"] == 360
    assert summary["original_pngs"] == 36
    stages = {}
    receipt_count = 0
    for stage in STAGES:
        original = summary["fresh_checkpoint_stages"][stage]
        assert original["adapter_sha256"] == CHECKPOINTS[stage]
        assert len(original["episodes"]) == 12
        by_size = {32: {}, 48: {}}
        episodes = []
        for ep in original["episodes"]:
            png = archive.read("original-experiment/" + ep["png"])
            assert png.startswith(b"\x89PNG\r\n\x1a\n") and digest(png) == ep["png_sha256"]
            t = json.loads(archive.read("original-experiment/" + ep["png"][:-4] + ".json"))
            assert len(t["scores"]) == len(t["record"]["events"]) == ep["decisions"] == 64
            counts = Counter()
            for scored, event in zip(t["scores"], t["record"]["events"]):
                obs, scores = scored["obs"], scored["six_logits"]
                assert len(scores) == 6 and all(math.isfinite(x) for x in scores)
                assert scored["raw"] == ACTIONS[max(range(6),key=lambda k:scores[k])]
                assert scored["applied"] == choose(obs,scores) == event["applied"] == event["raw_choice"]
                assert event["raw_legal"] is True
                assert scored["raw_legal"] == (scored["raw"] in obs["legal"])
                group = by_size[ep["size"]]
                key = canonical(obs)
                value = {"obs": obs,"scores": scores,"trapped": is_trapped(obs)}
                if key in group:
                    # The complete original prompt/observation must deterministically
                    # reproduce its full LFM action-score vector.
                    assert group[key] == value, "Repeated prompt produced different LFM logits"
                else:
                    group[key] = value
                counts["trapped"] += value["trapped"]
                counts["trapped_selected"] += value["trapped"] and scored["applied"] == "RETRACE"
                counts["premature"] += not value["trapped"] and scored["applied"] == "RETRACE"
                receipt_count += 1
            episodes.append({"seed":ep["seed"],"size":ep["size"],"family":ep["family"],
                             "unique_cells":ep["unique_cells"],"source_png_sha256":ep["png_sha256"],
                             **dict(counts)})
        size_overlap = len(set(by_size[32]).intersection(by_size[48]))
        if stage == "gp350014-step36":
            assert size_overlap == 0, "Target checkpoint world-size state leakage"
        stages[stage] = {"unique_by_size":by_size,"episodes":episodes,"size_overlap":size_overlap,
                         "mean_source_unique_cells":original["on_policy"]["mean_unique_cells"]}
    assert receipt_count == 2304
    return {"source_sha256":digest(blob),"members":len(archive.namelist()),
            "receipts":receipt_count,"original_pngs_verified":36,
            "stages":stages}

def margin(row):
    obs = row["obs"]
    if "RETRACE" not in obs["legal"]:
        return -1000.0
    s = row["scores"]
    return s[4] - max(s[ACTIONS.index(a)] for a in obs["legal"] if a != "RETRACE")

def train_scalar(rows, updates=160):
    """Real AdamW optimizer on ONE auxiliary scalar, not on frozen LFM weights."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(350017)
    torch.set_num_threads(1)
    candidates = [v for v in rows if "RETRACE" in v["obs"]["legal"]]
    positives = sum(int(x["trapped"]) for x in candidates)
    negatives = len(candidates) - positives
    assert positives >= 2 and negatives >= 40
    x = torch.tensor([margin(v) for v in candidates],dtype=torch.float64)
    y = torch.tensor([int(v["trapped"]) for v in candidates],dtype=torch.float64)
    weights = torch.where(y > 0, torch.tensor(.5/positives),torch.tensor(.5/negatives))
    bias = torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    optim = torch.optim.AdamW([bias],lr=.05,weight_decay=0.0)
    start = float((F.binary_cross_entropy_with_logits(x+bias,y,reduction="none")*weights).sum().detach())
    losses = []
    for step in range(updates):
        optim.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(x+bias,y,reduction="none")*weights).sum()+.005*bias.square()
        assert math.isfinite(float(loss.detach()))
        loss.backward()
        assert math.isfinite(float(bias.grad))
        optim.step()
        losses.append(float(loss.detach()))
    final_bias = float(bias.detach())
    assert math.isfinite(final_bias)
    return {"updates": updates,"lr": .05,"train_rows":len(candidates),
            "positive_rows":positives,"negative_rows":negatives,
            "initial_balanced_bce":start,"final_objective":losses[-1],
            "bias":final_bias,"scalar_f64_sha256":digest(struct.pack("<d",final_bias))}

def score_rows(rows, bias=None, gate=False, disable_retrace=False):
    counts = Counter()
    for item in rows:
        obs, logits, target = item["obs"], item["scores"], item["trapped"]
        if disable_retrace:
            allowed = [a for a in obs["legal"] if a != "RETRACE"]
            action = max(allowed,key=lambda a:logits[ACTIONS.index(a)])
        else:
            action = choose(obs,logits,retrace_bias=bias or 0.,geometry_gate=gate)
        counts["n"] += 1
        counts["positive"] += target
        counts["negative"] += not target
        counts["tp"] += target and action == "RETRACE"
        counts["fp"] += not target and action == "RETRACE"
        counts["fn"] += target and action != "RETRACE"
        counts["tn"] += not target and action != "RETRACE"
    assert counts["tp"]+counts["fp"]+counts["fn"]+counts["tn"] == len(rows)
    return dict(counts)

def run(path):
    source = read_original(path)
    stage = source["stages"]["gp350014-step36"]
    all_rows = {size:list(rows.values()) for size,rows in stage["unique_by_size"].items()}
    results = {}
    for training_size, test_size in ((32,48),(48,32)):
        fit = train_scalar(all_rows[training_size])
        test = all_rows[test_size]
        results[str(training_size)+"_to_"+str(test_size)] = {
            "train_size":training_size,"holdout_size":test_size,
            "fit":fit,"heldout_unique_states":len(test),
            "heldout": {
                "unchanged":score_rows(test),
                "no_retrace_control":score_rows(test,disable_retrace=True),
                "fixed_minus_half":score_rows(test,bias=-.5),
                "fixed_minus_one":score_rows(test,bias=-1.0),
                "trained_offset":score_rows(test,bias=fit["bias"]),
                "geometry_label_rule":score_rows(test,gate=True),
            }
        }
    stage_counts = {}
    for name,part in source["stages"].items():
        rows = part["unique_by_size"]
        stage_counts[name] = {"distinct_32":len(rows[32]),"distinct_48":len(rows[48]),
          "distinct_total":len(set(rows[32])|set(rows[48])),
          "shared_32_48_states":part["size_overlap"],
          "on_policy_source_coverage":part["mean_source_unique_cells"],
          "dedup_32":score_rows(rows[32].values()),
          "dedup_48":score_rows(rows[48].values())}
    return {
        "schema":"driv3r.lfm350.gp350017-unique-state-calibration/1",
        "source_archive_sha256":source["source_sha256"],
        "source_run":43943992509,"source_artifact_id":14266902004,
        "source_model_sha":EXPERIMENT_SHA,"model_revision":MODEL_REVISION,
        "source_zip_members_checked":source["members"],
        "source_original_pngs_checked":source["original_pngs_verified"],
        "source_full_model_decision_receipts_checked":source["receipts"],
        "new_full_model_forwards":0,"new_full_model_optimizer_updates":0,
        "tiny_auxiliary_scalar_optimizer_updates":320,
        "evaluated_on_recorded_states_only":True,
        "does_not_measure_replayed_policy_outcomes":True,
        "same_model_checkpoint":CHECKPOINTS["gp350014-step36"],
        "stage_controls":stage_counts,"size_disjoint_fits":results,
        "native_CyberSand_evidence":False,"vision_conditioned":False,
        "promoted":False,
    }

def selftest():
    o={"heading":0,"legal":["STRAIGHT","LEFT","RIGHT","RETRACE","STOP"],
       "local_5x5":[".....",".....","..@..",".....","....."]}
    s=[0.,0.,0.,0.,1.,0.]
    assert is_trapped(o) is False
    assert choose(o,s) == "RETRACE"
    assert choose(o,s,geometry_gate=True) != "RETRACE"
    assert choose(o,s,retrace_bias=-2.0) != "RETRACE"
    trapped = dict(o,local_5x5=[".....","..#..",".#@#.","..+..","....."])
    assert is_trapped(trapped)
    assert choose(trapped,s,geometry_gate=True) == "RETRACE"
    assert choose(trapped,s,2.0) == "RETRACE"
    assert score_rows([{"obs":o,"scores":s,"trapped":False},
                       {"obs":trapped,"scores":s,"trapped":True}])["tp"] == 1
    assert score_rows([{"obs":o,"scores":s,"trapped":False},
                       {"obs":trapped,"scores":s,"trapped":True}],gate=True)["fp"] == 0
    print("GP350017 small self-test PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("original_zip",type=Path,nargs="?")
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        selftest()
        return
    assert args.original_zip, "Original GP350-015 ZIP required"
    result=run(args.original_zip)
    data=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        assert not args.out.exists(),"Refuse overwrite of original evidence"
        args.out.write_text(data,encoding="utf-8")
    else:
        print(data,end="")
if __name__ == "__main__":
    main()
