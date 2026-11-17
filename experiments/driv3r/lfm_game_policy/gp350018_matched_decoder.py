#!/usr/bin/env python3
"""GP350-018: matched frozen LFM350 legal-action decoding A/B/C on synthetic W0.

Actual full-model inference required for results. This runner does NOT train
weights; archived-logit preflight is not a closed-loop experiment. No native
CyberSand assertion. No game-side writes. No decoder heuristic steering.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import statistics
import time

import lfm350 as c
import gp350_paired_margin_train as g

SCHEMA = "driv3r.lfm350.gp350018-matched-decoder/1"
MODEL_ADAPTER_SHA256 = "436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c"
# These parameters are frozen before evaluating closed-loop outcomes.
ARMS = (
    ("A-greedy", "greedy", None, 0.0),
    ("B-t1p25", "sample", 1.25, 0.0),
    ("B-t2p50", "sample", 2.50, 0.0),
    ("C-m0p10", "mixture", 1.25, 0.10),
    ("C-m0p25", "mixture", 1.25, 0.25),
)
PILOT_WORLDS = ((24, 9803), (64, 9821))
EXTENDED_WORLDS = ((24, 9803), (24, 9821), (64, 9803), (64, 9821))
STOP_REQUESTED = False

def interrupt(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

def percentile(values, p):
    if not values:
        return None
    a = sorted(values)
    return a[min(len(a)-1, max(0, math.ceil(p * len(a))-1))]

def probs(logits, legal, kind, temp, mixture):
    """Return host-legal probabilities; NEVER sample or score illegal actions."""
    c.check(len(logits) == 6 and all(math.isfinite(float(v)) for v in logits),
            "invalid actual six-way model scores")
    c.check(legal and len(legal) == len(set(legal))
            and set(legal).issubset(set(c.ACTIONS)),
            "unknown/duplicate host-legal action")
    if kind == "greedy":
        best = max(legal, key=lambda a: float(logits[c.ACTIONS.index(a)]))
        return {a: float(a == best) for a in legal}
    c.check(kind in ("sample", "mixture") and temp is not None and temp > 0
            and 0 <= mixture < 1, "invalid declared decoding policy")
    values = [float(logits[c.ACTIONS.index(a)]) for a in legal]
    hi = max(values)
    weights = [math.exp((v-hi)/temp) for v in values]
    denom = sum(weights)
    c.check(denom > 0 and math.isfinite(denom), "softmax normalization failed")
    return {a: (1-mixture) * w/denom + mixture/len(legal)
            for a, w in zip(legal, weights)}

def sampling_uniform(size, seed, family, arm, tick):
    """Common random numbers for first matched observation, keyed per turn."""
    key = f"gp350018|{size}|{seed}|{family}|{arm}|{tick}"
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")).random()

def pick(probabilities, u):
    c.check(0 <= u < 1 and abs(sum(probabilities.values()) - 1) < 1e-8,
            "probabilities not a normalized host-legal distribution")
    acc = 0.0
    for action, p in probabilities.items():
        acc += p
        if u < acc:
            return action
    return next(reversed(probabilities))  # Only tolerates floating rounding.

def shape(record, decisions):
    trajectory = [tuple(p) for p in record["trajectory"]]
    positions = [tuple(e["position_after"]) for e in record["events"]]
    repeats_2 = sum(trajectory[i] == trajectory[i-2]
                    for i in range(2, len(trajectory)))
    deltas = [(b[0]-a[0], b[1]-a[1]) for a,b in zip(trajectory,trajectory[1:])]
    last = None
    run = longest = changes = 0
    for delta in deltas:
        if delta == (0,0):
            run, last = 0, None
        else:
            if last is not None and delta != last:
                changes += 1
            run = run+1 if delta == last else 1
            longest = max(longest, run)
            last = delta
    actions = [d["applied"] for d in decisions]
    hooks_proxy = sum(a == b and a in ("LEFT", "RIGHT")
                      for a,b in zip(actions, actions[1:]))
    distinct = len(set(trajectory))
    return {
        "position_after_count": len(positions),
        "two_step_oscillations": repeats_2,
        "longest_constant_displacement_run": longest,
        "heading_vector_changes": changes,
        "consecutive_same_side_turn_pairs_proxy": hooks_proxy,
        "new_territory_distinct_cells": distinct,
        "novel_cells_per_decision": (distinct-1)/max(1,len(decisions)),
        "actions": dict(collections.Counter(actions)),
        "branch_actions": actions.count("BRANCH"),
        "retrace_actions": actions.count("RETRACE"),
        "stop_actions": actions.count("STOP"),
        "early_stop": bool(record["metrics"]["terminated_by_stop"]),
        "revisit_contacts": record["metrics"]["total_revisit_contacts"],
    }

def svg_path(record, size, caption):
    """Literal SVG trace reconstructed from actual W0 coordinates, not generated art."""
    path = record["trajectory"]
    c.check(len(path) >= 1 and all(len(p) == 2 for p in path), "bad W0 trace")
    unit = 7
    pts = " ".join(f"{(x+1)*unit},{(y+1)*unit+28}" for x,y in path)
    escaped = html.escape(caption)
    width = (size+2)*unit
    height = (size+2)*unit+28
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#101723"/>\n'
            f'<text x="8" y="17" fill="#eeeeee" font-size="10">{escaped}</text>\n'
            f'<polyline fill="none" stroke="#6ce1b2" stroke-width="1.5" '
            f'stroke-linejoin="round" points="{pts}"/>\n'
            f'<circle cx="{(path[0][0]+1)*unit}" cy="{(path[0][1]+1)*unit+28}" '
            'r="2.5" fill="#ffffff"/>\n'
            f'<circle cx="{(path[-1][0]+1)*unit}" cy="{(path[-1][1]+1)*unit+28}" '
            'r="3" fill="#ffb062"/>\n</svg>\n')

def episode(w0, model, tok, ids, out, stage, arm, size, seed, family,
            budget, started, deadline):
    import torch
    name, kind, temp, mix = arm
    rows = []
    latencies = []
    def selector(obs):
        c.check(not STOP_REQUESTED and time.monotonic()-started < deadline-120,
                "GP350018 CPU deadline reached (partial evidence retained)")
        t0 = time.perf_counter()
        with torch.inference_mode():
            z = g.rich_score(model, tok, ids, obs)
        latency = (time.perf_counter()-t0)*1000
        scores = [float(x) for x in z]
        legal = list(obs["legal"])
        distribution = probs(scores, legal, kind, temp, mix)
        u = sampling_uniform(size,seed,family,name,int(obs["tick"]))
        applied = pick(distribution, u)
        raw = c.ACTIONS[max(range(6),key=lambda i: scores[i])]
        c.check(applied in legal, "decoder violated host authority")
        latencies.append(latency)
        rows.append({"tick":obs["tick"],"obs":obs,"obs_sha256":
            c.sha_bytes(c.canonical(obs).encode()),"legal":legal,"six_logits":scores,
            "raw":raw,"raw_legal":raw in legal,"probabilities":distribution,
            "sampling_u":u,"applied":applied,"model_forward_ms":latency})
        return applied

    record = w0.run(seed, mode="callback", family=family, size=size,
                    steps=budget, selector=selector)
    c.check(0 < len(rows) == len(record["events"]) <= budget, "W0 receipt mismatch")
    for row,event in zip(rows,record["events"]):
        c.check(event["raw_legal"] and event["raw_choice"]==row["applied"]==event["applied"]
                and row["applied"] in row["legal"], "host applied non-decoder action")
    replay = w0.run(seed,mode="replay",family=family,size=size,steps=budget,
                    raw_actions=[r["applied"] for r in rows])
    png = w0.png_bytes(record)
    c.check(w0.png_bytes(replay) == png and
            replay["trajectory"] == record["trajectory"], "W0 replay not equivalent")
    prefix = f"{stage}-{name}-s{size}-{seed}-{family}-{budget}"
    pic = prefix+".png"
    trace = prefix+".svg"
    (out/pic).write_bytes(png)
    rawsvg=svg_path(record,size,prefix).encode()
    (out/trace).write_bytes(rawsvg)
    c.save_json(out/(prefix+".json"), {"schema":SCHEMA,
                "stage":stage,"arm":arm,"size":size,"seed":seed,"family":family,
                "decisions":rows,"record":record,"real_full_model_scores":True})
    m=record["metrics"]
    return {"stage":stage,"arm":name,"size":size,"seed":seed,"family":family,
            "turn_budget":budget,"executed":len(rows),"initial_observation_sha256":rows[0]["obs_sha256"],
            "initial_legal":rows[0]["legal"],"raw_illegal_preferences":
            sum(not r["raw_legal"] for r in rows),
            "host_illegal":m["invalid_raw_choices"],"model_forward_p50_ms":percentile(latencies,.5),
            "model_forward_p95_ms":percentile(latencies,.95),
            "model_forward_max_ms":max(latencies),
            "shape":shape(record,rows),"png":pic,"png_sha256":c.sha_bytes(png),
            "svg":trace,"svg_sha256":c.sha_bytes(rawsvg)}

def self_test():
    legal = ["LEFT","STOP"]
    q = [100., 1., -100.,-100.,-100.,0.]
    a = probs(q,legal,"greedy",None,0)
    assert a == {"LEFT":1.,"STOP":0.}
    assert pick(a,0.999)=="LEFT"
    for kind,t,mix in (("sample",1.25,0),("sample",2.5,0),
                       ("mixture",1.25,.10),("mixture",1.25,.25)):
        p=probs(q,legal,kind,t,mix)
        assert abs(sum(p.values())-1)<1e-12
        assert set(p)==set(legal) and all(0<=v<=1 for v in p.values())
        assert all(pick(p,x) in legal for x in (.0,.19,.49,.77,.9999))
    assert sampling_uniform(24,9803,"greebles","B-t1p25",3) == \
           sampling_uniform(24,9803,"greebles","B-t1p25",3)
    assert sampling_uniform(24,9803,"greebles","B-t1p25",3) != \
           sampling_uniform(24,9803,"greebles","B-t1p25",4)
    assert percentile([4,8,2,6],.95)==8
    assert len(ARMS)==5 and PILOT_WORLDS==((24,9803),(64,9821))
    print(c.canonical({"GP350018_SELF_TEST":"PASS","arms":ARMS,
           "pilot_worlds":PILOT_WORLDS,"source_model_scoring":"g.rich_score"}))

def run(args):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from huggingface_hub import HfApi,snapshot_download
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(350018)
    c.check(not args.out.exists(), "immutable output lease already exists")
    args.out.mkdir(parents=True)
    worlds = EXTENDED_WORLDS if args.extended else PILOT_WORLDS
    c.check(args.stage in ("stock","gp006") and args.turns in (64,128)
            and 1200 <= args.deadline <= 7200, "invalid bounded recipe")
    if args.stage=="gp006":
        c.check(args.adapter is not None and c.sha(args.adapter/"adapter_model.safetensors")
                == MODEL_ADAPTER_SHA256,"pinned GP006 adapter custody failed")
    w0=c.load_w0(args.w0)
    c.save_json(args.out/"lease.json",{"schema":SCHEMA,"model":c.MODEL,
        "model_revision":c.REVISION,"source_sha":os.getenv("GITHUB_SHA"),
        "run_id":os.getenv("GITHUB_RUN_ID"),"run_attempt":os.getenv("GITHUB_RUN_ATTEMPT"),
        "stage":args.stage,"adapter_sha256": MODEL_ADAPTER_SHA256 if args.stage=="gp006" else None,
        "W0_commit":c.W0_COMMIT,"W0_git_blob":c.W0_BLOB,
        "policy_arms":ARMS,"worlds":worlds,"families":c.FAMILIES,
        "turn_budget":args.turns,"optimizer_updates":0,"native":False,
        "model_vision_capable":False,"original_W13_is_NOT_matched_control":True,
        "draft_hold":True})
    info=HfApi().model_info(c.MODEL,revision=c.REVISION)
    c.check(info.sha==c.REVISION, "LFM350 upstream pin drift")
    snapshot_download(c.MODEL,revision=c.REVISION,local_dir=args.cache,max_workers=2,
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
          "generation_config.json","tokenizer*","special_tokens_map.json",
          "added_tokens.json","vocab.*","merges.txt","README.md","LICENSE*"])
    manifest={str(p.relative_to(args.cache)):c.sha(p) for p in sorted(args.cache.rglob("*"))
              if p.is_file() and ".cache" not in p.parts}
    c.check(any(p.endswith(".safetensors") for p in manifest),
            "pinned full LFM350 weights unavailable")
    c.save_json(args.out/"model-files-sha256.json",manifest)
    tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
    ids=c.id_map(tok)
    base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                                            trust_remote_code=False,dtype=torch.float32)
    c.check(base.config.model_type=="lfm2" and base.config.num_hidden_layers==16,
            "wrong actual LFM350 architecture")
    base.requires_grad_(False)
    base.eval()
    model=(base if args.stage=="stock" else
           PeftModel.from_pretrained(base,args.adapter,is_trainable=False))
    c.check(not any(x.requires_grad for x in model.parameters()),
            "frozen checkpoint unexpectedly trainable")
    model.eval()
    frozen=c.base_fingerprint(model)
    result=[]
    first={}
    for size,seed in worlds:
        for family in c.FAMILIES:
            for arm in ARMS:
                row=episode(w0,model,tok,ids,args.out,args.stage,arm,size,seed,family,
                            args.turns,started,args.deadline)
                key=(size,seed,family)
                if key in first:
                    c.check(row["initial_observation_sha256"]==first[key][0] and
                            row["initial_legal"]==first[key][1],
                            "not a common initial W0 world/menu")
                else:
                    first[key]=(row["initial_observation_sha256"],row["initial_legal"])
                c.check(row["host_illegal"]==0,"host legality failed")
                result.append(row)
                print(c.canonical({"GP350018_EPISODE":row}),flush=True)
    c.check(c.base_fingerprint(model)==frozen,"pretrained parent fingerprint changed")
    for row in result:
        c.check(c.sha(args.out/row["png"])==row["png_sha256"] and
                c.sha(args.out/row["svg"])==row["svg_sha256"],"gallery hash mismatch")
    summary={"schema":SCHEMA,"status":"complete","stage":args.stage,
        "source_sha":os.getenv("GITHUB_SHA"),"model":c.MODEL,"model_revision":c.REVISION,
        "adapter_sha256":MODEL_ADAPTER_SHA256 if args.stage=="gp006" else None,
        "worlds":worlds,"arms":ARMS,"families":c.FAMILIES,"turn_budget":args.turns,
        "outcomes":result,"episodes":len(result),
        "full_model_decisions":sum(x["executed"] for x in result),
        "host_illegal_actions":sum(x["host_illegal"] for x in result),
        "raw_illegal_preferences":sum(x["raw_illegal_preferences"] for x in result),
        "pngs":len(result),"svgs":len(result),"optimizer_steps":0,
        "native":False,"synthetic_W0_only":True,"matched_d1":"NOT RUN",
        "initial_worlds_matched_only":True,"post_action_menus_may_diverge":True,
        "wall_seconds":time.monotonic()-started,"cpu_seconds":time.process_time(),
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    c.save_json(args.out/"summary.json",summary)
    print(c.canonical({"GP350018_COMPLETE":args.stage,
                       "episodes":len(result),"decisions":summary["full_model_decisions"]}),
          flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--w0",type=Path)
    p.add_argument("--cache",type=Path)
    p.add_argument("--adapter",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--stage",choices=("stock","gp006"))
    p.add_argument("--turns",type=int,choices=(64,128),default=64)
    p.add_argument("--extended",action="store_true")
    p.add_argument("--deadline",type=int,default=2700)
    args=p.parse_args()
    self_test()
    if args.self_test:
        return
    c.check(args.w0 is not None and args.cache is not None
            and args.out is not None and args.stage is not None,
            "missing required full-model run inputs")
    try:
        run(args)
    except BaseException as exc:
        args.out.mkdir(parents=True,exist_ok=True)
        if not (args.out/"failure.json").exists():
            c.save_json(args.out/"failure.json",{"error":type(exc).__name__,
               "message":str(exc),"source_sha":os.getenv("GITHUB_SHA"),
               "real_full_model_complete":False,"retain_partial_artifact":True})
        raise

if __name__=="__main__":
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    main()
