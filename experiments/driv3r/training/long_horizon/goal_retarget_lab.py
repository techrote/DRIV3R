"""GOAL-RETARGET-001: train matched recovery/goal policies, then surprise-retarget them.

Original synthetic DEVELOPMENT only. Unseen development *seeds*, not sealed family
holdout; real CyberSand native action semantics and 64-move limit NOT established.
This module is intentionally scoped to experiments/driv3r/training/long_horizon.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_recovery_lab as prior, goal_conditioning_lab as goal
from .world import World, MOVE, digest, specs
from .train import _png, sha_file

VERSION="goal-retarget/1"
ARMS=("goal_only","goal_recovery","recovery_masked")
MODES=("static","retarget")
SWITCHES=(32,72,112)
HORIZONS=(128,160)
SOURCE_FILES=("goal_retarget_lab.py","goal_recovery_lab.py","goal_conditioning_lab.py",
              "world.py","policy.py","train.py")

def cases():
    # Different DEVELOPMENT seeds from preceding GOAL-RECOVERY-001; same
    # source structural families, NOT a sealed family-generalization claim.
    base=specs("development",2)[1::2]
    return base+[replace(s,challenge="trap_corridor") for s in base]

def targets_for(sp,seed,mode):
    worlds=World.create(sp) # Immutable instruction schedule, same for all arms
    if mode=="static":
        g=goal.goals(worlds,seed+17)
        return [g[0]]*4
    ts=[]
    for phase in range(4):
        candidates=goal.goals(worlds,seed+17+119*phase)
        ts.append(candidates[phase%len(candidates)])
    return ts

def active_phase(t,mode):
    return sum(t>=s for s in SWITCHES) if mode=="retarget" else 0

def run_episode(sp,models,arm,seed,horizon,mode,png=None):
    if sp.split!="development" or arm not in ARMS or mode not in MODES:
        raise ValueError("development-only experiment")
    w=World.create(sp);initial=w.base_sha(); targets=targets_for(sp,seed,mode)
    model=models["goal_only" if arm=="goal_only" else "goal_recovery"]
    mask=arm!="goal_recovery"
    rows=[];lat=[];reach_epochs=set();steps_changed=[]
    rng=random.Random(seed*997+sp.seed*31+horizon)
    for t in range(horizon):
        if w.stopped:break
        phase=active_phase(t,mode);target=targets[phase]
        previous=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        started=time.perf_counter_ns()
        asked,probs=model.choose(w,target,horizon,mask)
        lat.append((time.perf_counter_ns()-started)/1e6)
        actual=asked
        if t in (19,57,91,131):
            alternatives=[a for a in w.menu(12) if a in MOVE and a!=asked]
            if alternatives:actual=alternatives[rng.randrange(len(alternatives))]
        event=w.apply(actual,12)
        after=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        if target in w.ink and tuple(target) in [tuple(x) for x in event["painted"]]:
            reach_epochs.add(phase)
        rows.append({"phase":phase,"goal":list(target),"requested":asked,
                     "probabilities":probs,"forced":actual!=asked,"event":event,
                     "distance_change":previous-after})
        if mode=="retarget" and t in SWITCHES:
            steps_changed.append(t)
    if png is not None:_png(png,w)
    windows={}
    for start in SWITCHES if mode=="retarget" else (0,):
        window=[r for r in rows[start:start+8] if r["phase"]==active_phase(start,mode)]
        windows[str(start)]={"steps":len(window),"net_distance_gain":sum(r["distance_change"] for r in window),
                            "painted":sum(bool(r["event"]["painted"]) for r in window)}
    return {"version":VERSION,"arm":arm,"split":"development","spec":sp.ident,
            "seed":seed,"mode":mode,"horizon":horizon,
            "world_base_sha256":initial,"targets":[list(g) for g in targets],
            "rows":rows,"state_sha256":w.state_sha(),"trajectory_sha256":digest(w.decisions),
            "ink":sorted([list(p) for p in w.ink]),"metrics":w.metrics(horizon),
            "reached_epochs":sorted(reach_epochs),"switch_windows":windows,
            "mean_latency_ms":round(statistics.mean(lat),6) if lat else 0.,
            "action_counts":dict(sorted(Counter(r["event"]["chosen"] for r in rows).items())),
            "native_receipts":0}

def verify_episode(ep,sp,models):
    if ep["version"]!=VERSION or ep["split"]!="development" or ep["spec"]!=sp.ident:raise ValueError("identity")
    w=World.create(sp)
    if w.base_sha()!=ep["world_base_sha256"] or ep["targets"]!=[list(g) for g in targets_for(sp,ep["seed"],ep["mode"])]:
        raise ValueError("world/goal mismatch")
    model=models["goal_only" if ep["arm"]=="goal_only" else "goal_recovery"]
    rng=random.Random(ep["seed"]*997+sp.seed*31+ep["horizon"])
    reached=set()
    windows={}
    for t,r in enumerate(ep["rows"]):
        if w.stopped:raise ValueError("post-STOP action")
        phase=active_phase(t,ep["mode"]);target=tuple(ep["targets"][phase])
        before=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        asked,probs=model.choose(w,target,ep["horizon"],ep["arm"]!="goal_recovery")
        if r["phase"]!=phase or r["goal"]!=list(target) or r["requested"]!=asked or r["probabilities"]!=probs:
            raise ValueError("model action distribution corrupted")
        actual=asked
        if t in (19,57,91,131):
            alternatives=[a for a in w.menu(12) if a in MOVE and a!=asked]
            if alternatives:actual=alternatives[rng.randrange(len(alternatives))]
        if r["forced"]!=(actual!=asked) or r["event"]!=w.apply(actual,12):raise ValueError("state transition corrupted")
        after=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        if r["distance_change"]!=before-after:raise ValueError("goal progress corrupted")
        if target in w.ink and list(target) in r["event"]["painted"]:reached.add(phase)
    for start in SWITCHES if ep["mode"]=="retarget" else (0,):
        window=[r for r in ep["rows"][start:start+8] if r["phase"]==active_phase(start,ep["mode"])]
        windows[str(start)]={"steps":len(window),"net_distance_gain":sum(r["distance_change"] for r in window),
                            "painted":sum(bool(r["event"]["painted"]) for r in window)}
    if (ep["state_sha256"]!=w.state_sha() or ep["trajectory_sha256"]!=digest(w.decisions)
        or ep["metrics"]!=w.metrics(ep["horizon"])
        or ep["ink"]!=sorted([list(p) for p in w.ink]) or ep["switch_windows"]!=windows
        or ep["reached_epochs"]!=sorted(reached)
        or ep["action_counts"]!=dict(sorted(Counter(r["event"]["chosen"] for r in ep["rows"]).items()))
        or (not w.stopped and len(ep["rows"])!=ep["horizon"])):
        raise ValueError("summary corrupted")
    return len(ep["rows"])

def summary(eps):
    result={}
    for mode in MODES:
        result[mode]={}
        for arm in ARMS:
            es=[e for e in eps if e["mode"]==mode and e["arm"]==arm]
            a=Counter()
            for e in es:a.update(e["action_counts"])
            ws=[v for e in es for k,v in e["switch_windows"].items() if int(k)!=0]
            result[mode][arm]={
                "episodes":len(es),"mean_ink":round(statistics.mean(e["metrics"]["unique_painted_cells"] for e in es),3),
                "premature_STOP":sum(e["metrics"]["stopped_early"] for e in es),
                "targets_painted_during_instruction":sum(len(e["reached_epochs"]) for e in es),
                "mean_branch_cells":round(statistics.mean(e["metrics"]["branch_cells"] for e in es),3),
                "mean_turns":round(statistics.mean(e["metrics"]["turns"] for e in es),3),
                "mean_switch_8step_distance_gain":round(statistics.mean(w["net_distance_gain"] for w in ws),4) if ws else None,
                "switch_windows_observed":sum(w["steps"]>0 for w in ws),
                "action_counts":dict(sorted(a.items())),
                "mean_decision_latency_ms":round(statistics.mean(e["mean_latency_ms"] for e in es),6)}
    return result

def provenance():
    d=Path(__file__).parent
    return {name:sha_file(d/name) for name in SOURCE_FILES}

def run(out,seed=617,updates=60,quick=False):
    if out.exists():raise FileExistsError("immutable output exists")
    if not 1<=updates<=200:raise ValueError("updates bounded")
    out.mkdir(parents=True)
    started=time.perf_counter()
    models,history=prior.train(seed,updates)
    for name,model in models.items():
        cp=out/(name+".npz");model.save(cp)
        restored=prior.Ranker.load(cp)
        if any(not np.array_equal(model.p[k],restored.p[k]) for k in model.p):raise ValueError("checkpoint restore")
    (out/"traces").mkdir()
    es=[];n=0
    selected=cases()[:1] if quick else cases()
    for sp in selected:
        for h in HORIZONS[:1 if quick else 2]:
            for mode in MODES:
                for arm in ARMS:
                    png=out/"traces"/f"{sp.ident}-{h}-{mode}-{arm}.png" if h==160 and mode=="retarget" else None
                    ep=run_episode(sp,models,arm,seed,h,mode,png)
                    n+=verify_episode(ep,sp,models)
                    (out/"traces"/f"{sp.ident}-{h}-{mode}-{arm}.json").write_text(json.dumps(ep,sort_keys=True,separators=(",",":"))+"\n")
                    es.append(ep)
    rep={"version":VERSION,"seed":seed,"updates_per_stage":updates,
         "stages":(36,72,128),"history":history,
         "checkpoints":{name:{"sha256":sha_file(out/(name+".npz")),
                              "parameters":sum(p.size for p in m.p.values()),"updates":m.updates}
                        for name,m in models.items()},
         "summary":summary(es),"episodes":len(es),"decisions":n,"source_sha256":provenance(),
         "split":"development","new_development_seeds":[sp.seed for sp in selected],
         "sealed_structural_heldout_opened":False,"native_receipts":0,"paid_compute":False,
         "elapsed_s":round(time.perf_counter()-started,3),
         "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    (out/"SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in out.rglob("*") if p.is_file()},indent=2,sort_keys=True)+"\n")
    return rep

def verify(out):
    rep=json.loads((out/"report.json").read_text())
    if rep["version"]!=VERSION or rep["source_sha256"]!=provenance():raise ValueError("source identity mismatch")
    file_hashes={str(p.relative_to(out)):sha_file(p) for p in out.rglob("*") if p.is_file() and p.name!="SHA256SUMS.json"}
    if file_hashes!=json.loads((out/"SHA256SUMS.json").read_text()):raise ValueError("artifact hash mismatch")
    models={name:prior.Ranker.load(out/(name+".npz")) for name in ("goal_only","goal_recovery")}
    for name in models:
        if sha_file(out/(name+".npz"))!=rep["checkpoints"][name]["sha256"] or models[name].updates!=rep["checkpoints"][name]["updates"]:
            raise ValueError("checkpoint identity mismatch")
    available={s.ident:s for s in cases()}
    es=[];n=0
    for p in sorted((out/"traces").glob("*.json")):
        ep=json.loads(p.read_text())
        n+=verify_episode(ep,available[ep["spec"]],models);es.append(ep)
    if n!=rep["decisions"] or len(es)!=rep["episodes"]:raise ValueError("counts")
    summ=summary(es)
    for mode in MODES:
        for arm in ARMS:
            observed=dict(summ[mode][arm]);expected=dict(rep["summary"][mode][arm])
            observed.pop("mean_decision_latency_ms");expected.pop("mean_decision_latency_ms")
            if observed!=expected:raise ValueError("aggregated outcome mismatch")
    return {"episodes":len(es),"decisions":n,"checkpoints":len(models),"file_hashes":len(file_hashes)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path);p.add_argument("--verify",type=Path)
    p.add_argument("--seed",type=int,default=617);p.add_argument("--updates",type=int,default=60)
    p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,a.updates,a.quick),sort_keys=True))
