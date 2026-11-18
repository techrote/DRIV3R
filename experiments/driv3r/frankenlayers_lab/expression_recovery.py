#!/usr/bin/env python3
"""DRIV3R matched Q8/Q4 expressive-trajectory recovery, real System-One CPU research.

Only model-origin /v1/systemone probability vectors are decoded. The synthetic
W0 host retains sole authority for applying legal movement; no rule forces turns.
A/B quant specimens use the same pinned source-built binary, prompt, host seed,
starting world, legal action menu and common-random-number decoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import traceback
from xml.sax.saxutils import escape

from experiments.driv3r.frankenlayers_lab import q4_greyhound as pins

Q8_SHA = "073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c"
Q8_BYTES = 1846240576
Q8_REV = pins.Q4_REPO_REV
PINNED_LLAMA = pins.LLAMA_SHA
MODES = ("stock", "greyhound", "branchwitch", "aero4_no2", "grey3_cut")
QUANTS = ("q8", "q4")
DECODERS = ("greedy", "sampled")
SCREEN_FOLDS = ((570, "greebles"), (7703, "pipework"), (8807, "architecture"))
LONG_FOLDS = SCREEN_FOLDS + ((9601, "greebles"),)
SCREEN_STEPS = 40
LONG_STEPS = 96
TIME = 1.5
FLAT_MIX = 0.25
MIN_WORLDS = 3
STYLE_NICHES = ("Meanderer", "Curler", "Brancher", "Chaotic Artist")


def request_validation(request):
    if not isinstance(request, dict) or request.get("schema") != "driv3r/expression-recovery-request/v1":
        raise ValueError("Unexpected request schema")
    phase = request.get("phase")
    if phase not in ("screen", "long"):
        raise ValueError("Invalid phase")
    if request.get("branch") != pins.BRANCH or request.get("llama_commit") != PINNED_LLAMA:
        raise ValueError("Branch/runtime provenance mismatch")
    if request.get("q8_sha256") != Q8_SHA or request.get("q8_bytes") != Q8_BYTES:
        raise ValueError("Q8 donor identity mismatch")
    if request.get("q4_sha256") != pins.Q4_SHA or request.get("q4_bytes") != pins.Q4_BYTES:
        raise ValueError("Q4 vendor identity mismatch")
    if (request.get("temperature") != TIME or request.get("uniform_mixture") != FLAT_MIX
        or request.get("decoders") != list(DECODERS) or request.get("quants") != list(QUANTS)):
        raise ValueError("Unmatched decoding or quant assignment")
    if type(request.get("nonce")) is not str or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{3,79}", request["nonce"]):
        raise ValueError("Invalid request nonce")
    if request.get("modes") != list(MODES if phase == "screen" else ("aero4_no2", "grey3_cut")):
        raise ValueError("Unregistered mode ordering or changed research arms")
    if request.get("folds") != [list(x) for x in (SCREEN_FOLDS if phase == "screen" else LONG_FOLDS)]:
        raise ValueError("Wrong grouped source-world split")
    if request.get("steps_per_episode") != (SCREEN_STEPS if phase == "screen" else LONG_STEPS):
        raise ValueError("Unbounded or mismatched step horizon")
    if phase == "long" and not re.fullmatch(r"[0-9]{6,14}", str(request.get("screen_artifact_id", ""))):
        raise ValueError("Long horizon requires actual audited screen artifact ID")
    if request.get("cpu_only") is not True or request.get("max_parallel") != 1:
        raise ValueError("Not bounded to one CPU runner")
    for prop in ("paid_api", "paid_gpu", "automatic_push_on_source", "native_game",
                 "ready", "merge", "owner_ratings_supplied"):
        if request.get(prop) is not False:
            raise ValueError(f"Forbidden state {prop}")
    return request


def pin_validation():
    from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
    from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS
    assert PinnedFull() == list(range(24))
    for name in MODES[1:]:
        masks = (LAB_FLOWS if name in LAB_FLOWS
                 else pins.EXTRA_FLOWS if name in pins.EXTRA_FLOWS else FLOWS)
        assert name in masks and len(masks[name]) >= 3
    assert pins.Q4_SHA != Q8_SHA
    assert len(set(SCREEN_FOLDS)) == len(SCREEN_FOLDS)
    assert len({seed for seed,_ in LONG_FOLDS}) == len(LONG_FOLDS)
    return True


def PinnedFull():
    from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
    return list(FLOWS["stock"])


def _json(path, obj):
    pins.write_json(path, obj)


def q8_acquire(root, evidence):
    root, evidence = Path(root), Path(evidence)
    filename = "d1-2B-Q8_0.gguf"
    source = ("https://huggingface.co/LiquidAI/d1-2B-GGUF/resolve/"
              + Q8_REV + "/" + filename)
    dest = root/filename
    try:
        outcome = pins._download_bounded(source, dest, Q8_BYTES, Q8_SHA)
        _json(evidence/"official-q8-receipt.json",{
            "schema":"driv3r/official-q8-donor/1","url":source,"sha256":Q8_SHA,
            "expected_bytes":Q8_BYTES,"state":outcome,"revision":Q8_REV})
    except Exception as exc:
        _json(evidence/"q8-acquisition-failure.json",{
            "status":"FAILED","url":source,"expected_sha256":Q8_SHA,
            "error":repr(exc)})
        raise
    return dest


def normalize(model_probs, legal, decoder):
    if not legal or not set(legal).issubset(model_probs):
        raise ValueError("Legal option mismatch")
    ps=[float(model_probs[x]) for x in legal]
    if any(not math.isfinite(p) or p < 0 for p in ps) or abs(sum(ps)-1)>2e-5:
        raise ValueError("Invalid raw model probability vector")
    if decoder=="greedy":
        return {legal[i]:float(i == max(range(len(ps)),key=lambda j:ps[j])) for i in range(len(ps))}
    if decoder!="sampled":
        raise ValueError("Unexpected decoder")
    powered=[p ** (1.0 / TIME) for p in ps]
    denom=sum(powered)
    if denom <= 0:
        raise ValueError("All model candidates zero")
    return {name:(1-FLAT_MIX)*powered[i]/denom+FLAT_MIX/len(legal)
            for i,name in enumerate(legal)}


def common_uniform(seed, family, mode, tick, decoder):
    """Same random draw for Q8/Q4; no checkpoint label in hash."""
    key=f"driv3r-expression-v1|{seed}|{family}|{mode}|{tick}|{decoder}"
    return (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")+0.5) / 2**64


def sample(probs, unit):
    if not 0 < unit < 1:
        raise ValueError("Non-unit random draw")
    acc=0.
    for name,p in probs.items():
        acc+=p
        if unit<acc:
            return name
    return list(probs)[-1]


@contextmanager
def decoder_proxy(paired, decoder):
    """Never alter original scores. Record explicit host-sampled choice separately."""
    parent=paired.Server
    class Coded(parent):
        def choose(self, obs, records):
            argmax=super().choose(obs, records)
            rec=records[-1]
            legal=tuple(obs["legal"])
            distribution=normalize(rec["parsed"]["probabilities"],legal,decoder)
            unit=common_uniform(paired.SEED, obs["family"],self.mode,obs["tick"],decoder)
            chosen=argmax if decoder=="greedy" else sample(distribution,unit)
            if chosen not in legal:
                raise RuntimeError("Sampler attempted action outside host legal menu")
            rec["model_argmax"]=argmax
            rec["chosen_decoder"]=decoder
            rec["decoded_choice"]=chosen
            rec["sample_unit"]=unit if decoder!="greedy" else None
            rec["decoded_distribution"]=distribution
            rec["host_sampling_override"]=chosen!=argmax
            rec["choice"]=chosen   # W0 applies exactly this legal output
            return chosen
    paired.Server=Coded
    try:
        yield
    finally:
        paired.Server=parent


def cycle_report(positions):
    """Find sustained closed periodic paths, periods 2..16; forbid square-loop false success."""
    pts=[tuple(p) for p in positions]
    best=None
    for period in range(2,17):
        for start in range(max(0,len(pts)-3*period)):
            if pts[start]!=pts[start+period]:
                continue
            span=pts[start:start+period+1]
            repetitions=1
            while start+(repetitions+1)*period < len(pts):
                rhs=pts[start+repetitions*period:start+(repetitions+1)*period+1]
                if rhs!=span:
                    break
                repetitions+=1
            if repetitions>=3:
                candidate={"period":period,"repetitions":repetitions,"start":start,
                           "repeated_steps":period*repetitions}
                if best is None or candidate["repeated_steps"]>best["repeated_steps"]:
                    best=candidate
    return best or {"period":None,"repetitions":0,"start":None,"repeated_steps":0}


def describe_path(world, records):
    path=world["trajectory"]
    steps=len(world["events"])
    if len(path)!=steps+1 or len(records)!=steps:
        raise RuntimeError("Host trajectory length disagrees with model calls")
    for i, (event, call) in enumerate(zip(world["events"],records)):
        if (event["observation"] != call["observation"]
            or event["raw_choice"]!=call["decoded_choice"]
            or event["applied"]!=call["decoded_choice"]
            or not event["raw_legal"]
            or tuple(event["position_after"])!=tuple(path[i+1])):
            raise RuntimeError(f"Original model-to-host action mismatch at {i}")
        if call.get("status")!="ok" or call["http_status"]!=200:
            raise RuntimeError("Non-successful model response used as a decision")
    new = world["metrics"]["new_steps"]
    turn=0
    reversals=0
    straight_max=0
    run=0
    prev_heading=None
    for i,event in enumerate(world["events"]):
        if event["applied"] in ("RETRACE","STOP"):
            prev_heading=None
            run=0
            continue
        before=path[i];after=path[i+1]
        dx,dy=after[0]-before[0],after[1]-before[1]
        if abs(dx)+abs(dy)!=1:
            raise RuntimeError("Non-cardinal painted movement")
        heading=(dx,dy)
        if prev_heading is not None and heading != prev_heading:
            turn+=1
            if heading==(-prev_heading[0],-prev_heading[1]):
                reversals+=1
        run=run+1 if heading==prev_heading else 1
        straight_max=max(straight_max,run)
        prev_heading=heading
    revisits=world["metrics"]["total_revisit_contacts"]
    bbox=(min(x for x,y in path),min(y for x,y in path),
          max(x for x,y in path),max(y for x,y in path))
    c=cycle_report(path)
    return {
        "decisions":steps,"unique_positions":len(set(map(tuple,path))),
        "new_cells":new,"new_cell_fraction":round(new/max(1,steps),5),
        "turn_count":turn,"reverse_heading_changes":reversals,
        "longest_painted_straight_run":straight_max,
        "two_step_returns":sum(path[i]==path[i-2] for i in range(2,len(path))),
        "cycle_2_16":c,"square_loop_collapse":bool(c["period"]==4 and c["repetitions"]>=3),
        "revisit_contacts":revisits,"bounds":list(bbox),
        "spatial_area_cells":(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1),
        "displacement":round(math.dist(path[0],path[-1]),4),
        "branch_choices":world["metrics"]["branches"],
        "retraces":world["metrics"]["retraces"],
        "stop_early":world["metrics"]["terminated_by_stop"] and steps < world["step_budget"],
        "decoder_overrides":sum(c["host_sampling_override"] for c in records),
        "model_argmax_counts":dict(Counter(c["model_argmax"] for c in records)),
        "host_decoded_counts":dict(Counter(c["decoded_choice"] for c in records)),
        "raw_host_legal":all(c["decoded_choice"] in c["observation"]["legal"] for c in records),
        "latency_p50_s":statistics.median([c["latency_seconds"] for c in records]) if records else None,
        "latency_p95_s":sorted(c["latency_seconds"] for c in records)[
            min(len(records)-1,math.ceil(.95*len(records))-1)] if records else None,
        "artistic_personality":"UNRATED_OWNER_BLIND_REVIEW_REQUIRED"
    }


def svg_for_ink(world, title, *, is_blind=False):
    """Original host-applied positions; RETRACE is a lift, not a painted segment."""
    path=world["trajectory"]
    minx=min(p[0] for p in path)-3;miny=min(p[1] for p in path)-3
    maxx=max(p[0] for p in path)+3;maxy=max(p[1] for p in path)+3
    xspan=max(8,maxx-minx);yspan=max(8,maxy-miny)
    ink=[]
    for i,event in enumerate(world["events"]):
        if event["applied"] in ("RETRACE","STOP"):
            continue
        a=path[i];b=path[i+1]
        if a!=b:
            ink.append(f'<path d="M{a[0]},{a[1]} L{b[0]},{b[1]}" '
                       'stroke="#2473ac" stroke-width=".55" fill="none" '
                       'stroke-linecap="round"/>')
    movement=" ".join(f"{x},{y}" for x,y in path)
    label=escape(title)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{minx} {miny} {xspan} {yspan}">'
            f'<title>{label} — original executed synthetic W0 steps; RETRACE is pen-up</title>'
            f'<rect x="{minx}" y="{miny}" width="{xspan}" height="{yspan}" fill="#f9f7f2"/>'
            f'<polyline points="{movement}" stroke="#9c9eaa" stroke-width=".13" '
            'stroke-dasharray=".38,.45" fill="none"/>'
            + "".join(ink)
            +f'<circle cx="{path[0][0]}" cy="{path[0][1]}" r=".6" fill="#218e72"/>'
            +f'<circle cx="{path[-1][0]}" cy="{path[-1][1]}" r=".6" fill="#b04c3e"/>'
            +'</svg>\n')


def build_blind_gallery(rows, output, phase):
    """Original geometry A4 SVG sheets; never show model, quant or decoder on blind ballot."""
    from xml.etree import ElementTree as ET
    output=Path(output)
    rng=sorted(rows,key=lambda x:x["anonymous_id"])
    reveal={x["anonymous_id"]:{k:x[k] for k in ("quant","mode","decoder","seed","family",
                                               "source_relpath","metrics")} for x in rng}
    _json(output/"blind-identity-reveal.json",reveal)
    _json(output/"blank-owner-ballot.json",[
        {"id":x["anonymous_id"],"meanderer":None,"curler":None,"brancher":None,
         "chaotic_artist":None,"overall":None,"notes":""} for x in rng])
    per=15
    # Plots are derived exclusively from completed synthetic host trajectories.
    for sheet in range(math.ceil(len(rng)/per)):
        subset=rng[sheet*per:(sheet+1)*per]
        graphics=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
                  '<rect width="210" height="297" fill="#f8f7f4"/>',
                  '<text x="9" y="13" font-size="6.2" font-family="sans-serif" fill="#22334a">'
                  'DRIV3R / BLIND WORM PATHS</text>',
                  '<text x="9" y="20" font-size="2.9" fill="#687383">'
                  f'{escape(phase)} | sheet {sheet+1}/{math.ceil(len(rng)/per)} | '
                  'synthetic-host, model-origin decisions; raw step geometry</text>']
        for i,item in enumerate(subset):
            col=i%3;row=i//3
            left=8+col*66;top=26+row*52
            graphics.append(f'<rect x="{left}" y="{top}" width="62" height="49" rx="1.8"'
                            ' fill="white" stroke="#d3d8df" stroke-width=".35"/>')
            graphics.append(f'<text x="{left+2}" y="{top+5}" font-size="3" fill="#242e3c">'
                            f'{escape(item["anonymous_id"])}</text>')
            try:
                doc=ET.parse(output/item["source_relpath"]).getroot()
                view=list(map(float,doc.attrib["viewBox"].split()))
                paths=doc.findall(".//{http://www.w3.org/2000/svg}path")
                poly=doc.findall(".//{http://www.w3.org/2000/svg}polyline")
                # Translate scale preserve geometry; no Bezier smoothing.
                scale=min(56/max(8,view[2]),37/max(8,view[3]))
                ox=left+3+((56-view[2]*scale)/2)-view[0]*scale
                oy=top+9+((37-view[3]*scale)/2)-view[1]*scale
                graphics.append(f'<g transform="translate({ox:.4f} {oy:.4f}) scale({scale:.5f})">')
                for p in poly:
                    graphics.append('<polyline points="'+escape(p.attrib["points"])+'" '
                                    'stroke="#c4c7ce" stroke-width=".16" fill="none"/>')
                for p in paths:
                    graphics.append('<path d="'+escape(p.attrib["d"])+'" '
                                    'stroke="#23699b" stroke-width=".42" stroke-linecap="round" fill="none"/>')
                graphics.append('</g>')
            except Exception as err:
                graphics.append(f'<text x="{left+4}" y="{top+22}" font-size="3" fill="#a43b30">'
                                'MISSING/INVALID EVIDENCE</text>')
                item["gallery_error"]=str(err)
        graphics.append('<text x="9" y="292" font-size="2.8" fill="#74808d">'
                        'Unrated; separate sealed ID reveal and blank owner preference ballot. '
                        'NOT native CyberSand.</text></svg>')
        (output/f"blind-a4-{sheet+1:02d}.svg").write_text("\n".join(graphics)+"\n")


def run(request, binary, q8, q4, w0, legacy, receipt, evidence, source_commit):
    pins.source_guard(source_commit)
    req=request_validation(json.loads(Path(request).read_text()))
    for quant, path, sha, nbytes in (("q8",q8,Q8_SHA,Q8_BYTES),
                                     ("q4",q4,pins.Q4_SHA,pins.Q4_BYTES)):
        if Path(path).stat().st_size!=nbytes or pins.sha256_file(path)!=sha:
            raise ValueError(f"Frozen {quant} source model identity changed")
    evidence=Path(evidence)
    if evidence.exists():
        raise ValueError("Will not overwrite a past experiment")
    pins._add_legacy_to_import_paths(legacy)
    from experiments.driv3r.frankenlayers import infer as paired
    from experiments.driv3r.frankenlayers.patch_runtime import FLOWS
    from experiments.driv3r.frankenlayers_lab.flows import LAB_FLOWS
    from experiments.driv3r.frankenlayers_lab import wave20_structural as w20
    pin_validation()
    patch=json.loads(Path(receipt).read_text())
    if patch["llama_commit"]!=PINNED_LLAMA or patch["base_patch"]["lab_flows"]!={
            k:list(v) for k,v in LAB_FLOWS.items()}:
        raise ValueError("Wrong compiled source patch or layer names")
    if any(k in FLOWS for k in LAB_FLOWS) or any(k in FLOWS for k in pins.EXTRA_FLOWS):
        raise ValueError("Overwriting an existing flow registry")
    FLOWS.update(LAB_FLOWS)
    FLOWS.update(pins.EXTRA_FLOWS)
    host=paired.load_w0(w0)
    evidence.mkdir(parents=True)
    _json(evidence/"provenance.json",{
        "schema":"driv3r/matched-q8-q4-expressive/1",
        "source_commit":source_commit,"workflow_request":req,
        "llama_commit":PINNED_LLAMA,"binary_sha256":pins.sha256_file(binary),
        "graph_patch_sha256":pins.sha256_file(receipt),
        "official_q8_sha256":Q8_SHA,"official_q4_sha256":pins.Q4_SHA,
        "legacy_w0_sha256":pins.sha256_file(w0),
        "model_comparison":"same initial W0 world/seed, prompts, mode, pinned source-built CPU runtime",
        "after_first_divergence":"observations and legal menus can legitimately diverge",
        "decoder_comparison":"common uniform for Q8/Q4 sampled arms, no post-hoc force-turn rules",
        "host":"historical synthetic W0, not CyberSand native",
        "owner_preferences":"NO human ratings yet"})
    rows=[]
    folds=req["folds"]
    steps=req["steps_per_episode"]
    # Model execution is sequential, protecting resident memory on 4-vCPU runners.
    with w20.rss_instrument(paired):
        for seed,family in folds:
            for mode in req["modes"]:
                for decoder in DECODERS:
                    for quant,model in (("q8",q8),("q4",q4)):
                        prior=(paired.SEED,paired.STEPS,paired.FAMILIES)
                        paired.SEED,paired.STEPS,paired.FAMILIES =seed,steps,(family,)
                        case=evidence/"cases"/f"{seed}-{family}"/mode/decoder/quant
                        entry={"seed":seed,"family":family,"mode":mode,"quant":quant,
                               "decoder":decoder,"status":"FAILED",
                               "request_nonce":req["nonce"],"expected_budget":steps}
                        try:
                            with decoder_proxy(paired,decoder):
                                arm=paired.run_arm(binary,model,host,mode,case)
                            calls=json.loads((case/f"{mode}-{family}-choices.json").read_text())
                            world=json.loads((case/f"{mode}-{family}.json").read_text())
                            metrics=describe_path(world,calls)
                            entry["status"]="EXECUTED"
                            entry["metrics"]=metrics
                            entry["model_decisions"]=arm["model_decisions"]
                            if metrics["square_loop_collapse"] or metrics["stop_early"]:
                                entry["aesthetic_disposition"]="FAIL_REPETITIVE_OR_EARLY_STOP"
                            elif metrics["new_cell_fraction"]<.25:
                                entry["aesthetic_disposition"]="LOW_EXPLORATION"
                            else:
                                entry["aesthetic_disposition"]="CANDIDATE_NOT_OWNER_RATED"
                            anonymous=hashlib.sha256(
                                f"blind:{req['nonce']}:{quant}:{mode}:{decoder}:{seed}:{family}".encode()
                                ).hexdigest()[:9].upper()
                            entry["anonymous_id"]=anonymous
                            (case/"original-ink-path.svg").write_text(svg_for_ink(world,anonymous))
                            _json(case/"original-trajectory.json",world["trajectory"])
                            entry["source_relpath"]=str((case/"original-ink-path.svg").relative_to(evidence))
                        except Exception:
                            entry["error"]=traceback.format_exc()
                        finally:
                            paired.SEED,paired.STEPS,paired.FAMILIES=prior
                            case.mkdir(parents=True,exist_ok=True)
                            _json(case/"case-summary.json",entry)
                            rows.append(entry)
                            _json(evidence/"all-case-metrics.json",rows)
    valid=[r for r in rows if r["status"]=="EXECUTED"]
    build_blind_gallery(valid,evidence,req["phase"])
    control_valid=[r for r in rows if r["mode"]=="stock" and r["decoder"]=="greedy"
                   and r["status"]=="EXECUTED"]
    q8_continuations=[r for r in control_valid if r["quant"]=="q8" and
                      r["metrics"]["decisions"]>=min(16,steps)]
    report={"schema":"driv3r/matched-q8-q4-campaign-summary/1",
            "source_commit":source_commit,"phase":req["phase"],"nonce":req["nonce"],
            "cases":len(rows),"executed":len(valid),"failed":len(rows)-len(valid),
            "stock_q8_continued_16_of_40":len(q8_continuations),
            "stock_q8_control_count":len([r for r in control_valid if r["quant"]=="q8"]),
            "early_stops":sum(r["metrics"]["stop_early"] for r in valid),
            "four_cell_square_loop_cases":sum(r["metrics"]["square_loop_collapse"] for r in valid),
            "cycles_2_16":sum(r["metrics"]["cycle_2_16"]["period"] is not None for r in valid),
            "mean_new_cell_fraction":{q:statistics.mean(
                [r["metrics"]["new_cell_fraction"] for r in valid if r["quant"]==q])
                if any(r["quant"]==q for r in valid) else None for q in QUANTS},
            "native_qualified":False,"niche_rankings":"UNRATED",
            "result_gate":"NO_AUTOMATIC_LARGE_RUN_UNTIL_OWNER_AND_EVIDENCE_REVIEW",
            "status":"QUALIFIED_CASES_WITH_NEGATIVE_RESULTS_PRESERVED"}
    _json(evidence/"campaign-summary.json",report)
    print(json.dumps(report,sort_keys=True),flush=True)
    if len(valid)<len(rows):
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    r=sub.add_parser("run")
    for attr in ("request","binary","model-q8","model-q4","w0","legacy-root","patch-receipt","evidence"):
        r.add_argument("--"+attr,required=True,type=Path)
    r.add_argument("--source-commit",required=True)
    a=p.parse_args()
    if a.cmd=="run":
        run(a.request,a.binary,a.model_q8,a.model_q4,a.w0,a.legacy_root,
            a.patch_receipt,a.evidence,a.source_commit)


if __name__=="__main__":
    main()
