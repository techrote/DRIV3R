"""Opt-in model-zoo experiment: real pretrained Falcon3 BitNet -> 7-action head.

All worlds/legality are synthetic host fixtures, NEVER native CyberSand.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, statistics, time
from pathlib import Path

MODEL="tiiuae/Falcon3-1B-Instruct-1.58bit"
PIN="72fd3f95fcd82639c902304919629edda8c6f2b4"
SCHEMA="MZ019-FALCON3-BITNET-v1"
SYSTEM=("You control a 2D painting worm among blocked and protected material. "
        "Use only host offered legal moves. Navigate toward goal, branch or retrace.")

def validate_budget(a):
    if not (4<=a.train_seeds<=32 and 2<=a.eval_seeds<=12 and
            4<=a.steps<=16 and 1<=a.updates<=160 and
            4<=a.horizon<=32 and 1<=a.batch<=32):
        raise ValueError("invalid/unbounded training budget")

def legal_choice(scores, actions, menu):
    offered=[a for a in actions if a in menu]
    if not offered: raise ValueError("no host legal actions")
    return max(offered,key=lambda a:float(scores[actions.index(a)]))

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()

def pct(xs,p):
    return round(sorted(xs)[int((len(xs)-1)*p)],4) if xs else None

def verify_sets(parts):
    seeds=[{r[2] for r in p} for p in parts]
    if any(seeds[i]&seeds[j] for i,j in ((0,1),(0,2),(1,2))):
        raise RuntimeError("TRAIN/DEV/TEST seed leak")

def execute(a):
    validate_budget(a)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import snapshot_download,model_info
    from experiments.driv3r.model_zoo.mamba130.counterfactual import (
        samples,raw_features,paired_switch_metrics)
    from experiments.driv3r.model_zoo.mamba130.world import ACTIONS,World
    from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure
    torch.manual_seed(19019);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if model_info(MODEL,revision=PIN).sha!=PIN:
        raise RuntimeError("upstream revision mismatch")
    t=time.monotonic()
    directory=Path(snapshot_download(MODEL,revision=PIN,
                            allow_patterns=["*.json","*.safetensors"]))
    files=sorted(directory.glob("*.safetensors"))
    if not files:raise RuntimeError("immutable pretrained weights missing")
    hashes={p.name:sha256_file(p) for p in files}
    tok=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,
                                       trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(
        str(directory),local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32)
    if model.config.model_type!="llama" or not getattr(model.config,"is_bitnet_config",False):
        raise RuntimeError("not Falcon3 BitNet architecture")
    q=getattr(model.config,"quantization_config",{})
    if not isinstance(q,dict) or q.get("quant_method")!="bitnet":
        raise RuntimeError("unexpected non-BitNet model quantization")
    parameters=sum(p.numel() for p in model.parameters())
    if not 600_000_000<=parameters<=1_200_000_000:
        raise RuntimeError("unexpected trained pretrained-model parameter count")
    load_s=time.monotonic()-t
    model.eval()
    for p in model.parameters():p.requires_grad_(False)

    train=samples(range(400,400+a.train_seeds),a.steps)
    dev=samples(range(900,900+a.eval_seeds),a.steps)
    test=samples(range(1200,1200+a.eval_seeds),a.steps)
    verify_sets((train,dev,test))
    if paired_switch_metrics(test,[r[1] for r in test])["changed_target_pairs"]<8:
        raise RuntimeError("insufficient changed-goal held-out pairs")
    cache={};timings=[];lengths=[]
    def vector(percept):
        if percept not in cache:
            text=tok.apply_chat_template(
                [{"role":"system","content":SYSTEM},
                 {"role":"user","content":percept}],
                tokenize=False,add_generation_prompt=True)
            tokens=tok(text,return_tensors="pt",truncation=False)
            n=int(tokens["input_ids"].shape[1])
            if not 8<=n<=256:raise RuntimeError("context truncation would be needed")
            lengths.append(n)
            start=time.perf_counter()
            with torch.inference_mode():
                # Actual pretrained hidden-state forward; never random proxy features.
                states=model.model(**tokens,use_cache=False,return_dict=True).last_hidden_state
            timings.append((time.perf_counter()-start)*1000)
            v=states[0,-1].detach().float().cpu().clone()
            if not torch.isfinite(v).all():raise RuntimeError("nonfinite pretrained feature")
            cache[percept]=v
        return cache[percept]
    X=torch.stack([vector(r[0]) for r in train])
    D=torch.stack([vector(r[0]) for r in dev])
    T=torch.stack([vector(r[0]) for r in test])
    FX=torch.tensor([raw_features(r[0]) for r in train],dtype=torch.float32)
    FD=torch.tensor([raw_features(r[0]) for r in dev],dtype=torch.float32)
    FT=torch.tensor([raw_features(r[0]) for r in test],dtype=torch.float32)
    y=torch.tensor([r[1] for r in train],dtype=torch.long)
    def head(d):
        return torch.nn.Sequential(torch.nn.LayerNorm(d),torch.nn.Linear(d,64),
                                   torch.nn.GELU(),torch.nn.Linear(64,len(ACTIONS)))
    torch.manual_seed(19019)
    actual=head(X.shape[-1]);cheap=head(FX.shape[-1])
    opt0=torch.optim.AdamW(actual.parameters(),lr=.002)
    opt1=torch.optim.AdamW(cheap.parameters(),lr=.002)
    hist=torch.bincount(y,minlength=len(ACTIONS)).float()
    weights=torch.where(hist>0,(hist.sum()/hist.clamp(min=1)).sqrt(),0.)
    weights/=weights[weights>0].mean()
    generator=torch.Generator().manual_seed(19019)
    losses=[[],[]]
    for _ in range(a.updates):
        ids=torch.randint(len(y),(min(a.batch,len(y)),),generator=generator)
        for i,(net,opt,data) in enumerate(((actual,opt0,X),(cheap,opt1,FX))):
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(net(data[ids]),y[ids],weight=weights)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite optimizer loss")
            loss.backward();opt.step();losses[i].append(float(loss.detach()))
    actual.eval();cheap.eval()
    with torch.inference_mode():
        ad=actual(D).argmax(-1).tolist();at=actual(T).argmax(-1).tolist()
        bd=cheap(FD).argmax(-1).tolist();bt=cheap(FT).argmax(-1).tolist()
    majority=int(torch.bincount(y).argmax())
    scores={"falcon3":{"dev":measure(dev,ad),"test":measure(test,at)},
            "structured":{"dev":measure(dev,bd),"test":measure(test,bt)},
            "majority":{"test":measure(test,[majority]*len(test))}}
    trajectories=[]
    for seed in range(1807,1811):
        for kind in ("falcon3","structured","random-nonstop","teacher"):
            world=World.make(seed);rng=random.Random(seed*101+17);moves=[];raw_illegal=0
            for _ in range(a.horizon):
                menu=world.legal()
                if kind=="teacher":
                    raw=world.teacher(rng);action=raw
                elif kind=="random-nonstop":
                    raw=rng.choice([x for x in menu if x!="STOP"] or ["STOP"]);action=raw
                else:
                    feature=vector(world.prompt()) if kind=="falcon3" else torch.tensor(
                        raw_features(world.prompt()),dtype=torch.float32)
                    with torch.inference_mode():
                        logits=(actual if kind=="falcon3" else cheap)(feature[None])[0]
                    raw=ACTIONS[int(logits.argmax())]
                    raw_illegal+=int(raw not in menu)
                    action=legal_choice(logits,ACTIONS,menu)
                moves.append({"turn":world.turns,"position":world.pos,"raw":raw,
                              "legal":list(menu),"selected":action})
                world.apply(action)
                if world.stopped:break
            trajectories.append({"seed":seed,"policy":kind,"steps":moves,
                "raw_illegal":raw_illegal,"painted":len(world.painted),
                "stopped":world.stopped})
    torch.save({"schema":SCHEMA,"upstream_revision":PIN,"updates":a.updates,
        "trained_head":actual.state_dict(),"structured_control":cheap.state_dict()},
        out/"heads.pt")
    manifest={"schema":SCHEMA,"source_head":os.getenv("GITHUB_SHA","LOCAL-UNVERIFIED"),
        "upstream_model":MODEL,"upstream_revision":PIN,
        "weight_sha256":hashes,"parameters":parameters,"load_seconds":round(load_s,3),
        "training_updates_per_arm":a.updates,"scores":scores,
        "final_losses":[l[-1] for l in losses],
        "cpu_forward_p50_ms":pct(timings,.5),"cpu_forward_p95_ms":pct(timings,.95),
        "context_p50_tokens":pct(lengths,.5),"context_max_tokens":max(lengths),
        "peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "head_sha256":sha256_file(out/"heads.pt"),"synthetic_host_only":True,
        "native_cybersand_acceptance":False}
    (out/"metrics.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"trajectories.json").write_text(json.dumps(trajectories,indent=2)+"\n")
    (out/"train_losses.json").write_text(json.dumps(losses,indent=2)+"\n")
    print(json.dumps({"status":"trained","metrics":str(out/"metrics.json")}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--train-seeds",type=int,default=14)
    p.add_argument("--eval-seeds",type=int,default=4)
    p.add_argument("--steps",type=int,default=9)
    p.add_argument("--updates",type=int,default=64)
    p.add_argument("--horizon",type=int,default=16)
    p.add_argument("--batch",type=int,default=12)
    execute(p.parse_args())
if __name__=="__main__":main()
