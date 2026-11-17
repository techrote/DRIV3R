"""P20: opt-in real SmolLM2-135M LoRA population for synthetic CyberSand-like actions.

No toy policy is trained here. Nothing runs on import, push or PR. Requires an
offline copy of the EXACT pinned pretrained root, validated against a caller-
supplied SHA-256; never downloads or mutates CyberSand. Synthetic P19 host
is a REFERENCE FIXTURE, not the native CyberSand action/observation ABI.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import resource
import statistics
import time
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
NICHES = ("navigator", "branch_artist", "recoverer", "texture_worm",
          "material_cautious", "chaotic_brush")
ACTIONS = ("N", "E", "S", "W", "RETRACE", "STOP")
TRAIN_CARDS = (0, 1, 2, 3)
HOLDOUT_CARDS = (4, 5)
MAX_UPDATES = 12

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def envelope(niches, seeds, updates, horizon):
    if not 1 <= len(niches) <= 6 or len(set(niches)) != len(niches):
        raise ValueError("unique niches required")
    if any(n not in NICHES for n in niches):
        raise ValueError("unknown niche")
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("1..3 distinct seeds")
    if not 1 <= updates <= MAX_UPDATES or not 24 <= horizon <= 48:
        raise ValueError("bounded CPU training/rollout envelope")
    return True

def root_preflight(root, expected_sha256, *, require_packages=True):
    import importlib.util
    result = {"model_id": MODEL_ID, "revision": REVISION,
              "actual_backbone_forwards": 0, "actual_optimizer_updates": 0,
              "status": "BLOCKED", "failures": []}
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in expected_sha256):
        result["failures"].append("missing independently supplied 64-char lowercase model SHA-256")
    path = Path(root)
    weights = path / "model.safetensors"
    if not path.is_dir():
        result["failures"].append("pinned pretrained model directory missing")
    else:
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            if not (path / name).is_file():
                result["failures"].append("missing "+name)
        if weights.is_file():
            result["model_safetensors_sha256"] = digest(weights)
            if result["model_safetensors_sha256"] != expected_sha256:
                result["failures"].append("pinned model checksum mismatch")
    if require_packages:
        for name in ("torch", "transformers", "safetensors", "tokenizers"):
            if importlib.util.find_spec(name) is None:
                result["failures"].append("missing package "+name)
    result["status"] = "READY" if not result["failures"] else "BLOCKED"
    return result

def legal_mask(legal):
    return [int(a in legal) for a in ACTIONS]

def choose(logits, legal):
    if not legal or any(a not in ACTIONS for a in legal):
        raise ValueError("host action menu invalid")
    raw = ACTIONS[max(range(len(ACTIONS)), key=lambda i: float(logits[i]))]
    safe = max(legal, key=lambda a: (float(logits[ACTIONS.index(a)]), -ACTIONS.index(a)))
    return safe, int(raw not in legal)

def render_prompt(world, niche):
    if niche not in NICHES:
        raise ValueError(niche)
    legal = world.legal()
    # Source-host contains no semantic material types. Protected cells act as
    # a coarse material safety surrogate; don't mislabel this native sensing.
    return ("CyberSand-like synthetic painting policy. Style="+niche+
            "; cursor="+str(world.cursor)+"; goal="+str(world.goal)+
            "; step="+str(world.steps)+"/"+str(world.max_steps)+
            "; goal_switched="+str(int(world.switched))+
            "; painted_count="+str(len(world.painted))+
            "; protected_adj="+str(world.neighbours(world.cursor,world.protected))+
            "; blocked_adj="+str(world.neighbours(world.cursor,world.obstacles))+
            "; parent_available="+str(int(world.cursor in world.parent))+
            "; legal="+",".join(legal)+"; choose one legal action.")

def splits(seed):
    # Four seen cards versus two held-out geometry families, all seed streams
    # disjoint. Selection can reuse TRAIN families but never TRAIN identities.
    from population_lab import stable_seed
    return {
        "train": [(stable_seed("p20",seed,"train",i), TRAIN_CARDS[i%4]) for i in range(8)],
        "selection": [(stable_seed("p20",seed,"selection",i), TRAIN_CARDS[(i+1)%4]) for i in range(4)],
        "heldout": [(stable_seed("p20",seed,"heldout",i), HOLDOUT_CARDS[i%2]) for i in range(4)]
    }

def teacher_rows(niche, worlds, horizon):
    import numpy as np
    from population_lab import utility
    from population_p19_viable_novelty import scenario
    rows=[]
    for seed, card in worlds:
        world=scenario(seed,card,horizon)
        rng=random.Random(seed)
        for t in range(horizon):
            legal=world.legal()
            scores=[utility(niche,world.feature(a),world) for a in legal]
            best=max(range(len(legal)),key=lambda j:(scores[j],-j))
            rows.append({"text":render_prompt(world,niche),"label":ACTIONS.index(legal[best]),
                         "legal":legal,"identity":[seed,card,t]})
            # Teacher-directed with bounded exploratory variation in TRAIN only;
            # this records actual state transitions, not isolated labels.
            choice = best if rng.random()<.78 else rng.randrange(len(legal))
            action=legal[choice]
            world.step(action)
            if action=="STOP": break
    return rows

def inject_lora(backbone, torch, rank=4):
    from torch import nn
    from torch.nn import functional as F
    class LowRank(nn.Module):
        def __init__(self, linear):
            super().__init__()
            if not isinstance(linear,nn.Linear):raise TypeError("linear projection expected")
            self.base=linear.requires_grad_(False)
            self.A=nn.Parameter(torch.empty(rank,linear.in_features))
            self.B=nn.Parameter(torch.zeros(linear.out_features,rank))
            nn.init.kaiming_uniform_(self.A,a=5**.5)
            self.scale=8/rank
        def forward(self,x):
            return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale
    layers=getattr(backbone,"layers",None)
    if layers is None:layers=getattr(getattr(backbone,"model",None),"layers",None)
    if layers is None or len(layers)<2:raise RuntimeError("unknown SmolLM2 layer layout")
    names=[]
    for index in range(len(layers)-2,len(layers)):
        for name in ("q_proj","v_proj"):
            module=layers[index].self_attn
            setattr(module,name,LowRank(getattr(module,name)))
            names.append("layers.%d.self_attn.%s"%(index,name))
    return names

def adapters(backbone):
    return {n:p for n,p in backbone.named_parameters()
            if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))}

def encode(tok, texts):
    return tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=128)

def states(torch,backbone,tokens):
    h=backbone(**tokens,use_cache=False).last_hidden_state
    pos=tokens["attention_mask"].sum(dim=-1)-1
    return h[torch.arange(len(pos)),pos].float()

def fit_arm(torch,backbone,tok,head,rows,batches,*,train_lora):
    from torch.nn import functional as F
    params=list(head.parameters())+ (list(adapters(backbone).values()) if train_lora else [])
    opt=torch.optim.AdamW(params,lr=.0005 if train_lora else .003)
    history=[]
    for ids in batches:
        batch=[rows[i] for i in ids]
        tokens=encode(tok,[r["text"] for r in batch])
        mask=torch.tensor([legal_mask(r["legal"]) for r in batch],dtype=torch.bool)
        labels=torch.tensor([r["label"] for r in batch])
        if not bool(mask.gather(1,labels[:,None]).all()):raise AssertionError("illegal teacher")
        if train_lora:
            x=states(torch,backbone,tokens)
        else:
            with torch.no_grad():x=states(torch,backbone,tokens)
        scores=head(x).masked_fill(~mask,-1e6)
        loss=F.cross_entropy(scores,labels)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
        history.append(round(float(loss.detach()),6))
    return history

def rollout(torch,backbone,tok,head,niche,seed,card,horizon):
    from population_p19_viable_novelty import scenario
    from population_lab import utility
    world=scenario(seed,card,horizon)
    actions=[];illegal=0;times=[];reward=0.
    for _ in range(horizon):
        legal=world.legal()
        tokens=encode(tok,[render_prompt(world,niche)])
        started=time.perf_counter()
        with torch.inference_mode():
            scores=head(states(torch,backbone,tokens))[0]
        times.append((time.perf_counter()-started)*1000)
        choice,raw_illegal=choose(scores.tolist(),legal)
        illegal+=raw_illegal
        reward+=float(utility(niche,world.feature(choice),world))
        actions.append(choice)
        world.step(choice)
        if choice=="STOP":break
    if world.painted & (world.protected|world.obstacles):
        raise AssertionError("host safety invariant")
    return {"seed":seed,"card":card,"niche":niche,"reward_proxy":round(reward,5),
            "steps":world.steps,"early_stop":int(world.stopped and world.steps<horizon),
            "goal_switched":int(world.switched),
            "painted":len(world.painted)-1,"retraces":actions.count("RETRACE"),
            "raw_illegal_preferences":illegal,"executed_illegal":0,
            "descriptor":world.descriptor(),"actions":actions,
            "latency_ms":times}

def evaluation(torch,backbone,tok,head,niche,pairs,horizon):
    rows=[rollout(torch,backbone,tok,head,niche,seed,card,horizon)
          for seed,card in pairs]
    return {"episodes":len(rows),"mean_reward_proxy":round(statistics.mean(r["reward_proxy"] for r in rows),5),
            "mean_steps":round(statistics.mean(r["steps"] for r in rows),3),
            "early_stop_rate":round(statistics.mean(r["early_stop"] for r in rows),4),
            "switch_rate":round(statistics.mean(r["goal_switched"] for r in rows),4),
            "mean_painted":round(statistics.mean(r["painted"] for r in rows),3),
            "raw_illegal":sum(r["raw_illegal_preferences"] for r in rows),
            "actions_executed":sum(r["steps"] for r in rows),
            "rows":rows}

def manifest(output):
    return {str(p.relative_to(output)):digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name!="SHA256SUMS.json"}

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True,help="offline pinned pretrained SmolLM2 directory")
    ap.add_argument("--expected-root-sha256",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--niches",default="navigator,recoverer")
    ap.add_argument("--seeds",default="690")
    ap.add_argument("--updates",type=int,default=4)
    ap.add_argument("--horizon",type=int,default=32)
    ap.add_argument("--preflight-only",action="store_true")
    a=ap.parse_args(argv)
    niches=tuple(NICHES if a.niches=="all" else a.niches.split(","))
    seeds=tuple(int(s) for s in a.seeds.split(","))
    envelope(niches,seeds,a.updates,a.horizon)
    check=root_preflight(a.root,a.expected_root_sha256)
    if a.preflight_only:
        print(json.dumps(check,sort_keys=True,indent=2))
        return 0 if check["status"]=="READY" else 2
    if check["status"]!="READY":
        print(json.dumps(check,sort_keys=True,indent=2))
        return 2
    if a.output.exists():ap.error("immutable output directory already exists")
    import torch
    from torch import nn
    from transformers import AutoModel,AutoTokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    started=time.perf_counter()
    # Explicit offline loading, no HF requests and no trust_remote_code.
    tok=AutoTokenizer.from_pretrained(a.root,local_files_only=True,trust_remote_code=False)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    backbone=AutoModel.from_pretrained(a.root,local_files_only=True,
                                     trust_remote_code=False,use_safetensors=True)
    backbone.eval().requires_grad_(False)
    injected=inject_lora(backbone,torch)
    if len(adapters(backbone))!=8:raise AssertionError("four LoRA modules must have eight tensors")
    a.output.mkdir(parents=True)
    (a.output/"checkpoints").mkdir()
    report={"schema":"driv3r.population.p20.real_slm/1","status":"RUNNING",
            "synthetic_only":True,"native_cybersand_receipts":0,
            "model":MODEL_ID,"revision":REVISION,"root_sha256":check["model_safetensors_sha256"],
            "backbone_parameters":sum(p.numel() for p in backbone.parameters()),
            "lora_modules":injected,"niches":niches,"seeds":seeds,
            "train_cards":TRAIN_CARDS,"heldout_cards":HOLDOUT_CARDS,
            "actual_root_forwards":0,"actual_lora_updates":0,
            "arm_results":[],"disclosure":"host masking forces action validity; reward is a hand-designed proxy"}
    try:
        for seed in seeds:
            split=splits(seed)
            assert not set(split["train"]) & set(split["selection"])
            assert not set(split["heldout"]) & (set(split["train"])|set(split["selection"]))
            for niche in niches:
                train=teacher_rows(niche,split["train"],a.horizon)
                identity=hashlib.sha256(json.dumps([r["identity"] for r in train]).encode()).hexdigest()
                rng=random.Random(seed)
                batches=[[rng.randrange(len(train)) for _ in range(2)] for __ in range(a.updates)]
                for arm in ("frozen_head","lora"):
                    torch.manual_seed(seed*100+NICHES.index(niche))
                    for name,p in adapters(backbone).items():
                        with torch.no_grad():
                            if name.endswith(".B"):p.zero_()
                            else:torch.nn.init.kaiming_uniform_(p,a=5**.5)
                    head=nn.Linear(backbone.config.hidden_size,len(ACTIONS))
                    history=fit_arm(torch,backbone,tok,head,train,batches,train_lora=arm=="lora")
                    report["actual_root_forwards"]+=a.updates
                    report["actual_lora_updates"]+=a.updates if arm=="lora" else 0
                    torch.nn.utils.clip_grad_norm_(list(head.parameters()),1.)
                    head.eval().requires_grad_(False)
                    # Both selection and heldout are complete closed-loop rollouts.
                    dev=evaluation(torch,backbone,tok,head,niche,split["selection"],a.horizon)
                    hold=evaluation(torch,backbone,tok,head,niche,split["heldout"],a.horizon)
                    tensors={"head.weight":head.weight.detach().cpu().contiguous(),
                             "head.bias":head.bias.detach().cpu().contiguous()}
                    if arm=="lora":
                        tensors.update({k:p.detach().cpu().contiguous() for k,p in adapters(backbone).items()})
                    name=f"{seed}-{niche}-{arm}"
                    file=a.output/"checkpoints"/(name+".safetensors")
                    save_file(tensors,str(file))
                    row={"id":name,"niche":niche,"seed":seed,"arm":arm,
                         "checkpoint":file.name,"checkpoint_sha256":digest(file),
                         "train_identity_sha256":identity,
                         "train_rows":len(train),"updates":a.updates,"loss_history":history,
                         "selection":dev,"heldout":hold,
                         "lineage":"independent same-seed head/LoRA initialization; no offspring crossover"}
                    report["arm_results"].append(row)
        report["status"]="COMPLETED"
    except BaseException as e:
        report["status"]="FAILED"
        report["failure"]=repr(e)
        raise
    finally:
        report["wall_seconds"]=round(time.perf_counter()-started,3)
        report["peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (a.output/"SHA256SUMS.json").write_text(json.dumps(manifest(a.output),indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"checkpoint_count":len(report["arm_results"]),
                      "lora_updates":report["actual_lora_updates"],"wall_seconds":report["wall_seconds"]}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
