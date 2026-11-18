"""FT-021: checkpoint-anchored CPU training residency qualification.

Four genuine FT-020 trained 20x24x1 tiny policies are independently continued
on shifted synthetic CyberSand-like legal action menus. Compare 4 processes x
1 model versus 2 processes x 2 models, preserving identical update streams.
This is synthetic policy research, not native CyberSand ABI or a pretrained SLM.
Explicit invocation only. No network or game writes; checkpoint files stay outside Git.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ID = "FT021-20261116-training-residency"
SCHEMA = "driv3r.fleet_training.ft021/1"
NICHES = ("navigator", "recoverer", "branch_artist", "material_cautious")
FEATURES = ("bias", "paint", "retrace", "stop", "goal_progress", "onward", "blocked", "junction", "turn", "frontier", "material_edge", "paint_density", "early", "protected_near", "goal_switched", "direction_x", "direction_y", "guidance", "motif", "recovery_need")
PARENTS = {
    "navigator": "1debec1ad87fd49a6e8304d875ab766fa0e72ea3dca28ab68846beff12f2b5af",
    "recoverer": "22b99c3b807e2c161d9e47acb6e317658c4db41a783eb97aa5c7b92d82d44310",
    "branch_artist": "5493d1cbc0df1c18da10bb0b1ab6519e42d383e120ff78a0a9ddd3645d8e3fc7",
    "material_cautious": "4a763d88fced6831e1327ae1227e983e4bb6d61f9c00656f2e1d1a5c95215cac",
}
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
MAX_UPDATES = 1200
MAX_WALL_S = 120
MAX_RSS_KIB = 450_000
MAX_SUM_RSS_KIB = 1_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**32)


def softmax(a):
    z = np.exp(np.clip(a-np.max(a), -45, 0))
    return z/z.sum()


def menus(niche, split, count):
    """Held-out shifted one-step legal menus; STOP is always an available option.

    Dense blocked regions and permanent protected-material masks are built from
    independent seeds. RETRACE is synthetic and may not be a real host history
    edge, so the harness must not claim native action-menu conformance.
    """
    if split not in ("train", "heldout") or niche not in NICHES or not 1 <= count <= 512:
        raise ValueError("Invalid menu lease")
    rng = np.random.default_rng(seed(ID, niche, split))
    result = []
    for _ in range(count):
        protected = {(x, y) for x in range(8, 14) for y in range(8, 14)}
        blocked = {(int(x), int(y)) for x,y in rng.integers(1, 21, size=(70 if split == "heldout" else 45, 2))}
        free = [(x,y) for x in range(1,21) for y in range(1,21) if (x,y) not in protected | blocked]
        x,y = free[int(rng.integers(len(free)))]
        gx,gy = free[int(rng.integers(len(free)))]
        switched = int(rng.random() < (.75 if split == "heldout" else .4))
        heading = tuple(DIRECTIONS)[int(rng.integers(4))]
        occupied = protected | blocked
        options = []
        for direction,(dx,dy) in DIRECTIONS.items():
            a = (x+dx,y+dy)
            if 1 <= a[0] <= 20 and 1 <= a[1] <= 20 and a not in occupied:
                options.append((direction,a,(dx,dy)))
        if rng.random() < .65:
            # RETRACE is possible, but a parent's existence is not established.
            options.append(("RETRACE",(x,y),(0,0)))
        options.append(("STOP",(x,y),(0,0)))
        X = []
        for name,after,(dx,dy) in options:
            adj=[(after[0]+vx,after[1]+vy) for vx,vy in DIRECTIONS.values()]
            onward=sum(1<=ax<=20 and 1<=ay<=20 and (ax,ay) not in occupied for ax,ay in adj)
            nblocked=sum(v in blocked for v in adj)
            nprotected=sum(v in protected for v in adj)
            dist0=abs(x-gx)+abs(y-gy)
            dist1=abs(after[0]-gx)+abs(after[1]-gy)
            retrace=int(name=="RETRACE")
            paint=int(name in DIRECTIONS)
            X.append((1.,paint,retrace,int(name=="STOP"),
                (dist0-dist1)/2.,onward/4.,nblocked/4.,int(onward>=3),
                int(paint and heading!=name),int(onward>=2),int(nprotected>0),
                rng.random(),rng.random(),nprotected/4.,switched,float(dx),float(dy),
                int(dx*np.sign(gx-x)+dy*np.sign(gy-y)>0),rng.random(),
                retrace*int(onward>1)))
        data=np.asarray(X,dtype=np.float64)
        assert data.shape == (len(options),20) and np.isfinite(data).all()
        result.append(([o[0] for o in options],data))
    return result


def shifted_teacher(niche,X):
    c = {n:X[:,i] for i,n in enumerate(FEATURES)}
    base = .12*c["paint"] - .7*c["stop"]
    if niche=="navigator":
        return base+2.2*c["goal_progress"]+.7*c["onward"]+.7*c["goal_switched"]*c["goal_progress"]
    if niche=="recoverer":
        return base+1.5*c["retrace"]*c["recovery_need"]+1.3*c["onward"]-1.0*c["blocked"]
    if niche=="branch_artist":
        return base+.9*c["junction"]+.6*c["frontier"]+.9*c["turn"]+.4*c["motif"]
    return base+1.6*c["goal_progress"]-2.0*c["protected_near"]-1.2*c["blocked"]


class Policy:
    def __init__(self, params):
        self.p={k:np.array(v,dtype=np.float64) for k,v in params.items()}
        assert {k:v.shape for k,v in self.p.items()} == {"W1":(20,24),"b1":(24,),"W2":(24,),"b2":(1,)}
        assert all(np.isfinite(v).all() for v in self.p.values())
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.updates=0

    def logits(self,X):
        return np.tanh(X@self.p["W1"]+self.p["b1"])@self.p["W2"]+self.p["b2"][0]

    def teach(self,X,target):
        H=np.tanh(X@self.p["W1"]+self.p["b1"])
        probs=softmax(H@self.p["W2"]+self.p["b2"][0])
        d=probs-target
        dH=d[:,None]*self.p["W2"][None,:]*(1-H*H)
        grad={"W1":X.T@dH,"b1":dH.sum(axis=0),"W2":H.T@d,"b2":np.array([d.sum()])}
        self.updates+=1
        for k,g in grad.items():
            self.m[k]=.9*self.m[k]+.1*g
            self.v[k]=.999*self.v[k]+.001*g*g
            mh=self.m[k]/(1-.9**self.updates)
            vh=self.v[k]/(1-.999**self.updates)
            self.p[k]-=.007*mh/(np.sqrt(vh)+1e-8)
        return -float((target*np.log(np.maximum(probs,1e-12))).sum())


def validate_parent(root,niche):
    p = Path(root)/(niche+".json")
    if sha(p)!=PARENTS[niche]:
        raise ValueError("Original FT020 parent SHA-256 mismatch: "+niche)
    obj=json.loads(p.read_text())
    if obj.get("niche")!=niche or obj.get("schema")!="population-mlp-v1" or obj.get("features")!=list(FEATURES) or obj.get("adam_updates")!=180:
        raise ValueError("FT020 parent model metadata drift")
    return Policy(obj["parameters"])


def metrics(policy,niche,cases):
    correct=0
    ce=0.
    for _,X in cases:
        y=int(np.argmax(shifted_teacher(niche,X)))
        prob=softmax(policy.logits(X))
        correct+=int(np.argmax(prob)==y)
        ce-=float(np.log(max(prob[y],1e-12)))
    return {"accuracy":round(correct/len(cases),6),"cross_entropy":round(ce/len(cases),6)}


def train_one(root,output,niche,updates):
    p=validate_parent(root,niche)
    train=menus(niche,"train",192)
    heldout=menus(niche,"heldout",128)
    initial=metrics(p,niche,heldout)
    rng=np.random.default_rng(seed(ID,"updates",niche))
    losses=[]
    for _ in range(updates):
        _,X=train[int(rng.integers(len(train)))]
        target=softmax(shifted_teacher(niche,X)/.42)
        losses.append(p.teach(X,target))
    assert p.updates==updates
    dest=Path(output)/(niche+".json")
    dest.write_text(json.dumps({"schema":SCHEMA,"niche":niche,"parent_sha256":PARENTS[niche],"new_adam_updates":updates,
         "parameters":{k:v.tolist() for k,v in p.p.items()}},sort_keys=True,separators=(",",":"))+"\n")
    return {"niche":niche,"model_sha256":sha(dest),"parent_sha256":PARENTS[niche],"adam_updates":updates,
            "heldout_before":initial,"heldout_after":metrics(p,niche,heldout),"mean_training_loss":round(float(np.mean(losses)),6)}


def child(spec_path,out_path):
    spec=json.loads(Path(spec_path).read_text())
    if spec.get("schema")!=SCHEMA or spec.get("mode") not in ("one_resident","two_resident"):
        raise ValueError("Foreign dispatch")
    niches=spec["niches"]
    if not niches or len(set(niches))!=len(niches) or any(x not in NICHES for x in niches):
        raise ValueError("Duplicate or foreign model lease")
    if len(niches) != (1 if spec["mode"]=="one_resident" else 2):
        raise ValueError("Resident count drift")
    updates=spec["updates"]
    if type(updates) is not int or not 1<=updates<=MAX_UPDATES:
        raise ValueError("Invalid bounded updates")
    out=Path(out_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    ts=time.perf_counter()
    results=[train_one(spec["checkpoint_dir"],out.parent,n,updates) for n in niches]
    out.write_text(json.dumps({"schema":SCHEMA,"lease":spec["lease"],"mode":spec["mode"],"niches":niches,
          "pid":os.getpid(),"results":results,"elapsed_s":round(time.perf_counter()-ts,6),
          "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},sort_keys=True,indent=2)+"\n")


def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):return int(line.split()[1])
    except (FileNotFoundError,ValueError,ProcessLookupError):
        pass
    return 0


def run_mode(checkpoint_dir,out,mode,updates):
    groups=([[n] for n in NICHES] if mode=="one_resident" else [list(NICHES[:2]),list(NICHES[2:])])
    d=out/mode
    d.mkdir()
    started=time.perf_counter()
    workers=[]
    failure=[]
    try:
        for i,group in enumerate(groups):
            spec={"schema":SCHEMA,"mode":mode,"niches":group,"updates":updates,
                  "lease":f"{ID}/{updates}/{mode}/worker-{i}","checkpoint_dir":str(checkpoint_dir.resolve())}
            sp=d/(f"worker-{i}.spec.json")
            rp=d/(f"worker-{i}.result.json")
            sp.write_text(json.dumps(spec,sort_keys=True)+"\n")
            stdout=open(d/(f"worker-{i}.stdout.log"),"w")
            stderr=open(d/(f"worker-{i}.stderr.log"),"w")
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--child",str(sp),str(rp)],
                stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            workers.append((p,stdout,stderr,rp))
        peaks={p.pid:0 for p,_,_,_ in workers}
        while any(p.poll() is None for p,_,_,_ in workers):
            if time.perf_counter()-started>MAX_WALL_S:
                raise TimeoutError("FT021 runner deadline exceeded")
            for p,_,_,_ in workers:
                peaks[p.pid]=max(peaks[p.pid],rss(p.pid))
                if peaks[p.pid]>MAX_RSS_KIB:raise MemoryError("Per-process RSS exceeded")
            if sum(peaks.values())>MAX_SUM_RSS_KIB:raise MemoryError("Summed worker RSS exceeded")
            time.sleep(.02)
        if any(p.returncode != 0 for p,_,_,_ in workers):raise RuntimeError("Worker nonzero; original stderr preserved")
        docs=[]
        for p,_,_,rp in workers:
            obj=json.loads(rp.read_text())
            if obj["pid"]!=p.pid or obj["schema"]!=SCHEMA:raise AssertionError("Worker PID/schema mismatch")
            docs.append(obj)
        models=[m for doc in docs for m in doc["results"]]
        if len({m["niche"] for m in models})!=4:raise AssertionError("Duplicate/missing model dispatch")
        return {"mode":mode,"status":"completed","process_count":len(workers),
                "residents_per_process":len(groups[0]),"wall_s":round(time.perf_counter()-started,6),
                "aggregate_sampled_peak_rss_kib":sum(peaks.values()),"models":models}
    except BaseException as exc:
        failure.append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        for p,stdout,stderr,_ in workers:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=2)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            stdout.close();stderr.close()
        if failure:
            (d/"FAILED.json").write_text(json.dumps({"mode":mode,"errors":failure,"worker_returncodes":{str(p.pid):p.returncode for p,_,_,_ in workers}},indent=2)+"\n")


def run(checkpoint_dir,out,updates):
    if type(updates) is not int or not 1<=updates<=MAX_UPDATES:raise ValueError("Unbounded updates")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    report={"schema":SCHEMA,"request_id":ID,"source_sha256":sha(__file__),
            "original_checkpoint_sha256":PARENTS,"updates_per_model_per_mode":updates,
            "status":"starting","failure":None,"modes":[]}
    try:
        for n in NICHES:validate_parent(checkpoint_dir,n)
        # Interleave mode order on alternate workloads to limit warm-cache bias.
        order=("two_resident","one_resident") if updates%2 else ("one_resident","two_resident")
        for mode in order:report["modes"].append(run_mode(Path(checkpoint_dir),out,mode,updates))
        by_mode={m["mode"]:{x["niche"]:x for x in m["models"]} for m in report["modes"]}
        for n in NICHES:
            a=by_mode["one_resident"][n]
            b=by_mode["two_resident"][n]
            if a["model_sha256"]!=b["model_sha256"] or a["heldout_after"]!=b["heldout_after"]:
                raise AssertionError("Residency mode altered checkpoint training or outcomes: "+n)
        report["weight_and_metric_parity"]=True
        report["updates_total"]=8*updates
        report["one_over_two_throughput_ratio"]=round(next(m for m in report["modes"] if m["mode"]=="two_resident")["wall_s"]/
                                                next(m for m in report["modes"] if m["mode"]=="one_resident")["wall_s"],5)
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed";report["failure"]=type(exc).__name__+": "+str(exc)
        raise
    finally:
        (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (out/"SHA256SUMS").write_text("".join(sha(p)+"  "+str(p.relative_to(out))+"\n"
            for p in sorted(out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--updates",type=int,default=600)
    ap.add_argument("--child",type=Path)
    ap.add_argument("child_result",nargs="?",type=Path)
    a=ap.parse_args()
    if a.child:
        if not a.child_result:ap.error("--child requires result path")
        child(a.child,a.child_result)
    else:
        if a.checkpoints is None or a.output is None:ap.error("--checkpoints and --output required")
        r=run(a.checkpoints,a.output,a.updates)
        print(json.dumps({k:r[k] for k in ("status","updates_total","weight_and_metric_parity","one_over_two_throughput_ratio")},indent=2))


if __name__=="__main__":main()
