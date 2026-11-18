"""FT-020: bounded one/two/four-resident inference comparison.

Real Adam-trained 20x24x1 NumPy policies; independent host-like one-step legal
menus. This is a throughput experiment, NOT the authoritative CyberSand ABI,
full game rollout, FT-019 checkpoint continuation, pretrained SLM, or native W0.
No network, paid compute, or game-side changes. Explicit CLI only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np

ID = "FT-020-20261116-inference-residency"
SCHEMA = "driv3r.fleet_training.ft020/1"
HISTORICAL_POPULATION_BLOB = "8a974a072dd8a5172f9242d83a373f3de345bb3b"
NICHES = ("navigator", "recoverer", "branch_artist", "material_cautious")
FEATURES = ("bias", "paint", "retrace", "stop", "goal_progress", "onward",
            "blocked", "junction", "turn", "frontier", "material_edge",
            "paint_density", "early", "protected_near", "goal_switched",
            "direction_x", "direction_y", "guidance", "motif", "recovery_need")
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
UPDATES = 180
TRAIN_MENUS = 128
TEST_MENUS = 96
MAX_MENUS = 5000
MAX_SECONDS = 180
MAX_PROCESS_RSS_KIB = 450000
MAX_SUM_RSS_KIB = 900000
POLL_S = .02


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % 2**32


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def softmax(v):
    t = np.exp(np.clip(v - max(v), -45, 0))
    return t / t.sum()


def menus(seed, count):
    """Independent synthetic legal menus, not actual sequential world states."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        protected = {(x, y) for x in range(9, 12) for y in range(8, 14)}
        obstacles = {(int(x), int(y)) for x, y in
                     rng.integers(1, 21, size=(35, 2))}
        free = [(x, y) for x in range(1, 21) for y in range(1, 21)
                if (x, y) not in protected and (x, y) not in obstacles]
        x, y = free[int(rng.integers(len(free)))]
        goal = free[int(rng.integers(len(free)))]
        heading = tuple(DIRECTIONS)[int(rng.integers(4))]
        occupied = protected | obstacles
        options = []
        for name, (dx, dy) in DIRECTIONS.items():
            after = (x + dx, y + dy)
            if not (1 <= after[0] <= 20 and 1 <= after[1] <= 20) or after in occupied:
                continue
            options.append((name, after, (dx, dy)))
        if rng.random() < .45:
            neighbours = [(a, (x + dx, y + dy)) for a, (dx, dy) in DIRECTIONS.items()
                          if 1 <= x + dx <= 20 and 1 <= y + dy <= 20]
            if neighbours:
                _, parent = neighbours[int(rng.integers(len(neighbours)))]
                options.append(("RETRACE", parent, (0, 0)))
        options.append(("STOP", (x, y), (0, 0)))
        features = []
        for name, after, vec in options:
            paint = name in DIRECTIONS
            retrace = name == "RETRACE"
            stop = name == "STOP"
            dist0 = abs(x - goal[0]) + abs(y - goal[1])
            dist1 = abs(after[0] - goal[0]) + abs(after[1] - goal[1])
            adj = [(after[0] + dx, after[1] + dy) for dx, dy in DIRECTIONS.values()]
            onward = sum(1 <= nx <= 20 and 1 <= ny <= 20 and (nx,ny) not in occupied for nx,ny in adj)
            near_protected = sum(p in protected for p in adj)
            near_blocked = sum(p in obstacles for p in adj)
            f = [1.,float(paint),float(retrace),float(stop),
                 (dist0-dist1)/2.,onward/4.,near_blocked/4.,
                 float(onward >= 3),float(paint and heading != name),
                 float(onward >= 2),float(near_protected > 0),
                 float(rng.random()),float(rng.random()),near_protected/4.,
                 float(rng.random() < .5),float(vec[0]),float(vec[1]),
                 float((after[0] - x)*np.sign(goal[0] - x) > 0),
                 float(rng.random()),float(retrace and onward > 1)]
            features.append(f)
        X = np.asarray(features, dtype=np.float64)
        assert X.shape[1] == 20 and np.isfinite(X).all()
        yield [n for n, _, _ in options], X


def score(niche, X):
    cols = {name: X[:,i] for i,name in enumerate(FEATURES)}
    base = -.35*cols["stop"] + .12*cols["paint"]
    if niche == "navigator":
        return base + 1.7*cols["goal_progress"] + .4*cols["onward"]
    if niche == "recoverer":
        return base + .6*cols["retrace"]*cols["recovery_need"] + .65*cols["onward"] - .6*cols["blocked"]
    if niche == "branch_artist":
        return base + .85*cols["junction"] + .6*cols["frontier"] + .4*cols["turn"]
    return base + 1.2*cols["goal_progress"] - 1.5*cols["protected_near"] - .7*cols["blocked"]


class Policy:
    def __init__(self, seed, params=None):
        rng = np.random.default_rng(seed)
        self.p = (dict(W1=rng.normal(0,.15,(20,24)),b1=np.zeros(24),
                       W2=rng.normal(0,.15,24),b2=np.zeros(1)) if params is None
                  else {k:np.asarray(v,dtype=np.float64) for k,v in params.items()})
        if {k:v.shape for k,v in self.p.items()} != {"W1":(20,24),"b1":(24,),"W2":(24,),"b2":(1,)}:
            raise ValueError("Foreign/incompatible 20x24x1 policy")
        if not all(np.isfinite(v).all() for v in self.p.values()):
            raise ValueError("Nonfinite model weights")
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.updates=0

    def logits(self, X):
        return np.tanh(X@self.p["W1"]+self.p["b1"])@self.p["W2"]+self.p["b2"][0]

    def teach(self, X, target):
        H=np.tanh(X@self.p["W1"]+self.p["b1"])
        predicted=softmax(H@self.p["W2"]+self.p["b2"][0])
        d=predicted-target
        dH=d[:,None]*self.p["W2"][None,:]*(1-H*H)
        g=dict(W1=X.T@dH,b1=dH.sum(0),W2=H.T@d,b2=np.array([d.sum()]))
        self.updates+=1
        for k,grad in g.items():
            self.m[k]=.9*self.m[k]+.1*grad
            self.v[k]=.999*self.v[k]+.001*grad*grad
            m=self.m[k]/(1-.9**self.updates)
            v=self.v[k]/(1-.999**self.updates)
            self.p[k]-=.007*m/(np.sqrt(v)+1e-8)
        return -float(np.sum(target*np.log(np.maximum(predicted,1e-12))))

    def record(self, niche):
        return dict(schema="population-mlp-v1",niche=niche,features=list(FEATURES),
                    hidden=24,adam_updates=self.updates,
                    parameters={k:v.tolist() for k,v in self.p.items()})


def evaluate(p,niche,data):
    return dict(accuracy=round(sum(int(np.argmax(p.logits(X))==np.argmax(score(niche,X)))
                                   for _,X in data)/len(data),6),
                cross_entropy=round(sum(-math.log(max(float(softmax(p.logits(X))[np.argmax(score(niche,X))]),1e-12))
                                        for _,X in data)/len(data),6))


def train_models(root):
    folder=root/"checkpoints"
    folder.mkdir()
    summary={}
    for niche in NICHES:
        p=Policy(stable_seed(ID,"MODEL",niche))
        train=list(menus(stable_seed(ID,"TRAIN",niche),TRAIN_MENUS))
        test=list(menus(stable_seed(ID,"HELDOUT",niche),TEST_MENUS))
        initial=evaluate(p,niche,test)
        rng=np.random.default_rng(stable_seed(ID,"OPT",niche))
        losses=[]
        for _ in range(UPDATES):
            _,X=train[int(rng.integers(len(train)))]
            target=softmax(score(niche,X)/.42)
            losses.append(p.teach(X,target))
        after=evaluate(p,niche,test)
        dest=folder/(niche+".json")
        dest.write_text(json.dumps(p.record(niche),sort_keys=True,separators=(",",":"))+"\n")
        summary[niche]=dict(lease=ID+"/trained/"+niche,sha256=digest(dest),
                            adam_updates=p.updates,initial=initial,heldout=after,
                            mean_training_loss=round(float(np.mean(losses)),6))
    return summary


def live_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):return int(line.split()[1])
    except (FileNotFoundError,ProcessLookupError,ValueError):
        pass
    return 0


def child(spec_path,result_path):
    spec=json.loads(spec_path.read_text())
    niches=spec["niches"]
    if not niches or len(set(niches))!=len(niches) or any(n not in NICHES for n in niches):
        raise ValueError("Duplicate/foreign child model lease")
    count=spec["menus_per_model"]
    if type(count)!=int or not 1<=count<=MAX_MENUS:raise ValueError("Unbounded work")
    models={}
    for n in niches:
        p=spec_path.parent.parent/"checkpoints"/(n+".json")
        if digest(p)!=spec["model_sha256"][n]:raise ValueError("Checkpoint identity mismatch")
        obj=json.loads(p.read_text())
        if obj["schema"]!="population-mlp-v1" or obj["niche"]!=n or obj["features"]!=list(FEATURES) or obj["adam_updates"]!=UPDATES:
            raise ValueError("Invalid checkpoint provenance")
        models[n]=Policy(0,obj["parameters"])
    start=time.perf_counter()
    outputs={}
    for n in niches:
        h=hashlib.sha256()
        n_stop=0
        menu_total=0
        for legal,X in menus(stable_seed(ID,"INFERENCE",n),count):
            ix=int(np.argmax(models[n].logits(X)))
            selected=legal[ix]
            menu_total+=len(legal)
            n_stop+=int(selected=="STOP")
            h.update(selected.encode()+b"\0")
        outputs[n]=dict(actions=count,legal_options_seen=menu_total,stop_choices=n_stop,
                        choices_sha256=h.hexdigest())
    elapsed=time.perf_counter()-start
    result_path.write_text(json.dumps(dict(schema=SCHEMA,lease=spec["lease"],
        pid=os.getpid(),residents=niches,model_sha256=spec["model_sha256"],
        results=outputs,elapsed_s=round(elapsed,6),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sort_keys=True,indent=2)+"\n")


def cpu_ticks(pid):
    try:
        return sum(map(int,Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[11:13]))
    except (FileNotFoundError,ValueError,IndexError):
        return None


def launch_mode(out,name,groups,checkpoint_sha,count,deadline):
    directory=out/name
    directory.mkdir()
    workers=[]
    started=time.perf_counter()  # includes process launch, import and first inference
    first_sample=None
    simultaneous_progress=False
    try:
        for i,group in enumerate(groups):
            spec=dict(schema=SCHEMA,lease=f"{ID}/{name}/worker-{i}",niches=group,
                      menus_per_model=count,model_sha256={n:checkpoint_sha[n]["sha256"] for n in group})
            spec_path=directory/(f"worker-{i}.spec.json")
            result_path=directory/(f"worker-{i}.result.json")
            spec_path.write_text(json.dumps(spec,sort_keys=True)+"\n")
            stdout=open(directory/(f"worker-{i}.stdout.log"),"w")
            stderr=open(directory/(f"worker-{i}.stderr.log"),"w")
            env=dict(os.environ)
            env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--child",str(spec_path),str(result_path)],
                               stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            workers.append(dict(pid=p.pid,process=p,files=(stdout,stderr),spec=spec_path,
                                result=result_path,peak_rss_kib=0,rc=None))
        while any(w["rc"] is None for w in workers):
            if time.perf_counter()>deadline:raise TimeoutError("FT020 entire run exceeded deadline")
            alive=[]
            for w in workers:
                if w["rc"] is not None:continue
                rc=w["process"].poll()
                if rc is not None:
                    w["rc"]=rc
                else:
                    w["peak_rss_kib"]=max(w["peak_rss_kib"],live_rss(w["pid"]))
                    if w["peak_rss_kib"]>MAX_PROCESS_RSS_KIB:raise MemoryError("Individual inference worker RSS exceeded cap")
                    alive.append(w)
            if sum(w["peak_rss_kib"] for w in workers)>MAX_SUM_RSS_KIB:
                raise MemoryError("Inference group RSS exceeded cap")
            if len(alive)==len(workers) and len(workers)>1:
                ticks={w["pid"]:cpu_ticks(w["pid"]) for w in alive}
                if first_sample and all(ticks[k] is not None and ticks[k]>v for k,v in first_sample.items() if v is not None):
                    simultaneous_progress=True
                first_sample=ticks
            time.sleep(POLL_S)
        elapsed=time.perf_counter()-started
        if any(w["rc"]!=0 for w in workers):
            raise RuntimeError("One or more inference workers exited nonzero; stderr preserved")
        evidence=[]
        for w in workers:
            doc=json.loads(w["result"].read_text())
            if doc["pid"]!=w["pid"] or doc["schema"]!=SCHEMA:
                raise AssertionError("Inference worker identity mismatch")
            evidence.append(doc)
        return dict(mode=name,wall_s=round(elapsed,6),process_count=len(workers),
                    resident_models_per_process=len(groups[0]),
                    sampled_sum_peak_rss_kib=sum(w["peak_rss_kib"] for w in workers),
                    observed_multiple_process_cpu_progress=simultaneous_progress,
                    outputs=evidence)
    finally:
        for w in workers:
            if w["process"].poll() is None:
                os.killpg(w["pid"],signal.SIGKILL)
            w["process"].wait()
            for f in w["files"]:f.close()


def run(output,count):
    if not 1<=count<=MAX_MENUS:raise ValueError("Inference menu budget out of bounds")
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.perf_counter()+MAX_SECONDS
    report=dict(schema=SCHEMA,request_id=ID,status="starting",failures=[],
                executor_sha256=digest(Path(__file__)),
                historical_population_git_blob=HISTORICAL_POPULATION_BLOB,
                historical_fixture_executed=False,
                machine=dict(cpu_count=os.cpu_count(),affinity_cores=(len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None),
                             numpy=np.__version__,system=sys.platform),
                game="independent synthetic one-step menus, not a replayed host; not native W0 or canonical ABI",
                claimed_pretrained_slms=0,trained_checkpoint_source="fresh local synthetic Adam",
                model_count=4,menus_per_model=count,optimizer_updates_per_model=UPDATES,
                modes=[])
    try:
        report["models"]=train_models(output)
        groups={"one_resident":[[n] for n in NICHES],
                "two_resident":[list(NICHES[:2]),list(NICHES[2:])],
                "four_resident":[list(NICHES)]}
        for name,cohorts in groups.items():
            report["status"]="running_"+name
            report["modes"].append(launch_mode(output,name,cohorts,report["models"],count,deadline))
        expected=None
        for mode in report["modes"]:
            results={n:v for w in mode["outputs"] for n,v in w["results"].items()}
            if len(results)!=4:raise AssertionError("Missing/duplicate model dispatch")
            if expected is not None and expected!=results:
                raise AssertionError("Residency mode changed model decisions")
            expected=results
            mode["total_decisions"]=sum(v["actions"] for v in results.values())
            mode["decisions_per_second"]=round(mode["total_decisions"]/mode["wall_s"],3)
        report["status"]="completed"
        report["parity_verified"]=True
        rate1=report["modes"][0]["decisions_per_second"]
        report["two_vs_one_rate_ratio"]=round(report["modes"][1]["decisions_per_second"]/rate1,4)
        report["four_vs_one_rate_ratio"]=round(report["modes"][2]["decisions_per_second"]/rate1,4)
    except BaseException as e:
        report["status"]="failed"
        report["failures"].append(f"{type(e).__name__}: {e}")
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            f"{digest(f)}  {f.relative_to(output)}\n"
            for f in sorted(output.rglob("*")) if f.is_file() and f.name!="SHA256SUMS"))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path)
    ap.add_argument("--menus-per-model",type=int,default=1800)
    ap.add_argument("--child",type=Path)
    ap.add_argument("child_result",nargs="?",type=Path)
    a=ap.parse_args()
    if a.child:
        if not a.child_result:ap.error("--child needs child_result path")
        child(a.child,a.child_result)
    else:
        if not a.output:ap.error("--output required")
        r=run(a.output,a.menus_per_model)
        print(json.dumps({k:r[k] for k in ("status","two_vs_one_rate_ratio","four_vs_one_rate_ratio","parity_verified")},indent=2))


if __name__=="__main__":
    main()
