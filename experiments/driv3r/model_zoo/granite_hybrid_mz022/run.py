"""MZ-022: source-pinned IBM Granite 4.0 H 350M hybrid pretrained policy probe.

Experimental host from DRIV3R model_zoo/mamba130 only, NOT native CyberSand.
CPU-only, opt-in; no fake pretrained substitute if checkpoint/dependency missing.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import statistics
import time

MODEL = "ibm-granite/granite-4.0-h-350m"
PIN = "3b17b717b8f2f5d305b0a92c1491e239aeda19c8"
SCHEMA = "DRIV3R-MZ022-GRANITE-HYBRID-v1"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "PENUP", "STOP")


def budget(a):
    if not (4 <= a.train_seeds <= 24 and 2 <= a.eval_seeds <= 8 and
            3 <= a.steps <= 12 and 8 <= a.updates <= 128 and
            2 <= a.rollout_seeds <= 8 and 8 <= a.horizon <= 24 and
            2 <= a.batch <= 32 and 1 <= a.threads <= 8 and
            64 <= a.max_context <= 256):
        raise ValueError("MZ-022 hard CPU experiment budget exceeded")


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile(a, q):
    return round(sorted(a)[int((len(a) - 1) * q)], 5) if a else None


def host_choice(logits, legal):
    offered = [a for a in ACTIONS if a in legal]
    if not offered:
        raise ValueError("empty host legal action menu")
    return max(offered, key=lambda a: float(logits[ACTIONS.index(a)]))


def validate_splits(train, dev, test):
    seeds = [set(int(row[2]) for row in rows) for rows in (train, dev, test)]
    if any(seeds[i] & seeds[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("cross-split world seed leakage")
    prompts = [{row[0] for row in rows} for rows in (train, dev, test)]
    if any(prompts[i] & prompts[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("cross-split exact prompt leakage")


def preflight():
    from huggingface_hub.constants import HF_HUB_CACHE
    snapshot = Path(HF_HUB_CACHE) / ("models--" + MODEL.replace("/", "--")) / "snapshots" / PIN
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "transformers", "huggingface_hub", "safetensors")}
    weights = list(snapshot.glob("*.safetensors")) if snapshot.is_dir() else []
    missing = (["Transformers Python runtime missing"] if not dependencies["transformers"] else [])
    if not weights:
        missing.append("source-pinned pretrained weights absent from local HF cache")
    return {"schema": SCHEMA, "status": "BLOCKED" if missing else "CACHE_PRESENT_NOT_LOADED",
            "model": MODEL, "revision": PIN, "license_from_upstream_card": "apache-2.0",
            "architecture_from_upstream_config": "GraniteMoeHybridForCausalLM, 28 Mamba2 + 4 attention layers",
            "dependencies_present": dependencies, "cached_snapshot_path": str(snapshot),
            "cached_safetensors_files": [x.name for x in weights], "missing": missing,
            "notes": "Cache presence does not demonstrate loadability; no pretrained model execution or training here."}


def execute(a):
    budget(a)
    if a.preflight:
        result = preflight()
        print(json.dumps(result, indent=2))
        return result
    import torch
    from huggingface_hub import model_info, snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from experiments.driv3r.model_zoo.mamba130.counterfactual import (
        samples, raw_features, paired_switch_metrics)
    from experiments.driv3r.model_zoo.mamba130.world import World
    from experiments.driv3r.model_zoo.mamba130.run_counterfactual import measure

    torch.manual_seed(22022)
    random.seed(22022)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # One complete immutable upstream revision; network opt-in, no implicit main.
    snapshot = Path(snapshot_download(MODEL, revision=PIN, local_files_only=not a.allow_download))
    safetensors = sorted(snapshot.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError("no pretrained safetensors in pinned snapshot")
    if a.allow_download and model_info(MODEL, revision=PIN).sha != PIN:
        raise RuntimeError("upstream revision mismatch")
    hashes = {p.name: hash_file(p) for p in safetensors}
    tok = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.float32,
                        trust_remote_code=False, local_files_only=True, use_safetensors=True)
    config = model.config
    kinds = list(getattr(config, "layer_types", []))
    if (config.model_type != "granitemoehybrid" or kinds.count("mamba") != 28 or
            kinds.count("attention") != 4 or not hasattr(model, "model")):
        raise RuntimeError("unexpected model architecture; refuse surrogate/changed variant")
    parameters = sum(p.numel() for p in model.parameters())
    if not 300_000_000 <= parameters <= 400_000_000:
        raise RuntimeError("unexpected parameter count")
    model.eval().requires_grad_(False)
    load_seconds = round(time.monotonic() - started, 3)

    train = samples(range(400, 400+a.train_seeds), a.steps)
    dev = samples(range(900, 900+a.eval_seeds), a.steps)
    test = samples(range(1200, 1200+a.eval_seeds), a.steps)
    validate_splits(train, dev, test)
    if paired_switch_metrics(test, [row[1] for row in test])["changed_target_pairs"] < 4:
        raise RuntimeError("held-out changed-goal control inadequate")
    prompts = sorted({row[0] for rows in (train, dev, test) for row in rows})
    feature_cache = {}
    timings_ms, lengths = [], []

    def vector(prompt):
        if prompt not in feature_cache:
            text = "You are a worm painting a protected map. Use only host legal actions. " + prompt
            ids = tok(text, return_tensors="pt", truncation=False)
            length = int(ids["input_ids"].shape[1])
            if not 8 <= length <= a.max_context:
                raise RuntimeError(f"input has {length} tokens; no silent truncation allowed")
            lengths.append(length)
            t = time.perf_counter()
            with torch.inference_mode():
                states = model.model(**ids, use_cache=False, return_dict=True).last_hidden_state
                embedding = states[0, -1].to(torch.float32).detach().cpu().clone()
            timings_ms.append(1000 * (time.perf_counter()-t))
            if not torch.isfinite(embedding).all():
                raise RuntimeError("nonfinite pretrained hidden state")
            feature_cache[prompt] = embedding
        return feature_cache[prompt]

    # These are real pretrained features, not random or structural surrogates.
    for prompt in prompts:
        vector(prompt)

    def arrays(rows):
        X = torch.stack([vector(r[0]) for r in rows])
        S = torch.tensor([raw_features(r[0]) for r in rows], dtype=torch.float32)
        Y = torch.tensor([int(r[1]) for r in rows], dtype=torch.long)
        return X, S, Y
    X, SX, Y = arrays(train)
    D, SD, DY = arrays(dev)
    T, ST, TY = arrays(test)
    dim = X.shape[-1]

    # Train a bottleneck residual adapter over genuine frozen pretrained states.
    # Baselines: pretrained linear probe (no bottleneck), cheap structured-control.
    def adapter():
        return torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 32),
            torch.nn.Tanh(), torch.nn.Linear(32, dim), torch.nn.GELU(),
            torch.nn.Linear(dim, len(ACTIONS)))
    torch.manual_seed(22022)
    models = {"granite_adapter": adapter(), "pretrained_linear": torch.nn.Linear(dim, len(ACTIONS)),
              "structured": torch.nn.Linear(SX.shape[-1], len(ACTIONS))}
    opt = {k: torch.optim.AdamW(v.parameters(), lr=0.002) for k, v in models.items()}
    rng = torch.Generator().manual_seed(22022)
    hist = torch.bincount(Y, minlength=len(ACTIONS)).float()
    weight = (hist.sum()/hist.clamp(min=1)).sqrt()
    weight = torch.where(hist > 0, weight, 0.)
    weight /= weight[weight > 0].mean()
    losses = {k: [] for k in models}
    for _ in range(a.updates):
        ix = torch.randint(len(Y), (min(a.batch, len(Y)),), generator=rng)
        for key, net in models.items():
            opt[key].zero_grad(set_to_none=True)
            z = SX if key == "structured" else X
            loss = torch.nn.functional.cross_entropy(net(z[ix]), Y[ix], weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt[key].step()
            losses[key].append(float(loss.detach()))
    for net in models.values():
        net.eval()

    scores = {}
    with torch.inference_mode():
        for key, net in models.items():
            source = (SD, ST) if key == "structured" else (D, T)
            dv, tv = (net(z).argmax(-1).tolist() for z in source)
            scores[key] = {"dev": measure(dev, dv), "test": measure(test, tv),
                           "goal_pairs": paired_switch_metrics(test, tv),
                           "trained_parameters": sum(p.numel() for p in net.parameters()),
                           "loss_first_last": [losses[key][0], losses[key][-1]]}

    traces = []
    roll_seeds = list(range(1807, 1807+a.rollout_seeds))
    for seed in roll_seeds:
        for key in ("granite_adapter", "pretrained_linear", "structured", "teacher", "random_nonstop"):
            world = World.make(seed)
            r = random.Random(seed*101+17)
            events, illegal = [], 0
            for turn in range(a.horizon):
                offered = world.legal()
                if key == "teacher":
                    raw = selected = world.teacher(r)
                elif key == "random_nonstop":
                    raw = selected = r.choice([x for x in offered if x != "STOP"] or ["STOP"])
                else:
                    with torch.inference_mode():
                        v = (torch.tensor(raw_features(world.prompt()),dtype=torch.float32)
                             if key == "structured" else vector(world.prompt()))
                        logits = models[key](v[None])[0]
                    raw = ACTIONS[int(logits.argmax())]
                    illegal += int(raw not in offered)
                    selected = host_choice(logits, offered)
                events.append({"t": turn, "pos": list(world.pos), "raw": raw,
                               "selected": selected, "host_legal": list(offered)})
                world.apply(selected)
                if world.stopped:
                    break
            traces.append({"seed": seed, "policy": key, "steps": len(events),
                           "raw_illegal": illegal, "goal_reached": world.pos == world.goal,
                           "painted": len(world.painted), "events": events})

    # The large model is NOT serialized into Git; only the compact learned adapter.
    checkpoint = out/"granite_mz022_adapter.pt"
    torch.save({"schema": SCHEMA, "pretrained_model": MODEL, "revision": PIN,
                "upstream_weight_sha256": hashes, "train_updates": a.updates,
                "trained_models": {k: net.state_dict() for k, net in models.items()}}, checkpoint)
    manifest = {"schema": SCHEMA, "status": "PRETRAINED_BACKBONE_LOADED_ADAPTER_TRAINED",
                "source_commit": os.getenv("GITHUB_SHA", "LOCAL-UNVERIFIED"),
                "pretrained_model": MODEL, "revision": PIN, "license": "apache-2.0",
                "weight_sha256": hashes, "model_parameters": parameters,
                "load_seconds": load_seconds, "train_seeds": [400,400+a.train_seeds-1],
                "dev_seeds": [900,900+a.eval_seeds-1],
                "test_seeds": [1200,1200+a.eval_seeds-1], "rollout_seeds": roll_seeds,
                "updates": a.updates, "scores": scores, "rollouts": {
                    k: {"reached": sum(t["goal_reached"] for t in traces if t["policy"]==k),
                        "raw_illegal": sum(t["raw_illegal"] for t in traces if t["policy"]==k),
                        "painted": sum(t["painted"] for t in traces if t["policy"]==k)}
                    for k in ("granite_adapter", "pretrained_linear", "structured", "teacher", "random_nonstop")},
                "model_forward_p50_ms": quantile(timings_ms, 0.5),
                "model_forward_p95_ms": quantile(timings_ms, 0.95),
                "context_p50_tokens": quantile(lengths,0.5),
                "context_max_tokens": max(lengths),
                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "adapter_sha256": hash_file(checkpoint), "synthetic_host_not_native_cybersand": True,
                "limitations": ["Frozen pretrained backbone with trained bottleneck adapter/head, not backbone LoRA",
                   "Full-prefix uncached CPU feature inference; not recurrent cached decision latency",
                   "Teacher not oracle: goal-reached may be zero in all arms",
                   "Host mask applied only to executed actions; raw illegal measured before mask",
                   "Trained policy is synthetic, not real CyberSand PAINT/Workbench"]}
    (out/"metrics.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"trajectories.json").write_text(json.dumps(traces, indent=2)+"\n")
    print(json.dumps({"status": "trained", "metrics_path": str(out/"metrics.json")}))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/driv3r_mz022")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit pinned public HF checkpoint transfer")
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--rollout-seeds", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=256)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
