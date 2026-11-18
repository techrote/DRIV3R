#!/usr/bin/env python3
"""Offline, fail-closed gate for REAL Sprocket-500M LoRA execution.

Does NOT train, generate synthetic model outputs, or download weights.
Requires the pinned upstream ROOT/Instruct files, never chat/ variant.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

MODEL="kandivault/sprocket-500m"
REVISION="eb157d4ddddc4619e661d4a2426bb2dd000fc668"
FILES={
"model.safetensors":"1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864",
"config.json":"557aacbcb7e23a322221ab91b62c4343dd22130388efc3eb4dcdfe1edfd8d1b7",
"generation_config.json":"1000aeb432a4dfa8a5fc161dd05791b8e423216bb82858fc4c8017541c690ac3",
"tokenizer.json":"178a63d94a7de4aa273a6a9c218180c3b7d1912f5a18a91be34bab252f7277eb",
"tokenizer_config.json":"1d1c0733848e0c78a2ec48a7f20cd55ec5ee0b4cbcda9785249af9e489823408",
"special_tokens_map.json":"c575235775a1998faf1256ea0b6f791116ea9d89cc34238ea3bd532b8190773b",
"chat_template.jinja":"995184ace6367465228c6ae16dcc5cfbb36e13147ea6bdba956f6b0a077204fa",
"export_provenance.json":"d21a47177296132942101db6c4455676bf1e0f3d293f805e8827d67f03ab7894",
"README.md":"09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9"
}
PACKAGES=("torch","transformers","peft","accelerate","safetensors","tokenizers","jinja2")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def root_assets(folder):
    if not folder:
        return {"ok":False,"reason":"SPROCKET_MODEL_DIR not set","missing":list(FILES)}
    p=Path(folder).expanduser()
    missing=[n for n in FILES if not (p/n).is_file()]
    mismatch={n:sha(p/n) for n,ref in FILES.items() if (p/n).is_file() and sha(p/n)!=ref}
    return {"ok":not missing and not mismatch,"missing":missing,"mismatched_hashes":mismatch,
            "verified":len(FILES)-len(missing)-len(mismatch)}

def game_corpus(folder):
    if not folder:
        return {"ok":False,"reason":"SPROCKET_CORPUS_DIR not set"}
    p=Path(folder).expanduser()
    try:
        manifest=json.loads((p/"manifest.json").read_text())
        if manifest.get("schema")!="driv3r/corpus-v9":
            raise ValueError("expected corpus-v9")
        groups={}; counts={}
        for split in ("TRAIN","DEV","HELDOUT"):
            spec=manifest["shards"][split]
            if spec["file"]!=split+".jsonl":
                raise ValueError("unexpected shard name")
            data=(p/spec["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest()!=spec["sha256"]:
                raise ValueError("shard SHA-256 mismatch: "+split)
            rows=data.splitlines()
            if len(rows)!=spec["records"]:
                raise ValueError("shard count mismatch: "+split)
            counts[split]=len(rows)
            for line in rows:
                r=json.loads(line)
                if r["split"]!=split or r["target"] not in r["legal_actions"] or not isinstance(r["model_input"],dict):
                    raise ValueError("invalid split/action/model_input")
                topo=r["topology_group"]
                if topo in groups and groups[topo]!=split:
                    raise ValueError("topology-group leakage")
                groups[topo]=split
        return {"ok":True,"counts":counts,"topology_groups":len(groups),
                "manifest_sha256":sha(p/"manifest.json"),
                "scope":"synthetic one-decision supervised pairs, NOT closed-loop or native"}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {"ok":False,"reason":str(exc)[:250]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",default=os.getenv("SPROCKET_MODEL_DIR",""))
    ap.add_argument("--corpus-dir",default=os.getenv("SPROCKET_CORPUS_DIR",""))
    a=ap.parse_args()
    packages={p:importlib.util.find_spec(p) is not None for p in PACKAGES}
    model=root_assets(a.model_dir)
    corpus=game_corpus(a.corpus_dir)
    missing=[]
    if not all(packages.values()): missing.append("missing_real_model_runtime")
    if not model["ok"]: missing.append("missing_or_unverified_root_weights_and_tokenizer")
    if not corpus["ok"]: missing.append("missing_or_invalid_v9_shards")
    try: available=os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    except (OSError,ValueError,AttributeError): available=None
    report={"schema":"driv3r.sprocket.root_lora_preflight/1",
        "model":MODEL,"revision":REVISION,"python":platform.python_version(),
        "cpu_free_memory_bytes_at_check":available,"packages":packages,
        "model_assets":model,"corpus":corpus,"blockers":missing,
        "status":"BLOCKED" if missing else "ASSETS_VERIFIED_NOT_TRAINED",
        "root_forwards":0,"root_optimizer_steps":0,"checkpoint_created":False,
        "native_cybersand":False}
    print(json.dumps(report,indent=2,sort_keys=True))
    return 2 if missing else 0

if __name__=="__main__":
    sys.exit(main())
