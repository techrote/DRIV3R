#!/usr/bin/env python3
"""QL-023: executable *full* pinned LFM350 quantized-LoRA/W0 bridge.

Requires real root weights and tokenizer on an opt-in CPU host. --preflight only
checks genuine archived QL022 packed trained tensors; it is NOT model inference.
No synthetic score emulator, no native CyberSand writes, no external repo writes.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import struct
import sys
import tempfile
import time
import zipfile
import numpy as np

REVISION = '9921cd4d82c0e3986887aa4b376f9446729ef8a3'
MODEL = 'LiquidAI/LFM2.5-350M'
SOURCE_SHA = '1db95314355fb4ea0439b63825a0f829b14bb0a0f8b01284fd614446aca85a47'
ADAPTER_SHA = '436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c'
PACKED_SHA = '12d9aee60e09be9d01fe544cfbccb5e1e4b809d64b8d4614140f8bc4fa381644'
PER_MODULE_SHA = 'a30605ab04b2234425ff3fdccead82dba0650951c749097ea943a14fb65db6e4'
ACTIONS = ('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP')
MODES = ('fp32', 'q4', 'adaptive', 'q2', 'q1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def insist(ok, why):
    if not ok:
        raise ValueError(why)


def read_bundle(path):
    with zipfile.ZipFile(path) as outer:
        insist(outer.testzip() is None, 'QL022 outer ZIP CRC failure')
        raw = outer.read('evidence/actual-weight-variants.zip')
        insist(digest(raw) == PACKED_SHA, 'QL022 trained packed weights SHA drift')
        tab = outer.read('evidence/per-module.json')
        insist(digest(tab) == PER_MODULE_SHA, 'QL022 module catalogue SHA drift')
        rows = json.loads(tab)
        insist(len(rows) == 12 and [x['name'] for x in rows] ==
               sorted(x['name'] for x in rows), '12 source-ordered LoRA projections required')
        meta = json.loads(outer.read('evidence/results.json'))
        insist(meta['model_revision'] == REVISION and meta['step96_adapter_sha256'] == ADAPTER_SHA,
               'incorrect model/checkpoint provenance')
        archive = zipfile.ZipFile(io.BytesIO(raw))
        insist(archive.testzip() is None, 'QL022 inner ZIP CRC failure')
        insist(len(archive.namelist()) == 312, 'truncated packed variant set')
        return archive, rows, meta


def decode(blob, offset=0):
    """QL022 compact signed quant / FP16, row-major unpadded bitstream.

    Each row carries one FP16 group scale for every 64 input weights; packed
    codes carry only real cells, not group padding.  Returns tensor and cursor.
    """
    if blob[offset:offset+2] == b'QH':
        insist(len(blob) >= offset+10, 'short FP16 tensor')
        m,n = struct.unpack_from('<II',blob,offset+2)
        end=offset+10+2*m*n
        insist(0<m<=16384 and 0<n<=16384 and end<=len(blob),'bad FP16 bounds')
        return np.frombuffer(blob[offset+10:end],dtype='<f2').astype('f4').reshape(m,n).copy(),end
    insist(len(blob) >= offset+12,'short quant header')
    a,b,bits,group,m,n=struct.unpack_from('<BBBBII',blob,offset)
    insist((a,b)==(81,76) and bits in (1,2,3,4) and group==64
           and 0<m<=16384 and 0<n<=16384,'bad quant type/dimensions')
    groups=(n+63)//64
    scale_start=offset+12
    codes_start=scale_start+2*m*groups
    end=codes_start+(m*n*bits+7)//8
    insist(end<=len(blob),'short packed tensor')
    scales=np.frombuffer(blob[scale_start:codes_start],dtype='<f2').astype('f4').reshape(m,groups)
    data=blob[codes_start:end]
    codes=np.zeros(m*n,dtype='u1')
    mask=(1<<bits)-1
    for i in range(m*n):
        pos=i*bits; v=data[pos>>3]>>(pos&7)
        if (pos&7)+bits>8:v|=data[(pos>>3)+1]<<(8-(pos&7))
        codes[i]=v&mask
    c=codes.reshape(m,n).astype('f4')
    if bits==1:
        signed=2*c-1
    else:
        signed=2*c/((1<<bits)-1)-1
    out=signed*scales[:,np.arange(n)//64]
    out[scales[:,np.arange(n)//64]==0]=0
    insist(np.isfinite(out).all(),'nonfinite decoded real weight')
    return out,end


def choices(mode,rows,meta):
    if mode=='adaptive':
        return [tuple(p) for p in meta['adaptive']['Q2Q4v_budget']['A_B_bits']]
    if mode=='fp32':return []
    bits=int(mode[1:])
    return [(bits,bits)]*len(rows)


def unpack_model(mode,packed,rows,meta):
    if mode=='fp32':return {},688128
    selections=choices(mode,rows,meta)
    out={};total=0
    for j,(entry,(ab,bb)) in enumerate(zip(rows,selections)):
        payload=packed.read(f'{j:02d}/A{ab}_B{bb}.bin')
        a,k=decode(payload)
        b,end=decode(payload,k)
        insist(end==len(payload),'trailing packed tensor bytes')
        insist(a.shape[0]==b.shape[1]==8 and a.shape[1]>=8 and b.shape[0]>=8,
               'not a rank-8 real LoRA pair')
        key=entry['name'];out[key]=(a,b)
        total+=len(payload)
        evidence=next(c for c in entry['candidates'] if (c['A'],c['B'])==(ab,bb))
        insist(len(payload)==evidence['bytes'],'packed size not per-module source')
    insist(total==sum(next(c['bytes'] for c in r['candidates'] if
                      (c['A'],c['B'])==(ab,bb)) for r,(ab,bb) in zip(rows,selections)),
           'aggregated compressed size mismatch')
    return out,total


def source_checkpoint(source):
    insist(digest(Path(source).read_bytes())==SOURCE_SHA,'historical donor ZIP SHA drift')
    with zipfile.ZipFile(source) as z:
        insist(z.testzip() is None,'historical donor CRC failure')
        prefix='original-experiment/adapter-step-96/'
        weights=z.read(prefix+'adapter_model.safetensors')
        insist(digest(weights)==ADAPTER_SHA,'actual step96 checkpoint SHA drift')
        config=z.read(prefix+'adapter_config.json')
    return weights,config


def pin_root(cache,allow_download):
    if allow_download:
        from huggingface_hub import HfApi,snapshot_download
        insist(HfApi().model_info(MODEL,revision=REVISION).sha==REVISION,'upstream revision drift')
        snapshot_download(MODEL,revision=REVISION,local_dir=cache,max_workers=2,
            allow_patterns=['*.safetensors','*.safetensors.index.json','config.json',
            'generation_config.json','tokenizer*','special_tokens_map.json',
            'added_tokens.json','vocab.*','merges.txt','README.md','LICENSE*'])
    manifest={str(x.relative_to(cache)):digest(x.read_bytes()) for x in sorted(cache.rglob('*'))
              if x.is_file() and '.cache' not in x.parts}
    insist(any(x.endswith('.safetensors') for x in manifest),'full LFM root weights unavailable')
    insist(any(x.startswith('tokenizer') for x in manifest),'pinned tokenizer unavailable')
    return manifest


def run(args,packed,rows,meta):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load as load_safetensors
    # Reuse original full-LFM scorer and source-faithful synthetic W0 evaluator.
    import lfm350 as c
    import gp350018_matched_decoder as g
    insist(args.out and not args.out.exists(),'immutable output directory must not exist')
    insist(args.w0 and args.cache and args.source and args.golden,
           'missing full-model/GP350-012 golden runtime inputs')
    insist(c.REVISION==REVISION and tuple(c.ACTIONS)==ACTIONS,'LFM scorer ABI drift')
    insist(args.max_turns in (8,16,24,32) and 600<=args.deadline<=7200,
           'unsafe full-model decision or deadline budget')
    torch.set_num_threads(2)
    started=time.monotonic()
    w0=c.load_w0(args.w0)
    weights,config=source_checkpoint(args.source)
    manifest=pin_root(args.cache,args.allow_download)
    historical='88dfeee5360ae16d03de99d6bdfb2dbe79a819613af215373f6dcfb9fdcf0ae6'
    insist(digest(args.golden.read_bytes())==historical,'GP350-012 golden SHA mismatch')
    with zipfile.ZipFile(args.golden) as z:
        insist(z.testzip() is None,'golden ZIP CRC failure')
        expected=json.loads(z.read('original-experiment/full-model-manifest.json'))
        insist(expected['revision']==REVISION and
               all(manifest.get(n)==v for n,v in expected['sha256'].items()),
               'full model root differs from original evaluated frozen checkpoint')
    args.out.mkdir(parents=True)
    (args.out/'root-sha256.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp)
        (temp/'adapter_model.safetensors').write_bytes(weights)
        (temp/'adapter_config.json').write_bytes(config)
        tok=AutoTokenizer.from_pretrained(args.cache,local_files_only=True,trust_remote_code=False)
        ids=c.id_map(tok)
        base=AutoModelForCausalLM.from_pretrained(args.cache,local_files_only=True,
                                                  trust_remote_code=False,dtype=torch.float32)
        insist(base.config.model_type=='lfm2' and base.config.num_hidden_layers==16,
               'LFM architecture mismatch')
        model=PeftModel.from_pretrained(base,temp,is_trainable=False)
        model.requires_grad_(False);model.eval()
        before=c.base_fingerprint(model)
        params=dict(model.named_parameters())
        names={}
        for entry in rows:
            stem=entry['name']
            keys=(stem+'.lora_A.default.weight',stem+'.lora_B.default.weight')
            insist(all(k in params for k in keys),'PEFT live projection mapping mismatch: '+stem)
            names[stem]=keys
        historical_state=load_safetensors(weights)
        insist(len(historical_state)==24,'expected 24 original trained matrices')
        for raw_name,expected_weight in historical_state.items():
            adapted=raw_name.replace('.lora_A.weight','.lora_A.default.weight').replace(
                '.lora_B.weight','.lora_B.default.weight')
            insist(adapted in params and torch.equal(
                params[adapted].detach().cpu().float(),expected_weight.float()),
                'live PEFT tensor not identical to source-pinned checkpoint: '+raw_name)
        originals={k:params[k].detach().clone() for keys in names.values() for k in keys}
        summary=[]
        for mode in MODES:
            # Restore all 24 exact original PEFT tensor values before each arm.
            with torch.no_grad():
                for k,v in originals.items():params[k].copy_(v)
            replacements,payload_bytes=unpack_model(mode,packed,rows,meta)
            with torch.no_grad():
                for stem,(a,b) in replacements.items():
                    keys=names[stem]
                    insist(tuple(params[keys[0]].shape)==a.shape and
                           tuple(params[keys[1]].shape)==b.shape,'projection shape mismatch')
                    params[keys[0]].copy_(torch.from_numpy(a))
                    params[keys[1]].copy_(torch.from_numpy(b))
            insist(before==c.base_fingerprint(model),'frozen base mutated by quantized adapter')
            for family in c.FAMILIES:
                if time.monotonic()-started > args.deadline-120:
                    raise TimeoutError('bounded model CPU deadline hit, partial evidence retained')
                item=g.episode(w0,model,tok,ids,args.out,'ql023-'+mode,
                               ('A-greedy','greedy',None,0.0),24,9803,family,
                               args.max_turns,started,args.deadline)
                item['precision']=mode;item['packed_adapter_bytes']=payload_bytes
                episode_json=args.out/(f'ql023-{mode}-A-greedy-s24-9803-{family}-{args.max_turns}.json')
                receipt=json.loads(episode_json.read_text())
                real_forwards=receipt['decisions']
                seconds=sum(x['model_forward_ms'] for x in real_forwards)/1000
                processed=sum(len(tok(g.g.rich_prompt(x['obs']),add_special_tokens=False)['input_ids'])
                              for x in real_forwards)
                item['input_prompt_tokens']=processed
                item['full_forward_input_tokens_per_second']=round(processed/max(seconds,1e-9),2)
                item['model_forward_time_seconds']=round(seconds,4)
                item['tokens_per_second_scope']='prefill prompt tokens/full model forward; NOT autoregressive generation tps'
                # FP32 control must reproduce historical original greedy states/logits.
                if mode=='fp32':
                    with zipfile.ZipFile(args.golden) as z:
                        prior=json.loads(z.read(f'original-experiment/gp350006-s24-9803-{family}.json'))
                    insist(len(prior['decisions'])>=len(real_forwards),'historical baseline too short')
                    delta=0.0
                    for new,old in zip(real_forwards,prior['decisions']):
                        insist(new['obs']==old['obs'] and new['applied']==old['applied'],
                               'FP32 historical on-policy prefix mismatch')
                        delta=max(delta,max(abs(float(a)-float(b)) for a,b in zip(
                            new['six_logits'],old['six_logits'])))
                    insist(delta<0.25,'FP32 historical logit drift')
                    item['golden_fp32_prefix_max_logit_delta']=delta
                summary.append(item)
                print(json.dumps({'QL023_EPISODE':item},sort_keys=True),flush=True)
        insist(before==c.base_fingerprint(model),'full LFM root mutated')
    output={'schema':'driv3r.quant_lab.ql023/1','status':'REAL_FULL_MODEL_COMPLETE',
            'model':MODEL,'revision':REVISION,'source_checkpoint_sha256':ADAPTER_SHA,
            'packed_archive_sha256':PACKED_SHA,'source_host_W0_blob':c.W0_BLOB,
            'candidate_modes':list(MODES),'world_size':24,'world_seed':9803,
            'heldout_worlds_only':True,'native_cybersand':False,
            'full_model_forward_decisions':sum(x['executed'] for x in summary),
            'host_illegal_actions':sum(x['host_illegal'] for x in summary),
            'raw_illegal_preferences':sum(x['raw_illegal_preferences'] for x in summary),
            'outcomes':summary,'cpu_seconds':time.process_time(),
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'full_model_root_safetensors_file_bytes':sum(x.stat().st_size for x in
                       args.cache.rglob('*.safetensors') if '.cache' not in x.parts),
            'no_accelerated_quantized_kernel':True,
            'synthetic_host_not_native_cybersand':True,
            'golden_original_gp350012_sha256':historical}
    (args.out/'summary.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True,help='QL022 local evidence ZIP, verified internally')
    p.add_argument('--preflight',action='store_true',help='Codec/metadata custody ONLY, no model inference')
    p.add_argument('--out',type=Path)
    p.add_argument('--w0',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--source',type=Path,help='Pinned GP350-006 source artifact ZIP')
    p.add_argument('--golden',type=Path,help='Pinned actual GP350-012 GP006 on-policy ZIP')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--max-turns',type=int,default=16)
    p.add_argument('--deadline',type=int,default=1800)
    args=p.parse_args()
    packed,rows,meta=read_bundle(args.bundle)
    reports=[]
    for mode in MODES:
        decoded,num_bytes=unpack_model(mode,packed,rows,meta)
        reports.append({'mode':mode,'modules':len(decoded),'packed_adapter_bytes':num_bytes})
    report={'schema':'driv3r.quant_lab.ql023-preflight/1','trained_source':MODEL,
            'revision':REVISION,'checkpoint_sha256':ADAPTER_SHA,
            'QL022_archive_sha256':PACKED_SHA,'variants_scanned':312,
            'preflight':reports,'real_full_model_forwards':0,
            'closed_loop_game_actions':0,'not_GGUF_IQ':True}
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)
    if not args.preflight:
        try:
            run(args,packed,rows,meta)
        except BaseException as e:
            if args.out:
                args.out.mkdir(exist_ok=True,parents=True)
                (args.out/'failure.json').write_text(json.dumps({
                    'error':type(e).__name__,'message':str(e),
                    'real_full_model_complete':False,'retain_partial_evidence':True},indent=2)+'\n')
            raise


if __name__=='__main__':main()
