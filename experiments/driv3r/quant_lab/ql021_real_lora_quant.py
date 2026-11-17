#!/usr/bin/env python3
"""QL-021: actual pinned GP350-006 LFM350 LoRA weight quantisation.

NOT full-LFM inference, GGUF IQx, native CyberSand, or full-model speedup.
Source ZIP must be immutable historical GP350-006 Actions artifact #14199280586.
Q1-Q4 and mixed apply to actual learned adapter tensors. No synthetic score surrogate.
"""
import argparse, hashlib, io, json, os, resource, statistics, struct, time, zipfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load as safetensors_load

EXPECTED = {32:'5231a7f07c5362e2e0c31051a6ef6ee08bd35e5533ec89359699877342ec5370',64:'85fc33c6b3d5f1520f8e0d8b15a2850c8c8ba65980c2d2d970b717cc376dd3d1',96:'436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c'}
ROOT='LiquidAI/LFM2.5-350M@9921cd4d82c0e3986887aa4b376f9446729ef8a3'
ACTIONS=('STRAIGHT','LEFT','RIGHT','RETRACE','BRANCH','STOP')
SEED=20261117

def sha(b):return hashlib.sha256(b).hexdigest()
def read_source(path):
    z=zipfile.ZipFile(path); ds={}; meta=json.loads(z.read('original-experiment/model-download.json'))
    assert meta['model']=='LiquidAI/LFM2.5-350M' and meta['revision']==ROOT.rsplit('@',1)[1]
    for step,digest in EXPECTED.items():
        data=z.read(f'original-experiment/adapter-step-{step}/adapter_model.safetensors')
        assert sha(data)==digest,(step,'bad source checksum')
        raw=safetensors_load(data); assert len(raw)==24
        ds[step]={k:v.float().contiguous() for k,v in raw.items()}
    z.close();return ds,meta

def pack_codes(codes,bits):
    vals=np.asarray(codes,dtype=np.uint8).ravel(); assert np.all(vals < (1<<bits))
    out=bytearray((len(vals)*bits+7)//8)
    for i,v in enumerate(vals):
        q=i*bits; out[q>>3]|=(int(v)<<(q&7))&255
        if (q&7)+bits>8:out[(q>>3)+1]|=int(v)>>(8-(q&7))
    return bytes(out)

def unpack_codes(data,bits,n):
    assert len(data)==(n*bits+7)//8
    arr=np.zeros(n,dtype=np.uint8);mask=(1<<bits)-1
    for i in range(n):
        q=i*bits;v=data[q>>3]>>(q&7)
        if(q&7)+bits>8:v|=data[(q>>3)+1]<<(8-(q&7))
        arr[i]=v&mask
    return arr

def quant(w,bits,group=64):
    x=w.detach().cpu().numpy().astype(np.float32);m,n=x.shape
    pad=(-n)%group;g=np.pad(x,((0,0),(0,pad))).reshape(m,-1,group)
    if bits==1:
        scales=np.mean(np.abs(g),axis=2).astype(np.float16)
        code=(g>=0).astype(np.uint8)
    else:
        scales=np.max(np.abs(g),axis=2).astype(np.float16)
        levels=(1<<bits)-1
        code=np.rint((np.clip(g/np.where(scales[:,:,None]>0,scales[:,:,None].astype(np.float32),1),-1,1)+1)*levels/2).astype(np.uint8)
    raw=pack_codes(code,bits)
    rec=unpack_codes(raw,bits,code.size).reshape(code.shape)
    assert np.array_equal(rec,code)
    if bits==1:y=(rec.astype(np.float32)*2-1)*scales[:,:,None].astype(np.float32)
    else:y=(2*rec.astype(np.float32)/levels-1)*scales[:,:,None].astype(np.float32)
    y=np.where(scales[:,:,None]==0,0,y).reshape(m,-1)[:,:n]
    blob=struct.pack('<BBBBII',81,76,bits,group,m,n)+scales.tobytes()+raw
    assert len(blob)==12+scales.size*2+len(raw)
    return torch.from_numpy(y.copy()),blob

def quant_ste(w,bits,group=64):
    m,n=w.shape;pad=(-n)%group
    t=torch.nn.functional.pad(w,(0,pad)).reshape(m,-1,group)
    if bits==1:
        scale=t.detach().abs().mean(-1,keepdim=True).clamp_min(1e-9)
        y=torch.where(t>=0,scale,-scale)
    else:
        scale=t.detach().abs().amax(-1,keepdim=True).clamp_min(1e-9)
        lv=(1<<bits)-1;y=(2*torch.round((torch.clamp(t/scale,-1,1)+1)*lv/2)/lv-1)*scale
    return (t+(y-t).detach()).reshape(m,-1)[:,:n]

def pairs(state):
    out=[]
    for k in state:
        if '.lora_A.weight' in k:
            b=k.replace('.lora_A.weight','.lora_B.weight');assert b in state
            out.append((k[:-len('.lora_A.weight')],state[k],state[b]))
    assert len(out)==12
    return sorted(out)

def activations(n,seed,count=96,sparse=False):
    r=np.random.default_rng(seed)
    a=r.normal(0,1,(count,n)).astype(np.float32)
    if sparse:a*=r.uniform(size=a.shape)<0.25
    return torch.from_numpy(a)

def delta(x,a,b):return (x@a.T)@b.T*2.0

def relative(x,y):
    num=torch.mean((x-y)**2).sqrt().item();den=max(torch.mean(x**2).sqrt().item(),1e-10)
    cos=torch.nn.functional.cosine_similarity(x.reshape(1,-1),y.reshape(1,-1)).item()
    return {'relative_rms_error':round(num/den,7),'cosine':round(cos,7)}

def repair(a,b,bits=2,steps=60,seed=SEED):
    # True optimizer updates on the real trained LoRA A/B matrices; teacher is FP32 LoRA.
    torch.manual_seed(seed)
    x=activations(a.shape[1],seed+17,96)
    ref=delta(x,a,b).detach()
    aa=torch.nn.Parameter(a.detach().clone());bb=torch.nn.Parameter(b.detach().clone())
    opt=torch.optim.AdamW([aa,bb],lr=3e-6,weight_decay=0.)
    for _ in range(steps):
        opt.zero_grad();y=delta(x,quant_ste(aa,bits),quant_ste(bb,bits));loss=(y-ref).square().mean();loss.backward();opt.step()
    qa,rawa=quant(aa.detach(),bits);qb,rawb=quant(bb.detach(),bits)
    return qa,qb,rawa,rawb,float(loss.item())

def main(args):
    torch.set_num_threads(1)
    allmodels,meta=read_source(args.source)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    corpus=[];full_payload={};repaired={}
    modes=('fp32','q4','q3','q2','q1','A16_B2','A2_B16','Q2Q4_v','sparse_rank2_q2','q2_repair')
    for step,state in allmodels.items():
        for name,a,b in pairs(state):
            xf=activations(a.shape[1],SEED+step+31,96)
            xs=activations(a.shape[1],SEED+step+101,96,True)
            y=delta(xf,a,b);ys=delta(xs,a,b)
            norms=(b.norm(dim=0)*a.norm(dim=1)).detach().numpy()
            for mode in modes:
                if mode=='fp32':qa,qb=a,b;ba=a.numel()*4;bb=b.numel()*4;raw=b''
                elif mode=='q2_repair':
                    qa,qb,ra,rb,loss=repair(a,b,2,steps=60,seed=SEED+step)
                    if step==96: repaired[name]=(qa,qb)
                    ba=len(ra);bb=len(rb);raw=ra+rb
                elif mode=='sparse_rank2_q2':
                    aa=a.clone();bc=b.clone();idx=np.argsort(norms)[:2];aa[idx,:]=0;bc[:,idx]=0
                    qa,ra=quant(aa,2);qb,rb=quant(bc,2);ba=len(ra);bb=len(rb);raw=ra+rb
                else:
                    if mode.startswith('q') and mode[1:].isdigit():abits=bbits=int(mode[1:])
                    elif mode=='A16_B2':abits,bbits=16,2
                    elif mode=='A2_B16':abits,bbits=2,16
                    elif mode=='Q2Q4_v':abits=bbits=4 if '.v_proj' in name else 2
                    else:raise ValueError(mode)
                    if abits==16:qa=a.half().float();ra=a.half().numpy().tobytes()
                    else:qa,ra=quant(a,abits)
                    if bbits==16:qb=b.half().float();rb=b.half().numpy().tobytes()
                    else:qb,rb=quant(b,bbits)
                    ba=len(ra);bb=len(rb);raw=ra+rb
                yf=delta(xf,qa,qb);ysp=delta(xs,qa,qb)
                row={'step':step,'module':name,'mode':mode,'payload_bytes':ba+bb,
                    'gaussian':relative(y,yf),'sparse':relative(ys,ysp),
                    'params':a.numel()+b.numel()}
                if mode=='q2_repair':row['updates']=60;row['final_train_mse']=round(loss,9)
                corpus.append(row)
                if step==96 and mode!='fp32':full_payload[(mode,name)]=raw
    models=[]
    for mode in modes:
        rows=[r for r in corpus if r['step']==96 and r['mode']==mode]
        assert len(rows)==12
        mean_err=statistics.mean(r['gaussian']['relative_rms_error'] for r in rows)
        total_bytes=sum(r['payload_bytes'] for r in rows)
        models.append({'mode':mode,'payload_bytes':total_bytes,'ratio_to_fp32':round(sum(r['params']*4 for r in rows)/total_bytes,3),
                       'mean_module_relative_rms_error':round(mean_err,7),'max_module_relative_rms_error':round(max(r['gaussian']['relative_rms_error'] for r in rows),7),
                       'mean_sparse_relative_rms_error':round(statistics.mean(r['sparse']['relative_rms_error'] for r in rows),7)})
    # Host-independent neural adapter operator timing; decodes stored bits to resident FP32 (not low-bit execution)
    lat=[]
    for mode in ('fp32','q4','q2','q1','A16_B2','q2_repair'):
        vals=[]
        for name,a,b in pairs(allmodels[96]):
            if mode=='fp32':qa,qb=a,b
            elif mode=='A16_B2':qa=a.half().float();qb,_=quant(b,2)
            elif mode=='q2_repair':qa,qb=repaired[name]
            else:
                bits=2 if mode=='q2_repair' else int(mode[1:]);qa,_=quant(a,bits);qb,_=quant(b,bits)
            x=activations(a.shape[1],SEED,1)
            for _ in range(4):delta(x,qa,qb)
            for _ in range(20):
                t=time.perf_counter_ns();delta(x,qa,qb);vals.append((time.perf_counter_ns()-t)/1e6)
        lat.append({'mode':mode,'p50_ms':round(float(np.percentile(vals,50)),6),'p95_ms':round(float(np.percentile(vals,95)),6),'n':len(vals)})
    sourcebytes=Path(args.source).read_bytes()
    report={'schema':'QL021_real_trained_LFM350_adapter_weight_quant/v1','model':ROOT,'historical_source_run':43806503356,
            'historical_artifact_id':14199280586,'artifact_zip_sha256':sha(sourcebytes),
            'adapter_sha256_by_step':EXPECTED,'model_root_available':False,
            'full_model_forwards':0,'closed_loop_episodes':0,'legal_action_rate':None,'tokens_per_second':None,
            'no_model_root_reason':'LFM350 parent model.safetensors/tokenizer unavailable to isolated runner; adapter only',
            'quantization':'true bitpacked uniform signed codes with FP16 per-64-element group scales, no GGUF IQ codecs',
            'probe':'96 Gaussian and 96 sparse frozen feature probes per module, per step; no game policy predictions',
            'repair':'60 AdamW real LoRA A/B STE-Q2 updates per module at each step; FP32 LoRA teacher, no game-supervised claim',
            'adapter_tensors':24,'modules_per_step':12,'models_step96':models,'timing_adapter_ops_only':lat,
            'max_rss_kib_shared_process':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    (out/'per-module.json').write_text(json.dumps(corpus,indent=2,sort_keys=True))
    with zipfile.ZipFile(out/'packed-adapter-variants.zip','w',zipfile.ZIP_DEFLATED,6) as z:
        for (mode,name),b in sorted(full_payload.items()):
            info=zipfile.ZipInfo(f'{mode}/{name.replace("base_model.model.model.","")}.bin',date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b,compresslevel=6)
    receipt={'results.json':sha((out/'results.json').read_bytes()),'per-module.json':sha((out/'per-module.json').read_bytes()),'packed-adapter-variants.zip':sha((out/'packed-adapter-variants.zip').read_bytes())}
    (out/'SHA256SUMS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True))
    print(json.dumps({'models':models,'timing':lat,'max_rss':report['max_rss_kib_shared_process'],'out':str(out)},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);main(p.parse_args())
