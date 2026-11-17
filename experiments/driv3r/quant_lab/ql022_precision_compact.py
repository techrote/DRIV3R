#!/usr/bin/env python3
"""QL-022. Real GP350-006 trained LFM350 LoRA; no parent-model inference.

Repack tail-padded 64-wide Q1-Q4 code blocks losslessly; globally allocate A/B
precision using TRAIN probes, evaluate on disjoint HELDOUT probes; test Q2
rank-gauge QAT. Custom bitpack is NOT GGUF/IQ or accelerated inference.
"""
import argparse,hashlib,json,math,resource,statistics,time,zipfile
from pathlib import Path
import numpy as np
import torch
from ql021_real_lora_quant import read_source,pairs,sha,pack_codes,unpack_codes,delta,quant,quant_ste
SOURCE='1db95314355fb4ea0439b63825a0f829b14bb0a0f8b01284fd614446aca85a47'
SEED=22017

def compact(w,b):
    if b==16:
        y=w.half().float();return y,b'QH'+np.array(w.shape,dtype='<u4').tobytes()+w.half().numpy().tobytes()
    y,old=quant(w,b);m,n=w.shape;groups=(n+63)//64
    scales=old[12:12+2*m*groups]
    codes=unpack_codes(old[12+2*m*groups:],b,m*groups*64).reshape(m,groups*64)[:,:n]
    blob=old[:12]+scales+pack_codes(codes,b)
    assert len(blob)==12+2*m*groups+(m*n*b+7)//8
    saved=unpack_codes(blob[12+2*m*groups:],b,m*n)
    assert np.array_equal(saved.reshape(m,n),codes)
    return y,blob

def probes(n,seed,num=72):
    rng=np.random.default_rng(seed)
    g=rng.normal(size=(num,n)).astype('f4')
    s=rng.normal(size=(num,n)).astype('f4');s*=rng.random(s.shape)<.20
    c=rng.normal(size=(num,n)).astype('f4');c=(c+np.roll(c,1,1)+np.roll(c,-1,1))/(3**.5)
    return torch.from_numpy(np.concatenate([g,s,c]))

def mse(x,y):return float((x-y).square().mean()/x.square().mean().clamp_min(1e-12))

def gauge(a,b,seed):
    torch.manual_seed(seed);x=probes(a.shape[1],seed+100,60);ref=delta(x,a,b).detach()
    ls=torch.nn.Parameter(torch.zeros(a.shape[0]));opt=torch.optim.AdamW([ls],lr=.025,weight_decay=0.)
    traces=[]
    for i in range(72):
        opt.zero_grad();s=ls.clamp(-1.2,1.2).exp()
        z=delta(x,quant_ste(a*s[:,None],2),quant_ste(b/s[None,:],2))
        loss=(z-ref).square().mean()/ref.square().mean().clamp_min(1e-12)
        loss.backward();opt.step()
        if i in (0,36,71):traces.append(round(float(loss.detach()),8))
    s=ls.detach().clamp(-1.2,1.2).exp()
    assert torch.allclose(delta(x,a*s[:,None],b/s[None,:]),ref,rtol=2e-5,atol=2e-5)
    return s,traces

def allocate(rows,budget):
    cap=budget//256;dp={0:(0.,[])}
    for r in rows:
        nd={}
        for u,(v,choice) in dp.items():
            for j,c in enumerate(r['candidates']):
                key=u+(c['bytes']+255)//256
                if key<=cap and (key not in nd or v+c['train_mse']<nd[key][0]):
                    nd[key]=(v+c['train_mse'],choice+[j])
        assert nd;dp=nd
    best=min(dp,key=lambda v:dp[v][0]);sel=[r['candidates'][j] for r,j in zip(rows,dp[best][1])]
    return {'budget_bytes':budget,'actual_bytes':sum(c['bytes'] for c in sel),
            'train_mse':statistics.mean(c['train_mse'] for c in sel),
            'heldout_mse':statistics.mean(c['heldout_mse'] for c in sel),
            'A_B_bits':[[c['A'],c['B']] for c in sel]}

def main(source,out):
    torch.set_num_threads(1)
    assert sha(Path(source).read_bytes())==SOURCE
    allstates,_=read_source(source);modules=pairs(allstates[96]);assert len(modules)==12
    output=Path(out);output.mkdir(parents=True,exist_ok=False)
    rows=[];repair=[];variants={};perf=[]
    for j,(name,a,b) in enumerate(modules):
        xt=probes(a.shape[1],SEED+110+j);xv=probes(a.shape[1],SEED+220+j)
        yt=delta(xt,a,b);yv=delta(xv,a,b)
        stored={bits:(compact(a,bits),compact(b,bits)) for bits in (1,2,3,4,16)}
        candidates=[]
        for ab in (1,2,3,4,16):
            for bb in (1,2,3,4,16):
                qa,ra=stored[ab][0];qb,rb=stored[bb][1]
                candidates.append({'A':ab,'B':bb,'bytes':len(ra)+len(rb),
                    'train_mse':mse(yt,delta(xt,qa,qb)),
                    'heldout_mse':mse(yv,delta(xv,qa,qb))})
                variants[f'{j:02d}/A{ab}_B{bb}.bin']=ra+rb
        s,trace=gauge(a,b,SEED+j)
        ga,ba=compact(a*s[:,None],2);gb,bb=compact(b/s[None,:],2)
        variants[f'{j:02d}/q2_gauge.bin']=ba+bb
        before=next(c for c in candidates if c['A']==c['B']==2)
        repair.append({'module':name,'updates':72,'train_trace':trace,'rank_scales':s.tolist(),
                       'heldout_q2_mse_before':before['heldout_mse'],
                       'heldout_q2_mse_after':mse(yv,delta(xv,ga,gb))})
        rows.append({'name':name,'candidates':candidates})
    def policy(name,pick):
        cs=[next(c for c in r['candidates'] if (c['A'],c['B'])==pair) for r,pair in zip(rows,pick)]
        return {'name':name,'bytes':sum(c['bytes'] for c in cs),
          'heldout_mse':statistics.mean(c['heldout_mse'] for c in cs)}
    baseline={f'Q{b}':policy(f'Q{b}',[(b,b)]*12) for b in (1,2,3,4,16)}
    baseline['Q2Q4v']=policy('Q2Q4v',[(4,4) if '.v_proj' in r['name'] else (2,2) for r in rows])
    baseline['A16B2']=policy('A16B2',[(16,2)]*12)
    baseline['A2B16']=policy('A2B16',[(2,16)]*12)
    adaptive={'Q2_budget':allocate(rows,baseline['Q2']['bytes']),
              'Q2Q4v_budget':allocate(rows,baseline['Q2Q4v']['bytes'])}
    for label in ('FP32','Q4','Q2','Q1','adaptive_mix','Q2_gauge'):
        times=[]
        for j,(_,a,b) in enumerate(modules):
            if label=='FP32':qa,qb=a,b
            elif label=='adaptive_mix':
                aa,bb=adaptive['Q2Q4v_budget']['A_B_bits'][j];qa=compact(a,aa)[0];qb=compact(b,bb)[0]
            elif label=='Q2_gauge':
                s=torch.tensor(repair[j]['rank_scales']);qa=compact(a*s[:,None],2)[0];qb=compact(b/s[None,:],2)[0]
            else:
                k=int(label[1:]);qa=compact(a,k)[0];qb=compact(b,k)[0]
            x=probes(a.shape[1],SEED+999+j,1)[:1]
            for _ in range(4):delta(x,qa,qb)
            for _ in range(24):
                t=time.perf_counter_ns();delta(x,qa,qb);times.append((time.perf_counter_ns()-t)/1e6)
        perf.append({'model':label,'n':len(times),'p50_ms':float(np.percentile(times,50)),
                     'p95_ms':float(np.percentile(times,95))})
    summary={'experiment':'QL-022','model':'LiquidAI/LFM2.5-350M','model_revision':'9921cd4d82c0e3986887aa4b376f9446729ef8a3',
     'historic_source_artifact':14199280586,'source_archive_sha256':SOURCE,
     'step96_adapter_sha256':'436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c',
     'training_probe_seeds':[SEED+110+j for j in range(12)],'heldout_probe_seeds':[SEED+220+j for j in range(12)],
     'baseline':baseline,'adaptive':adaptive,'Q2_gauge_QAT_optimizer_updates':72*12,
     'gauge_heldout_MSE_before':statistics.mean(x['heldout_q2_mse_before'] for x in repair),
     'gauge_heldout_MSE_after':statistics.mean(x['heldout_q2_mse_after'] for x in repair),
     'same_cpu_runner_LoRA_only_latency':perf,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     'full_model_forwards':0,'new_host_legal_game_decisions':0,'closed_loop_episodes':0,'GGUF_IQx':False,
     'caveat':'Actual trained LoRA matrices but SYNTHETIC activation probes; no LFM parent root, hidden states, tokenization or CyberSand native game. FP32 matmul after bitpack decoding.'}
    for n,obj in [('results.json',summary),('per-module.json',rows),('gauge-repair.json',repair)]:
        (output/n).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(output/'actual-weight-variants.zip','w',zipfile.ZIP_DEFLATED,6) as z:
        for n,data in sorted(variants.items()):
            zi=zipfile.ZipInfo(n,(1980,1,1,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;z.writestr(zi,data,compresslevel=6)
    (output/'SHA256SUMS.json').write_text(json.dumps({p:sha((output/p).read_bytes()) for p in ['results.json','per-module.json','gauge-repair.json','actual-weight-variants.zip']},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'base':baseline,'adaptive':adaptive,'gauge':{'before':summary['gauge_heldout_MSE_before'],'after':summary['gauge_heldout_MSE_after']},'RSS_KiB':summary['process_peak_RSS_KiB']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    a=parser.parse_args();main(a.source,a.out)
