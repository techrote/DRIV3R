"""P19 original-producer blind comparison and *model-derived* preference pilot.

This does not generate model rollouts, infer human ratings, or validate native CyberSand.
P18's fixed producer and P10's frozen-W16 sources are independently hash-checked.
Training is a small torch CPU probe of explicitly assistant-generated visual ratings.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, tempfile, zipfile, shutil
from pathlib import Path
import numpy as np

P18_SHA = 'b6651d1747b7bbe18469d98a08c7147f8ccc61ea54506cce5488918fb6ac13de'
P10_SHA = 'ab0f24fd2ffe3c9c30e5bc77f8f66bea0e823b9776bab3f5811b7fefab236fa6'
NICHES = ('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST')
FEATURES = ('turn90_rate','longest_straight','reversal2','repeated_edge_rate','xspan','yspan','net_displacement','unique_cells','junction_motifs','cycles','hooks','nonlocal_steps')

def sha(b): return hashlib.sha256(b).hexdigest()
def checked_zip(path, expected):
    p=Path(path)
    actual=sha(p.read_bytes())
    if actual!=expected: raise ValueError(f'{p.name}: SOURCE SHA-256 MISMATCH {actual} != {expected}')
    return zipfile.ZipFile(p)

def sources(p18_file, p10_file):
    import p18_reverify_ballots as p18
    import p17_cross_model as geo
    with checked_zip(p18_file, P18_SHA) as p18_zip, tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); manifest=tmp/'manifest.json'
        manifest.write_bytes(p18_zip.read('P18/driv3r_wormpaths_source_manifest.json'))
        aroot=tmp/'artifacts'; aroot.mkdir()
        for name in p18_zip.namelist():
            if name.startswith('P18/original_historical_artifacts/') and name.endswith('.zip'):
                (aroot/Path(name).name).write_bytes(p18_zip.read(name))
        originals=p18.verify(manifest,aroot)
        basepairs=p18.paired(originals)
        lf, extra=p18.verify_lfm_006(aroot/'gp350006-step96.zip')
        assert lf['paired_worlds']==9 and lf['same_path']==9 and lf['different_logit_ticks']==216
        historical=[]
        for p in basepairs:
            a,b=p['first'],p['second'];assert a['positions']!=b['positions']
            historical.append({'id':p['id'],'cohort':'P18_HISTORICAL','block':':'.join(map(str,p['world_block'])),
                 'family':str(p['world_block'][1]).split('/')[0].strip().lower(),
                 'a':a,'b':b,'original_source_hashes':[a['source_json_sha256'],b['source_json_sha256']]})
        assert len(historical)==10
    w16=[]
    with checked_zip(p10_file,P10_SHA) as p10_zip:
        entries=sorted(f for f in p10_zip.namelist() if f.startswith('new-rollouts/') and f.endswith('-late-intact.json'))
        assert len(entries)==27
        for item in entries:
            other=item.replace('-late-intact.json','-open-intact.json')
            ar=p10_zip.read(item); br=p10_zip.read(other)
            a=json.loads(ar);b=json.loads(br)
            assert a['family']==b['family'] and a['seed']==b['seed']
            for d in (a,b):
                assert d['contract']=='cybersand.worm.observation-action/0-experimental'
                assert len(d['events'])+1==len(d['trajectory'])
                for i,e in enumerate(d['events']):
                    assert tuple(e['position_after'])==tuple(d['trajectory'][i+1])
                    assert (e['observation']['x'],e['observation']['y'])==tuple(d['trajectory'][i])
                    assert e['applied'] in e['observation']['legal']
                    assert e['observation']['tick']==i
            if a['trajectory']==b['trajectory']: continue
            pair_id='P19-'+sha((sha(ar)+sha(br)).encode())[:12]
            ra={'name':'anonymous original gated','group':'w16','world':f'{a["family"]}/{a["seed"]}',
                'decisions':len(a['events']),'source_json_sha256':sha(ar),'positions':a['trajectory'],
                'ink':[True]*len(a['events']),'trace':item}
            rb={'name':'anonymous original open','group':'w16','world':f'{b["family"]}/{b["seed"]}',
                'decisions':len(b['events']),'source_json_sha256':sha(br),'positions':b['trajectory'],
                'ink':[True]*len(b['events']),'trace':other}
            # Deterministic alternate balanced side assignment; the whole pair ID is independent of side.
            if int(pair_id[-2:],16)%2: ra,rb=rb,ra
            w16.append({'id':pair_id,'cohort':'P10_W16','block':f'{a["family"]}/{a["seed"]}',
                'family':a['family'],'a':ra,'b':rb,'original_source_hashes':[ra['source_json_sha256'],rb['source_json_sha256']]})
    assert len(w16)==18
    pairs=historical+w16
    for p in pairs:
        p['source_pair_id']=p['id']
        p['id']='B19-'+sha(p['source_pair_id'].encode())[:12]
    pairs=sorted(pairs,key=lambda p:p['id'])
    assert len({p['id'] for p in pairs})==28
    for p in pairs:
        for k in ('a','b'):
            p[k]['geometry']=geo.geometry(p[k]['positions'])
    return pairs, lf

def pair_svg(pairs):
    import p18_reverify_ballots as p18
    pages=[]
    for page in range(math.ceil(len(pairs)/2)):
        subset=[]
        for q in pairs[2*page:2*page+2]:
            subset.append({'id':q['id'],'first':q['a'],'second':q['b']})
        pages.append(p18.svg_page(subset,page))
    return pages

def slim(r):
    return {k:r[k] for k in ('name','group','world','decisions','source_json_sha256','geometry')}

def preview(pairs,lf,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for idx,page in enumerate(pair_svg(pairs),1):
        (out/f'P19-BLIND-A4-{idx:02}.svg').write_text(page)
    with (out/'P19_BLANK_BALLOT.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','niche','vote_A_B_TIE_SKIP','rating_source','comment'])
        writer.writeheader()
        for pair in pairs:
            for n in NICHES:writer.writerow({'pair_id':pair['id'],'niche':n})
    reveal={'schema':'driv3r.preference_eval.p19.reveal/1','caveat':'open only after visual ratings; nonidentical pairs only; no human preferences',
        'lfm_9_pairs_visual_tie':lf['same_path'],
        'pairs':[{'id':p['id'],'cohort':p['cohort'],'block':p['block'],'family':p['family'],
                  'source_pair_id':p['source_pair_id'],'A':slim(p['a']),'B':slim(p['b'])} for p in pairs]}
    (out/'P19_REVEAL_AFTER_RATING.json').write_text(json.dumps(reveal,indent=2,sort_keys=True)+'\n')
    return reveal

def feat(r):
    g=r['geometry']; steps=max(1,r['decisions'])
    return np.array([g['turn90_rate'], g['longest_straight']/steps, g['reversal2']/steps,
        g['repeated_edge_rate'],g['xspan']/steps,g['yspan']/steps,g['net_displacement']/steps,
        g['unique_cells']/steps,g['junction_motifs']/steps,g['cycles']/steps,g['hooks']/steps,
        g['nonlocal_steps']/steps],dtype=np.float64)

def baselines(p,n):
    a,b=p['a']['geometry'],p['b']['geometry']
    if n=='MEANDERER':
        def score(m):return m['turn90_rate']*(m['xspan']+m['yspan'])*(1-m['repeated_edge_rate'])
    elif n=='CURLER':
        def score(m):return m['hooks']+3*m['cycles']
    elif n=='BRANCHER':
        def score(m):return m['junction_motifs']+3*m['cycles']
    else:
        def score(m):return m['turn90_rate']*m['unique_cells']*(1-m['repeated_edge_rate'])
    q=score(a)-score(b)
    return 'A' if q>1e-8 else 'B' if q< -1e-8 else 'TIE'

def fit_logit(train,epochs=200):
    # Low-capacity ridge-logistic ranker. A/B order is antisymmetric by construction.
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(19)
    x=np.stack([e[0] for e in train]); y=np.array([e[1] for e in train],dtype=np.float32)
    scale=np.maximum(np.std(x,axis=0),.12)
    xx=torch.tensor(x/scale,dtype=torch.float32)
    yy=torch.tensor(y,dtype=torch.float32)
    w=torch.nn.Parameter(torch.zeros(xx.shape[1]))
    opt=torch.optim.Adam([w],lr=.03)
    for _ in range(epochs):
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(xx@w,yy)+0.25*(w*w).sum()
        loss.backward();opt.step()
    return w.detach().numpy(),scale

def fit_report(pairs,votes,out):
    from collections import Counter
    import time
    pair_index={p['id']:p for p in pairs}
    with open(votes,newline='') as f: read=list(csv.DictReader(f))
    assert len(read)==len(pairs)*4 and len({(r['pair_id'],r['niche']) for r in read})==len(read)
    dataset=[]
    for r in read:
        assert r['rating_source']=='GPT6_ASSISTANT_VISUAL_PILOT_NOT_HUMAN'
        assert r['niche'] in NICHES and r['pair_id'] in pair_index
        assert r['vote_A_B_TIE_SKIP'] in ('A','B','TIE','SKIP')
        p=pair_index[r['pair_id']]
        x=feat(p['a'])-feat(p['b'])
        dataset.append({'p':p,'x':x,'y':{'A':1,'B':0}.get(r['vote_A_B_TIE_SKIP']),
                        'vote':r['vote_A_B_TIE_SKIP'],'niche':r['niche']})
    holdout={'train_on_W16_test_historical':('P10_W16','P18_HISTORICAL'),
             'train_historical_test_W16':('P18_HISTORICAL','P10_W16')}
    result={'schema':'driv3r.preference_eval.p19.visual_model_pilot/1','labels':'MODEL_DERIVED_ASSISTANT_VISUAL_ONLY_NOT_HUMAN',
            'sampled_original_pairs':len(pairs),'model_derived_slots':len(dataset),'vote_counts':dict(Counter(e['vote'] for e in dataset)),
            'no_new_full_slm_forwards':True,'native_cybersand_acceptance':False,'source_disjoint_train_test':holdout,
            'model':'12-feature torch CPU ridge logistic, pairwise antisymmetric, no root-SLM training',
            'visual_pilot_leakage':'Ratings were made from earlier images whose pair ID prefixes signalled original source cohort (P18/P19). Final B19 galleries remove this cue but do not retrospectively blind the same reviewer.',
            'evaluations':{},'caveat':'No tuning on heldout; identical GP350 stock/LoRA A/B omitted; small clustered/nonindependent model labels.'}
    for niche in NICHES:
        result['evaluations'][niche]={}
        for split,(tr,te) in holdout.items():
            train=[(e['x'],e['y']) for e in dataset if e['niche']==niche and e['p']['cohort']==tr and e['y'] is not None]
            test=[e for e in dataset if e['niche']==niche and e['p']['cohort']==te and e['y'] is not None]
            if len(train)<2 or not test:
                result['evaluations'][niche][split]={'status':'INSUFFICIENT_LABELS','n_train':len(train),'n_test':len(test)};continue
            start=time.perf_counter();w,scale=fit_logit(train)
            latency=(time.perf_counter()-start)*1000
            ck_dir=out.parent/'P19_LOCAL_CPU_CHECKPOINTS_NOT_ACTIONS';ck_dir.mkdir(exist_ok=True)
            ck_path=ck_dir/f'{niche}_{split}.npz'
            np.savez_compressed(ck_path,weights=w.astype('<f4'),scale=scale.astype('<f8'),niche=niche,split=split)
            model_correct=0;proxy_correct=0;novel_correct=0
            preds=[]
            for e in test:
                pred='A' if ((e['x']/scale)@w)>=0 else 'B'
                baseline=baselines(e['p'],niche)
                ng=e['p']['a']['geometry']['unique_cells']-e['p']['b']['geometry']['unique_cells']
                novelty='A' if ng>0 else 'B' if ng<0 else 'TIE'
                model_correct+=pred==e['vote']
                proxy_correct+=baseline==e['vote']
                novel_correct+=novelty==e['vote']
                preds.append({'id':e['p']['id'],'actual':e['vote'],'model':pred,'niche_rule':baseline,'novelty':novelty})
            result['evaluations'][niche][split]={'status':'CPU_TRAINED_EVALUATED','n_train':len(train),'n_test':len(test),
                'model_correct':model_correct,'transparent_niche_rule_correct':proxy_correct,'coverage_rule_correct':novel_correct,
                'optimizer_updates':200,'training_ms':round(latency,2),'weights_sha256':sha(np.asarray(w,dtype='<f4').tobytes()),
                'local_checkpoint_sha256':sha(ck_path.read_bytes()),'checkpoint_immutable_actions_artifact':False,
                'predictions':preds}
    out=Path(out);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--p18',required=True);ap.add_argument('--p10',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--votes')
    args=ap.parse_args()
    pairs,lf=sources(args.p18,args.p10)
    preview(pairs,lf,args.output)
    if args.votes:
        print(json.dumps(fit_report(pairs,args.votes,Path(args.output)/'P19_MODEL_PILOT_RESULTS.json')['evaluations'],indent=2))
    else: print('28 source-original nonidentical pairs; 14 A4 pages; 112 BLANK model/human slots. NO RATINGS OR TRAINING.')

if __name__=='__main__':main()