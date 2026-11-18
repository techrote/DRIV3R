"""P18: original producer-byte audit and anonymous *unlabelled* preference ballots.

Read-only downloaded historical GitHub Actions artifacts; no model inference, no
native acceptance, and no claimed human preferences. 4 niches remain separate.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, html, itertools, json, zipfile
from pathlib import Path

SOURCE_MANIFEST_SHA = '873d208e370bec2d26586af1c86545957deef4b01fc5a9521f3630fd1920a88d'
ARCHIVES = {14147579972:'native-free.zip',14147233975:'native-soft.zip',
 14185385444:'lfm-w13-alpha.zip',14185425004:'lfm-w13-beta.zip',
 14181997061:'mutagen-stock.zip',14183260000:'mutagen-scoop.zip',
 14183965523:'mutagen-gouge.zip',14183591467:'mutagen-scrape.zip',
 14184017182:'franken-greyhound.zip',14183687090:'franken-branchwitch.zip',
 14184395599:'franken-greyhound5.zip',14184295982:'franken-branchwitch8.zip'}
NICHES=('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST')

def digest(b): return hashlib.sha256(b).hexdigest()
def jsonbytes(o): return json.dumps(o,separators=(',',':')).encode()

def verify(source, archive_dir):
    raw=Path(source).read_bytes()
    if digest(raw)!=SOURCE_MANIFEST_SHA: raise ValueError('historical manifest hash mismatch')
    specimens=json.loads(raw)['specimens']; rows=[]
    for s in specimens:
        meta=s['source']; aid=meta['artifact_id']
        if aid not in ARCHIVES: raise ValueError('unknown original artifact')
        archive=Path(archive_dir)/ARCHIVES[aid]
        with zipfile.ZipFile(archive) as z:
            b=z.read(meta['entry'])
        if digest(b)!=meta['json_sha256']: raise ValueError('original producer JSON checksum mismatch: '+s['name'])
        doc=json.loads(b)
        actions=doc if isinstance(doc,list) else doc['events']
        if isinstance(doc,list):
            pos=[actions[0]['cursor_before']]+[e['cursor_after'] for e in actions]
            ink=[bool(e['material_changed']) for e in actions]
        else:
            pos=doc['trajectory'];ink=s['pen_down_segments']
            if doc['steps_executed']!=len(actions): raise ValueError('step budget mismatch')
        if pos!=s['positions'] or ink!=s['pen_down_segments']: raise ValueError('curated position/ink mismatch: '+s['name'])
        if len(pos)!=len(actions)+1 or len(actions)!=s['decisions']:raise ValueError('trajectory length mismatch')
        counts=collections.Counter()
        for i,e in enumerate(actions):
            if isinstance(doc,list):
                if e['index']!=i or e['cursor_before']!=pos[i] or e['cursor_after']!=pos[i+1]:raise ValueError('native receipt order mismatch')
                if e['choice'] not in e['candidate_ids']:raise ValueError('native candidate menu violation')
                if i and actions[i-1]['world_after_sha256']!=e['source_world_sha256']:raise ValueError('native world hash chain mismatch')
                action=e['choice']; counts['native_ink']+=bool(e['material_changed'])
            else:
                obs=e['observation']
                if (obs['x'],obs['y'])!=tuple(pos[i]) or e['position_after']!=pos[i+1] or obs['tick']!=i:raise ValueError('W0 event/position mismatch')
                if e['applied'] not in obs['legal']:raise ValueError('W0 host menu violation')
                action=e['applied'];counts['raw_illegal']+=int(not e.get('raw_legal',True))
            counts[action]+=1
        rows.append({'name':s['name'],'group':s['group'],'world':s['world'],'decisions':len(actions),
            'source_artifact_id':aid,'source_entry':meta['entry'],'source_json_sha256':digest(b),
            'source_zip_sha256':digest(archive.read_bytes()),'positions_sha256':digest(jsonbytes(pos)),
            'positions':pos,'ink':ink,'original_model_or_native_actions':dict(counts),
            'menu_violations':0,'source_level':'ORIGINAL_PRODUCER_BYTES_VERIFIED',
            'host':'NATIVE_HISTORICAL_REFERENCE' if isinstance(doc,list) else 'EXTERNAL_SYNTHETIC_W0'})
    if len(rows)!=14:raise ValueError('not the fixed 14 specimens')
    return rows

def paired(rows):
    block=collections.defaultdict(list)
    for r in rows:block[(r['group'],r['world'],r['decisions'])].append(r)
    pairs=[]
    for key in sorted(block):
        for a,b in itertools.combinations(sorted(block[key],key=lambda r:r['source_json_sha256']),2):
            p_id='P18-'+digest(jsonbytes(sorted([a['source_json_sha256'],b['source_json_sha256']])))[:12]
            pairs.append({'id':p_id,'world_block':list(key),'first':a,'second':b})
    # Deterministic balanced anonymisation; revealed labels stored in separate key only.
    pairs.sort(key=lambda p:digest(p['id'].encode()))
    for i,p in enumerate(pairs):
        if i%2:p['first'],p['second']=p['second'],p['first']
        p['index']=i+1
    return pairs

def path_svg(r,x,y,lo_x,lo_y,scale):
    chunks=[];pts=r['positions'];ink=r['ink']
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        if a==b:continue
        X1=x+(a[0]-lo_x)*scale;Y1=y+(a[1]-lo_y)*scale
        X2=x+(b[0]-lo_x)*scale;Y2=y+(b[1]-lo_y)*scale
        stroke='#242424' if ink[i] else '#bababa'
        dash='' if ink[i] else ' stroke-dasharray=".9 1.2"'
        chunks.append(f'<path d="M {X1:.2f},{Y1:.2f} L {X2:.2f},{Y2:.2f}" fill="none" stroke="{stroke}" stroke-width=".55"{dash}/>')
    for p,color in [(pts[0],'#257454'),(pts[-1],'#b44848')]:
        X=x+(p[0]-lo_x)*scale;Y=y+(p[1]-lo_y)*scale
        chunks.append(f'<circle cx="{X:.2f}" cy="{Y:.2f}" r=".95" fill="{color}"/>')
    return ''.join(chunks)

def svg_page(pairs,page):
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
     '<rect width="210" height="297" fill="white"/>',
     '<text x="10" y="12" font-size="6" font-family="sans-serif">BLIND WORM PAIRS - original applied paths</text>',
     '<text x="10" y="18" font-size="3" font-family="sans-serif">Compare separately for MEANDERER / CURLER / BRANCHER / CHAOTIC ARTIST</text>']
    for j,p in enumerate(pairs):
        top=25+j*130; a=p['first'];b=p['second'];c=a['positions']+b['positions'];
        low_x=min(v[0] for v in c);low_y=min(v[1] for v in c)
        span_x=max(v[0] for v in c)-low_x+1;span_y=max(v[1] for v in c)-low_y+1
        scale=min(73/span_x,91/span_y)
        out.append(f'<text x="10" y="{top}" font-family="monospace" font-size="4">{html.escape(p["id"])}  (two original paths, matched source world and horizon)</text>')
        for k,r in enumerate([a,b]):
            col=10+k*99
            out.append(f'<rect x="{col}" y="{top+5}" width="91" height="111" rx="1" fill="white" stroke="#c7c7c7" stroke-width=".4"/>')
            out.append(f'<text x="{col+3}" y="{top+11}" font-size="4" font-family="sans-serif">{"A" if k==0 else "B"}</text>')
            out.append(path_svg(r,col+9,top+17,low_x,low_y,scale))
    out+=['<text x="10" y="290" font-size="2.8" font-family="sans-serif">Same scale and orientation within pair; different pairs not size-comparable. Gray = pen lifted.</text>',
       '<text x="10" y="294" font-size="2.8" font-family="sans-serif">Green start; red end. No identities, scores or preference labels on this sheet.</text>','</svg>']
    return '\n'.join(out)+'\n'

def verify_lfm_006(archive):
    """Original frozen LFM350 stock vs real step-96 LoRA, 9 paired 24-tick executions."""
    zhash=digest(Path(archive).read_bytes())
    with zipfile.ZipFile(archive) as z:
        adapter=digest(z.read('original-experiment/adapter-step-96/adapter_model.safetensors'))
        if adapter!='436159bd6508063221d31d46f057b5fba7ee5b6f05b27edf003c54e76863898c':
            raise ValueError('actual LFM350 adapter checksum mismatch')
        pairs=[];records=[]
        for family in ('architecture','greebles','pipework'):
            for seed in (9503,9521,9539):
                members=[]
                for arm in ('stock','trained'):
                    entry=f'original-experiment/{arm}-{seed}-{family}.json';raw=z.read(entry);doc=json.loads(raw)
                    traj=doc['record']['trajectory'];events=doc['record']['events'];dec=doc['decisions']
                    if not (len(events)==len(dec)==24 and len(traj)==25):raise ValueError('LFM 24-decision receipt mismatch')
                    for i,(d,e) in enumerate(zip(dec,events)):
                        obs=e['observation']
                        if obs['tick']!=i or obs['x']!=traj[i][0] or obs['y']!=traj[i][1] or e['position_after']!=traj[i+1]:raise ValueError('LFM position receipt inconsistency')
                        if d['applied']!=e['applied'] or d['applied'] not in obs['legal'] or len(d['logits'])!=6:raise ValueError('LFM illegal action or incomplete model logits')
                    r={'name':f'LFM350 {arm} {seed} {family}','group':'lfm350','world':f'{family} / seed {seed}',
                       'decisions':24,'source_artifact_id':14199280586,'source_entry':entry,
                       'source_json_sha256':digest(raw),'source_zip_sha256':zhash,'positions_sha256':digest(jsonbytes(traj)),
                       'positions':traj,'ink':[True]*24,'original_model_or_native_actions':dict(collections.Counter(d['applied'] for d in dec)),
                       'source_level':'ORIGINAL_PRODUCER_BYTES_VERIFIED','host':'EXTERNAL_SYNTHETIC_W0','menu_violations':0,'decisions':24}
                    members.append((r,dec));records.append(r)
                (a,ad),(b,bd)=members
                different_logits=sum(any(abs(x-y)>1e-6 for x,y in zip(da['logits'],db['logits'])) for da,db in zip(ad,bd))
                pairs.append({'family':family,'seed':seed,'stock':a,'trained':b,
                    'identical_applied_trajectory':a['positions']==b['positions'],
                    'different_logit_ticks':different_logits,
                    'identical_applied_actions':all(x['applied']==y['applied'] for x,y in zip(ad,bd))})
    if len(pairs)!=9:raise ValueError('expected 9 real paired LFM worlds')
    return {'source_zip_sha256':zhash,'adapter_step96_sha256':adapter,
        'paired_worlds':9,'same_path':sum(x['identical_applied_trajectory'] for x in pairs),
        'same_actions':sum(x['identical_applied_actions'] for x in pairs),
        'model_forward_decisions':9*24*2,'different_logit_ticks':sum(x['different_logit_ticks'] for x in pairs),
        'pairs':[{'seed':p['seed'],'family':p['family'],'same_path':p['identical_applied_trajectory'],
                  'different_logit_ticks':p['different_logit_ticks'],
                  'stock_json_sha256':p['stock']['source_json_sha256'],
                  'trained_json_sha256':p['trained']['source_json_sha256']} for p in pairs]},pairs

def execute(manifest,archive_dir,outdir,lfm_archive=None):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rows=verify(manifest,archive_dir);pairs=paired(rows)
    lfm_summary=None
    if lfm_archive:
        lfm_summary,extra=verify_lfm_006(lfm_archive)
        # Three explicitly disclosed identical-result attention checks; NOT 9 independent style pairs.
        for family in ('architecture','greebles','pipework'):
            p=next(p for p in extra if p['family']==family and p['seed']==9503)
            a,b=p['stock'],p['trained']
            pairs.append({'id':'P18-'+digest(('anonymous-original/'+a['source_json_sha256']+'/'+b['source_json_sha256']).encode())[:12],
                         'world_block':['lfm350',a['world'],24],'first':a,'second':b,
                         'catch_identical_original_path':True})
        for i,p in enumerate(pairs):p['index']=i+1
    if len(pairs)!=(13 if lfm_archive else 10):raise ValueError('paired sources unexpectedly changed')
    output={'schema':'driv3r.preference.p18.original_producer_reverify/1',
        'provenance':'All original JSON producer bytes verified against P17 manifest and source archive; not native ABI equivalence',
        'curation_manifest_sha256':SOURCE_MANIFEST_SHA,'original_producers_verified':len(rows),
        'model_applied_W0':sum(r['group']!='native' for r in rows),
        'native_references':sum(r['group']=='native' for r in rows),
        'model_ratings':0,'human_ratings':0,'new_SLM_forwards':0,'optimizer_updates':0,
        'original_source_records':[{k:v for k,v in r.items() if k not in ('positions','ink')} for r in rows],
        'matched_pairs':len(pairs),'historical_distinct_trajectory_pairs':10,
        'lfm350_original_applied_identical_catch_trials':3 if lfm_archive else 0,
        'independent_world_blocks':len({tuple(p['world_block']) for p in pairs}),
        'lfm350_source_verification':lfm_summary,
        'cross_model_heldout_ranker':'NOT_TRAINED_NO_HUMAN_LABELS_OR_INDEPENDENT_MATCHED_WORLD_FAMILIES',
        'absent_lanes':['Tiny pretrained original model rollouts','Population pretrained original model rollouts','GP350-018 sampled full LFM original model rollouts']}
    (out/'P18_SOURCE_AUDIT.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    mapping=[]
    with (out/'P18_BLIND_BALLOT.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['anonymous_pair_id','niche','choice_A_B_tie_skip','owner_confidence_1_to_5','owner_notes'])
        for p in pairs:
            for niche in NICHES:w.writerow([p['id'],niche,'','',''])
    for p in pairs:
        mapping.append({'pair_id':p['id'],'world_block':p['world_block'],
            'A':{'name':p['first']['name'],'source_json_sha256':p['first']['source_json_sha256']},
            'B':{'name':p['second']['name'],'source_json_sha256':p['second']['source_json_sha256']},
            'owner_vote':None,'model_preference':None,'weak_label':None,
            'identical_result_catch_trial':bool(p.get('catch_identical_original_path'))})
    (out/'P18_REVEAL_AFTER_RATING.json').write_text(json.dumps(mapping,indent=2,sort_keys=True)+'\n')
    for i in range(0,len(pairs),2):
        (out/f'P18-blind-A4-{i//2+1:02}.svg').write_text(svg_page(pairs[i:i+2],i//2+1))
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--archives',required=True);ap.add_argument('--output',required=True);ap.add_argument('--lfm-archive')
    a=ap.parse_args();r=execute(a.manifest,a.archives,a.output,a.lfm_archive);print('P18',r['original_producers_verified'],'verified original producer JSONs;',r['matched_pairs'],'matched blind pairs;',r['human_ratings'],'ratings')
