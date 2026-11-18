"""BUDGET-ORDER-001: pinned, read-only DRIV3R DEVELOPMENT action-order audit.

Input: original SHA-256 verified BUDGET-CALIBRATION-001 evidence ZIP.
No new optimization, native execution or structural HELDOUT access.
Budget groups are NOT causal interventions: detour ink/prehistory differs.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse, json, math, resource, statistics, time, zipfile

VERSION='budget-order-audit/1'
EXPECTED_ARCHIVE='06573cd6d9502f7e132b3b3effd16a9c74dff2654f5668c692c9c9407906c2f9'
EXPECTED_EPISODES='e38b26ae35394f051fdc08be547821012f89d62ce9ddc0bebc80d18588a71f33'
ARMS=('calibrated','fixed3','budget_blind')
BUDGETS=(1,3,5)

def sha(data):return sha256(data).hexdigest()

def load_verified(archive):
    if sha(Path(archive).read_bytes())!=EXPECTED_ARCHIVE:
        raise ValueError('Original archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        sums=json.loads(z.read('evidence/SHA256SUMS.json'))
        for f,h in sums.items():
            if sha(z.read('evidence/'+f))!=h:raise ValueError('Original member hash mismatch: '+f)
        raw=z.read('evidence/episodes.json')
        if sha(raw)!=EXPECTED_EPISODES:raise ValueError('Original episode hash mismatch')
        return (json.loads(raw),json.loads(z.read('evidence/samples.json')),
                json.loads(z.read('evidence/report.json')))

def analyze(rows,samples):
    triads=defaultdict(dict)
    for r in rows:
        k=(r['case'],r['rotation'],r['kind'],r['seed'],r['arm'])
        if r['budget'] in triads[k]:raise AssertionError('Duplicated budget')
        triads[k][r['budget']]=r
    if len(triads)!=960 or any(set(d)!=set(BUDGETS) for d in triads.values()):
        raise AssertionError('Incomplete source census')
    if any(len({r['depth'] for r in d.values()})!=1 for d in triads.values()):
        raise AssertionError('Goal depths changed')
    if any(len({r['state_before'] for r in d.values()})!=3 for d in triads.values()):
        raise AssertionError('Budget prehistories unexpectedly identical')
    if any(any(d[b]['label']=='BACKTRACK' and d[b+2]['label']=='STOP' for b in (1,3)) for d in triads.values()):
        raise AssertionError('Oracle labels were not monotonic')
    census={}; witnesses=[]
    for arm in ARMS:
        gg=[(k,d) for k,d in triads.items() if k[4]==arm]
        inv=[(k,d) for k,d in gg if any(d[b]['first']=='BACKTRACK' and d[b+2]['first']=='STOP' for b in (1,3))]
        census[arm]={
            'paired_budget_triads':len(gg),'order_inversions':len(inv),
            'inversions_1_to_3':sum(d[1]['first']=='BACKTRACK' and d[3]['first']=='STOP' for k,d in gg),
            'inversions_3_to_5':sum(d[3]['first']=='BACKTRACK' and d[5]['first']=='STOP' for k,d in gg),
            'backtrack_by_budget':{str(b):sum(d[b]['first']=='BACKTRACK' for k,d in gg) for b in BUDGETS},
            'correct_by_budget':{str(b):sum(d[b]['correct'] for k,d in gg) for b in BUDGETS},
            'futile_backtrack_by_budget':{str(b):sum(d[b]['first']=='BACKTRACK' and d[b]['label']=='STOP' for k,d in gg) for b in BUDGETS},
            'missed_reachable_by_budget':{str(b):sum(d[b]['first']=='STOP' and d[b]['label']=='BACKTRACK' for k,d in gg) for b in BUDGETS}}
        witnesses += [{'arm':arm,'case':k[0],'rotation':k[1],'kind':k[2],'seed':k[3],
                       'choices':{str(b):d[b]['first'] for b in BUDGETS},
                       'oracle':{str(b):d[b]['label'] for b in BUDGETS},
                       'before_sha256':{str(b):d[b]['state_before'] for b in BUDGETS},
                       'trace_sha256':{str(b):d[b]['trace_sha'] for b in BUDGETS}}
                      for k,d in inv]
    probs={}
    for arm in ARMS:
        ss=[s for s in samples if s['arm']==arm]
        if len(ss)!=114:raise AssertionError('Non-original sample count')
        errors=[];ll=[];bins=defaultdict(list)
        for s in ss:
            if not s['rows']:raise AssertionError('Missing sampled action receipt')
            q=s['rows'][0]; menu=q['menu']; p=q['probs']
            if not (len(menu)==len(p) and q['action']==s['first'] and
                    abs(sum(p)-1)<1e-6 and q['action'] in menu and s['label'] in menu):
                raise AssertionError('Invalid legal-menu probability receipt')
            conf=max(p)
            if not s['correct']:errors.append(conf)
            ll.append(-math.log(max(float(p[menu.index(s['label'])]),1e-20)))
            bins[min(4,int(conf*5))].append((conf,int(s['correct'])))
        ece=sum(len(b)/len(ss)*abs(statistics.mean(v[0] for v in b)-statistics.mean(v[1] for v in b)) for b in bins.values())
        probs[arm]={'sampled':len(ss),'wrong':len(errors),
                    'mean_confidence_wrong':round(statistics.mean(errors),6),
                    'max_confidence_wrong':round(max(errors),6),
                    'mean_log_loss':round(statistics.mean(ll),6),
                    'top_class_ECE_5_bins':round(ece,6),
                    'nonrandom_sample':True}
    return {'version':VERSION,'source_episodes':len(rows),'sampled_receipts':len(samples),
            'census':census,'sampled_probability_diagnostic':probs,
            'inversion_witness_count':len(witnesses),'witnesses_first_24':witnesses[:24],
            'note':'Budget variation also changes prior detour, ink and state hash; not causal.'}

def main(archive,output):
    t=time.perf_counter()
    rows,samples,report=load_verified(archive);a=analyze(rows,samples)
    a.update(original_archive_sha256=EXPECTED_ARCHIVE,original_episodes_sha256=EXPECTED_EPISODES,
             checkpoint_sha256=report['model_sha256'],
             inherited_model_latency_ms=report['model_probability_ms'],
             inherited_peak_rss_kib=report['peak_rss_kib'],
             audit_wall_s=round(time.perf_counter()-t,4),
             audit_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             new_optimizer_updates=0,native_execution=False,heldout_opened=False)
    Path(output).write_text(json.dumps(a,sort_keys=True,indent=2)+'\n')
    return a

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('original_archive');p.add_argument('output_json')
    x=p.parse_args();res=main(x.original_archive,x.output_json)
    print(json.dumps({'census':res['census'],'probability_diagnostic':res['sampled_probability_diagnostic']},indent=2))
