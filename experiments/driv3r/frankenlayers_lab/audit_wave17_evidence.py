"""Offline independent verification of the historical Wave-17 original Actions ZIP.

No model, game, GitHub or external dependencies. Audits checksums, original
host actions, trajectories, novelty/revisits and paired observation divergence.
Not a scientific reproduction or native game evaluation.
"""
import collections
import hashlib
import json
from pathlib import PurePosixPath
import statistics
import sys
import zipfile

VARIANTS = ('east_open','east_blocked','north_blocked','south_blocked','far_open','far_blocked')
PAIRS = (('east_open','east_blocked'),('north_blocked','south_blocked'),('far_open','far_blocked'))
ARMS = ('q8_stock','q4_stock','q4_mutant_raw','q4_mutant_strong','q4_mutant_guarded')
ACTIONS = frozenset(('STRAIGHT','LEFT','RIGHT','BRANCH','RETRACE','STOP'))


def difference(a,b,key):
    return next((i for i in range(min(len(a),len(b))) if key(a[i])!=key(b[i])),None)


def verify_zip(z):
    names=z.namelist()
    if len(names)!=len(set(names)) or 'SHA256SUMS.txt' not in names:
        raise ValueError('Missing manifest or duplicate ZIP paths')
    listed=set()
    for row in z.read('SHA256SUMS.txt').decode().splitlines():
        if not row.strip():continue
        expected,name=row.split(None,1)
        name=name.strip().removeprefix('*').removeprefix('./')
        if (len(expected)!=64 or name in listed or name not in names or
            PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts):
            raise ValueError('Malformed or unsafe manifest entry: '+name)
        listed.add(name)
        if hashlib.sha256(z.read(name)).hexdigest()!=expected.lower():
            raise ValueError('Payload SHA-256 mismatch: '+name)
    if set(names)-listed!={'SHA256SUMS.txt'}:
        raise ValueError('Unlisted evidence payload')
    return len(listed)


def extract(z,arm,variant):
    prefix='inference/eval-'+arm.replace('q4_mutant_','q4-mutant-').replace('_','-')+'/'+variant+'/'
    calls=[n for n in z.namelist() if n.startswith(prefix) and n.endswith('-choices.json')]
    hosts=[n for n in z.namelist() if n.startswith(prefix) and n.endswith('.json') and
           not n.endswith('-choices.json') and '/process/' not in n]
    if len(calls)!=1 or len(hosts)!=1:raise ValueError('Ambiguous episode '+prefix)
    return json.loads(z.read(calls[0])),json.loads(z.read(hosts[0]))


def rescore_episode(calls,host):
    positions=[tuple(p) for p in host['trajectory']]
    events=host['events']
    if not calls or len(calls)>36 or len(positions)!=len(calls)+1 or len(events)!=len(calls):
        raise ValueError('Inconsistent episode length')
    seen={positions[0]};revisits=0
    for i,(c,e) in enumerate(zip(calls,events)):
        obs=c['observation'];legal=obs['legal'];choice=c['choice']
        if (c['status']!='ok' or choice not in legal or len(set(legal))!=len(legal) or
            not set(legal)<=ACTIONS or obs['tick']!=i or
            choice!=e['applied'] or obs!=e['observation'] or
            tuple(e['position_after'])!=positions[i+1]):
            raise ValueError('Action, host observation or trajectory disagrees')
        if choice not in ('STOP','RETRACE') and positions[i+1] in seen:
            revisits+=1
        seen.add(positions[i+1])
        if 'adapter' in c:
            a=c['adapter']
            if a['base']!=c['base_model_choice'] or a['adapted']!=choice:
                raise ValueError('Adapter provenance mismatch')
            if 'shielded' in a:
                wall=any('#' in row for row in obs['local_5x5'])
                shield=wall and a['base'] not in ('STOP','RETRACE') and a['pre_shield_adapted']!=a['base']
                if shield!=a['shielded'] or a['wall_visible']!=wall or (shield and choice!=a['base']):
                    raise ValueError('Observed wall guard does not agree with intervention')
    if (len(seen)!=host['metrics']['unique_trail_cells'] or len(seen)-1!=host['metrics']['new_steps'] or
        revisits!=host['metrics']['total_revisit_contacts'] or len(calls)!=host['steps_executed']):
        raise ValueError('Host novelty/revisit summary mismatch')
    return len(seen)-1,revisits


def audit(path):
    archive_sha=hashlib.sha256(open(path,'rb').read()).hexdigest()
    with zipfile.ZipFile(path) as z:
        count=verify_zip(z)
        report=json.loads(z.read('inference/comparison.json'))
        if report['schema']!='frankenlayers-wave17-causal-shield-ablation/1' or report['native_game']:
            raise ValueError('Not a research-only Wave-17 ZIP')
        results={}
        for arm in ARMS:
            original=report['arms'][arm]
            episodes={v:extract(z,arm,v) for v in VARIANTS}
            scores={v:rescore_episode(*episodes[v]) for v in VARIANTS}
            decisions=sum(len(calls) for calls,_ in episodes.values())
            counts=collections.Counter(c['choice'] for calls,_ in episodes.values() for c in calls)
            novel=sum(x[0] for x in scores.values())
            revisits=sum(x[1] for x in scores.values())
            median=statistics.median(statistics.median(c['latency_seconds'] for c in episodes[v][0]) for v in VARIANTS)
            summary=original['summary']
            if (decisions!=summary['decisions'] or novel!=summary['new_cells'] or
                revisits!=summary['revisits'] or dict(counts)!={k:v for k,v in summary['choice_counts'].items() if v} or
                abs(median-summary['median_of_variant_p50_seconds'])>1e-9):
                raise ValueError('Aggregate mismatch: '+arm)
            observed=responded=0
            for x,y in PAIRS:
                a,b=episodes[x][0],episodes[y][0]
                o=difference(a,b,lambda c:c['observation'])
                d=difference(a,b,lambda c:c['choice'])
                if d is not None and (o is None or d<o):
                    raise ValueError('Action change precedes observation change')
                observed+=o is not None
                responded+=o is not None and d is not None
            premature=sum(bool(calls) and calls[-1]['choice']=='STOP' and len(calls)<36
                          for calls,_ in episodes.values())
            results[arm]={'decisions':decisions,'new_cells':novel,'revisits':revisits,
                          'premature_stops':premature,'responded_pairs':responded,
                          'exposed_pairs':observed,'median_decision_seconds':median}
        return {'archive_sha256':archive_sha,'verified_payloads':count,
                'source_sha':report['source_sha'],'adapter_weights_sha256':report['adapter_weights_sha256'],
                'arms':results,'strong_minus_guarded_new_cells':
                    results['q4_mutant_strong']['new_cells']-results['q4_mutant_guarded']['new_cells']}


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: python -m experiments.driv3r.frankenlayers_lab.audit_wave17_evidence original.zip')
    print(json.dumps(audit(sys.argv[1]),sort_keys=True,indent=2))
