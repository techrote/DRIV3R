"""Re-execute selected independent C55 validation worlds and verify every action hash.

Selections show positive and negative outcomes across both unseen and familiar families.
There is NO post-hoc filtering of quantitative aggregates (only illustrative gallery).
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
from safetensors.torch import load_file
import c55_observed_waypoint_distillation as c55
import c55_external_validation as ext
import c54_delayed_branch_option as c54

PICKS=(('wilson','aligned_wins'),('wilson','aligned_loses'),
       ('kruskal','aligned_wins'),('kruskal','route_wins'),
       ('prim_braid','aligned_loses'),('braid','aligned_wins'),
       ('prim','aligned_wins'),('tree','aligned_wins'),
       ('kruskal','shared_miss'))

def pick(rows):
    out=[];used=set()
    for fam,kind in PICKS:
        eligible=[]
        for row in rows:
            if row['family']!=fam or row['arm']!=['backtrack',1.]:continue
            k=(row['graph'],row['tick'])
            if k in used:continue
            z=row['outcomes'];a=z['C55-ALIGNED']['goal'];b=z['C34-RESET']['goal'];t=z['C55-ROUTE']['goal']
            good={'aligned_wins':a>b,'aligned_loses':a<b,
                  'route_wins':t>a,'shared_miss':not (a or b or t)}[kind]
            if good:eligible.append(row)
        if eligible:
            item=sorted(eligible,key=lambda x:(x['graph'],x['tick']))[0]
            out.append(item);used.add((item['graph'],item['tick']))
    return out

def run(prior,trainout,externalout,svgout):
    prior=Path(prior);trainout=Path(trainout);externalout=Path(externalout)
    s=json.loads((trainout/'C55-summary.json').read_text());ledger=json.loads((externalout/'C55-external-summary.json').read_text())
    chosen=pick(json.loads((externalout/'C55-external-online.json').read_text()))
    heads={}
    for name in ('C55-ALIGNED','C55-SHUFFLED'):
        m=c55.Head();m.load_state_dict(load_file(str(trainout/(name+'.safetensors'))));m.eval();heads[name]=m
    base,option,_=c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    # We can reconstruct worlds by their original saved seeds and family; all are
    # independently validated to have exactly matching graph SHA and prefix hash.
    result=[]
    for row in chosen:
        family,case,seed=row['family'],row['case'],row['seed']
        maker=c54.c52.c50.c47.c42.world if family in ('wilson','kruskal') else c54.c52.c50.c47.c41.world
        # c52.worlds retains both personality and seed; locate the authoritative
        # per-world receipt rather than inferring personality from seed parity.
        receipt=next(r for r in ledger['receipts']['online'] if r['graph']==row['graph'])
        personality=receipt['personality']
        w,info=maker(seed,family,case,personality)
        assert info['graph_sha']==row['graph'],(family,case,seed,info['graph_sha'],row['graph'])
        got=c54.prefix(base,w,seed,row['tick']);assert got is not None
        snap,rng,log,prefix=got
        assert prefix['prefix_sha']==row['prefix_sha']
        target=c55.waypoint(log,snap.pos);assert list(target)==row['target']
        for p in ('C34-RESET','C55-ROUTE','C55-ALIGNED'):
            outcome=c55.rollout(base,option,snap,rng,log,row['tick'],target,('backtrack',1.),p,heads,True)
            ref=row['outcomes'][p]
            assert (outcome['action_sha'],outcome['ink_sha'],outcome['goal'])==(ref['action_sha'],ref['ink_sha'],ref['goal'])
            result.append({'graph':row['graph'],'family':family,'case':case,'seed':seed,'tick':row['tick'],
                 'initial':prefix['initial'],'positions':prefix['positions'],
                 'target':target,'protected':sorted(snap.protected),'policy':p,'outcome':outcome})
    from c55_gallery import render
    Path(svgout).write_text(render(result))
    (Path(svgout).parent/'C55-curated-traces.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
    print('re-executed',len(result),'curated trajectories from',len(chosen),'independent worlds')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('prior');a.add_argument('trainout');a.add_argument('externalout');a.add_argument('svgout')
    z=a.parse_args();run(z.prior,z.trainout,z.externalout,z.svgout)
