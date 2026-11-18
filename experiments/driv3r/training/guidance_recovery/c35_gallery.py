"""Replay-verified C35 path gallery, synthetic host only; no game mutation."""
from __future__ import annotations
import argparse,hashlib,html,json
from pathlib import Path
from experiment import sha
from signed_feedback import HostAdvice
from c35_cross_generator_calibration import make_world


def replay(record):
    w,info=make_world(record['seed'],record['family'],record['case'],record['personality'])
    assert info['graph_sha']==record['graph_sha']
    assert sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])==record['prefix_sha']
    material=sha(sorted(w.protected));cue=HostAdvice(w)
    if record['directive']!='none':getattr(cue,'issue_'+record['directive'])(w,strength=record['strength'],ttl=9)
    seen={e['t']:e for e in record['path']};actions=[];coordinates=[];switched=False
    for t in range(max(seen)+1 if seen else 0):
        if record['switch'] and t==50:
            w.goal=info['second_goal'];w.goal_change=True;w.goals_seen+=1;w.stuck=2;switched=True
            if record['directive']!='none':getattr(cue,'issue_'+record['directive'])(w,strength=record['strength'],ttl=9)
        if t not in seen:
            assert w.distance()==0 and record['switch'] and t<50
            continue
        step=seen[t];pos=tuple(step['pos'])
        menu=w.candidates(record['arm'])
        candidates=[c for c in menu if c.legal and c.action[0]==step['action'] and c.target==pos]
        assert len(candidates)==1,(record['model'],t,step,candidates)
        action=candidates[0]
        assert w.apply(action) and w.invalid_attempts==0
        assert tuple(w.pos)==pos and w.distance()==step['distance']
        assert tuple(w.goal)==tuple(step['goal'])
        actions.append((action.action,action.target));coordinates.append(pos);cue.advance(w)
    assert sha(actions)==record['actions_sha']
    assert sha(sorted(w.inked))==record['ink_sha']
    assert sha(sorted(w.protected))==material
    assert w.invalid_attempts==0
    return w,info,coordinates


def make_gallery(source,destination):
    rows=json.loads(Path(source).read_text());panels=[];receipts=[]
    for i,row in enumerate(rows):
        w,info,positions=replay(row)
        allowed={(x,y) for x in range(15) for y in range(15)}-set(w.protected)
        scale=14
        blocks=''.join(f'<rect x="{x*scale}" y="{y*scale}" width="{scale}" height="{scale}" fill="#222e3b"/>' for x,y in sorted(allowed))
        for x,y in sorted(w.protected):
            color='#786665' if (x+y)%3==0 else '#53565c'
            blocks+=f'<rect x="{x*scale}" y="{y*scale}" width="{scale}" height="{scale}" fill="{color}"/>'
        points=[tuple(positions[0])] if positions else []
        coordinates=[(float(x*scale+scale/2),float(y*scale+scale/2)) for x,y in positions]
        poly=' '.join(f'{x:.1f},{y:.1f}' for x,y in coordinates)
        goal=tuple(row['path'][0]['goal']) if row['path'] else (0,0)
        polygons=f'<polyline points="{poly}" fill="none" stroke="#78e2d4" stroke-opacity="0.8" stroke-width="2"/>'
        marker=f'<circle cx="{goal[0]*scale+scale/2}" cy="{goal[1]*scale+scale/2}" r="4" fill="#f9be65"/>'
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 210 210" role="img">{blocks}{polygons}{marker}</svg>'
        label=f"{row['family']} · {row['case']} · {'switch' if row['switch'] else 'single'} · {row['model']}"
        panels.append(f'<figure>{svg}<figcaption><strong>{html.escape(label)}</strong><span>goal events {row["total_success"]} · moves {row["steps"]} · SHA {row["actions_sha"][:11]}</span></figcaption></figure>')
        receipts.append({'model':row['model'],'actions_sha':row['actions_sha'],'ink_sha':row['ink_sha'],
            'graph_sha':row['graph_sha'],'replay_verified':True,'path_length':len(row['path'])})
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>C35 replayed trajectories</title><style>body{font-family:system-ui;background:#10171f;color:#e5e9ea;max-width:1400px;margin:auto;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(225px,1fr));gap:18px}figure{margin:0;border:1px solid #34414c;border-radius:9px;padding:10px;background:#1c2630}svg{width:100%;height:auto;border-radius:4px}strong,span{display:block;font-size:12px;line-height:1.6}span{color:#abb5ba}p{color:#abb5ba}</style><h1>C35 host-executed trajectory evidence</h1><p>Native game is not involved. Each path replays against the original synthetic material mask and verifies action, ink and graph hashes. Cyan = executed route, amber = first target, dark/grey = traversable/protected cells.</p><main>'''+''.join(panels)+'</main></html>'
    destination=Path(destination);destination.write_text(content)
    receipt={'schema':'driv3r-c35-gallery/1','panels':len(receipts),
        'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'html_sha256':hashlib.sha256(content.encode()).hexdigest(),'entries':receipts}
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('html');a=p.parse_args()
    print(json.dumps(make_gallery(a.source,a.html),sort_keys=True))
