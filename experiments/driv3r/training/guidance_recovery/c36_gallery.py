"""C36 source-replayed host-executed trace gallery (no screenshots)."""
import argparse, copy, hashlib, html, json
from pathlib import Path
from experiment import sha
from c34_independent_maze import make_world

def verify(row):
    w,info=make_world(row['seed'],row['family'],row['case'],row['personality'])
    initial=sha(sorted(w.protected));coords=[tuple(w.pos)];actual=[];last_t=-1
    for step in row['trace']:
        t=step['t'];assert t>last_t;last_t=t
        if row['switch'] and t>=50 and not getattr(w,'_c36_switched',False):
            w.goal=info['second_goal'];w.goal_change=True;w.goals_seen+=1;w.stuck=2
            w._c36_switched=True
        target=tuple(step['pos']);menu=w.candidates(row['arm'])
        matches=[c for c in menu if c.legal and c.action[0]==step['kind'] and c.target==target]
        assert len(matches)==1,(row['seed'],row['model'],t,step)
        assert w.apply(matches[0]) and w.invalid_attempts==0
        actual.append((matches[0].action,matches[0].target));coords.append(target)
    assert sha(actual)==row['action_sha'] and sha(sorted(w.inked))==row['ink_sha']
    assert sha(sorted(w.protected))==initial
    return w,coords,info

def render(rows,output):
    selected=[];seen=set()
    for row in rows:
        if row['trace'] is None or row['directive']!='backtrack' or row['strength']!=1.:continue
        k=(row['family'],row['case'],row['switch'],row['seed'])
        if k not in seen:
            if len(seen)>=4:continue
            seen.add(k)
        if row['model'] in ('C18-A0','C34-RESET','C36-LEARNED'):selected.append(row)
    assert len(selected)==12
    parts=['<html><head><meta charset="utf-8"><title>C36 executed trajectory gallery</title><style>body{font:14px system-ui;background:#101923;color:#e2e8f0;margin:24px} .grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:18px}figure{background:#1c2937;padding:12px;margin:0;border:1px solid #415263}svg{width:100%;height:auto}figcaption{margin-top:8px;line-height:1.5}h1{font-size:25px}</style></head><body><h1>C36: executed and replay-verified recovery paths</h1><p>White: legal cells. Dark: protected material. Cyan: initial executed prefix. Orange: evaluated path. Magenta: current goal. Synthetic host only.</p><div class="grid">']
    for row in selected:
        w,coords,info=verify(row)
        cells={(x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected}
        rects=''.join(f'<rect x="{x*10}" y="{y*10}" width="10" height="10" fill="#e0e6e2"/>' for x,y in cells)
        prefix=[(1,1)]+[e[2] for e in w.events[:24]] if False else None
        # Rebuild prefix independently from the generated host.
        wi,_=make_world(row['seed'],row['family'],row['case'],row['personality'])
        prefix=[wi.events[0][1]]+[e[2] for e in wi.events]
        pp=' '.join(f'{10*x+5},{10*y+5}' for x,y in prefix)
        path=' '.join(f'{10*x+5},{10*y+5}' for x,y in coords)
        target=tuple(w.goal)
        parts.append('<figure><svg viewBox="0 0 150 150" role="img" aria-label="Executed trajectory">'+rects+f'<polyline points="{pp}" stroke="#27b9ca" stroke-width="1.8" fill="none"/><polyline points="{path}" stroke="#f68f40" stroke-width="2" fill="none"/><circle cx="{target[0]*10+5}" cy="{target[1]*10+5}" r="4" fill="#c342d5"/></svg><figcaption>'+html.escape(f"{row['family']} · {row['case']} · {'switch' if row['switch'] else 'single'} · {row['model']} · goals {row['goals']} · {row['steps']} legal actions")+'</figcaption></figure>')
    parts.append('</div></body></html>');Path(output).write_text('\n'.join(parts))
    return {'panels':len(selected),'verified_action_and_ink_hashes':len(selected),'html_sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest(),'selected_source_rows_sha256':sha([{k:v for k,v in r.items() if k!='trace'} for r in selected])}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('rows');p.add_argument('out');a=p.parse_args()
 print(json.dumps(render(json.loads(Path(a.rows).read_text()),a.out),indent=2))
