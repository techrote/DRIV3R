"""Self-contained SVG gallery from recorded C34 executed traces (stdlib only)."""
from __future__ import annotations
import argparse,html,json
from pathlib import Path
from c34_independent_maze import make_world
from experiment import sha


def verified_trace(row,w,info):
    # Re-execute EVERY recorded action through the host's independent legality
    # check; never use rendered coordinates as a substitute for evidence.
    actions=[];switched=False;original_protected=sha(sorted(w.protected))
    for step in (row['path'] or []):
        if row['switch'] and int(step['t'])>=50 and not switched:
            w.goal=info['second_goal'];w.goal_change=True;w.goals_seen+=1;w.stuck=2
            switched=True
        target=tuple(step['pos']);kind=step['action']
        available=[c for c in w.candidates(0.) if c.legal and c.action[0]==kind and c.target==target]
        assert len(available)==1,(kind,target,w.pos)
        c=available[0];assert w.apply(c) and w.invalid_attempts==0
        actions.append((c.action,c.target))
    assert sha(actions)==row['actions_sha'],('actions hash mismatch',row['model'])
    assert sha(sorted(w.inked))==row['ink_sha'],('ink hash mismatch',row['model'])
    assert sha(sorted(w.protected))==original_protected
    return True


def panel(row):
    w,info=make_world(row['seed'],row['family'],row['case'],row['personality'])
    assert info['graph_sha']==row['graph_sha']
    assert row['prefix_sha']==sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    verify_world,verify_info=make_world(row['seed'],row['family'],row['case'],row['personality'])
    verified_trace(row,verify_world,verify_info)
    C=13;rect=[]
    for y in range(15):
        for x in range(15):
            p=(x,y);color='#354354' if p in w.protected else '#edf1f4'
            if p in w.inked and p not in w.protected:color='#c8d4da'
            rect.append(f'<rect x="{x*C}" y="{y*C}" width="{C-1}" height="{C-1}" fill="{color}"/>')
    def points(ps):return ' '.join(f'{(p[0]+.5)*C:.1f},{(p[1]+.5)*C:.1f}' for p in ps)
    pre=[*w.history,w.pos]
    rect.append(f'<polyline points="{points(pre)}" fill="none" stroke="#0a9278" stroke-width="2" opacity=".75"/>')
    trace=row['path'] or []
    executed=[w.pos,*[tuple(step['pos']) for step in trace]]
    if len(executed)>1:rect.append(f'<polyline points="{points(executed)}" fill="none" stroke="#e17231" stroke-width="1.8" opacity=".7"/>')
    for p,c in ((w.pos,'#09856d'),(info['goal'],'#dc3f3f'),(info['second_goal'],'#7d4ed0')):
        rect.append(f'<circle cx="{(p[0]+.5)*C:.1f}" cy="{(p[1]+.5)*C:.1f}" r="3.5" fill="{c}"/>')
    heading=f"{row['model']} · {row['family']} / {row['case']} · {'2 goals' if row['switch'] else '1 goal'}"
    return ('<div class="panel"><h3>'+html.escape(heading)+'</h3>'+
      f'<svg viewBox="0 0 {C*15} {C*15}" role="img" aria-label="executed maze trajectory">'+''.join(rect)+'</svg>'+
      f'<p>Completed {row["total_success"]}/{2 if row["switch"] else 1} goals · '
      f'{row["steps"]} legal actions · SHA {html.escape(row["actions_sha"][:12])}</p></div>')


def main(src,out):
    data=json.loads(Path(src).read_text())
    rows=data['rows'] if isinstance(data,dict) else data
    selected=[r for r in rows if r['path'] is not None and r['model'] in ('C18-A0','C34-RESET','C34-HEAD')]
    assert len(selected)==24,(len(selected),src)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>DRIV3R C34 synthetic trajectory gallery</title>
<style>body{font:14px system-ui,sans-serif;background:#e8edf0;color:#152936;margin:24px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}
.panel{background:white;border:1px solid #c7d0d7;border-radius:8px;padding:12px}
svg{display:block;width:100%;max-width:215px;margin:auto}h1{font-size:24px}
h3{font-size:14px}p{font-size:12px;color:#435565;line-height:1.5}</style>
<h1>DRIV3R C34 — independently generated mazes</h1>
<p>Synthetic host-executed trajectories, cohort A, signed BACKTRACK cue strength 1.0.
Grey = protected, white = traversable, green = previously executed prefix,
orange = subsequent selected path, red = first changed goal, purple = second changed goal.
Not native CyberSand acceptance. Reconstructed topology and prefix hashes verified per panel.</p>
<main>'''+''.join(panel(row) for row in selected)+'</main></html>'
    Path(out).write_text(page)
    return {'panels':len(selected),'action_sha256':[r['actions_sha'] for r in selected]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort_json');p.add_argument('out_html');a=p.parse_args()
    print(json.dumps(main(a.cohort_json,a.out_html)))
