"""Produce actual-path SVG without fabrication or image-provider lookup."""
import argparse, html, json
from pathlib import Path
from .world import Spec, World, MATERIAL_Z

def draw(traces,output):
    rows=[]
    chosen_specs=[]
    for tr in traces:
        s=tr['summary']
        if s['family'] in ('pickets','ridges','islands') and not s['perturb'] and s['arm']=='goal_oracle' and s['spec'] not in chosen_specs:
            chosen_specs.append(s['spec'])
    selected=[]
    for ident in chosen_specs[:3]:
        selected.extend([next(t for t in traces if t['summary']['spec']==ident and not t['summary']['perturb'] and t['summary']['arm']==arm) for arm in ('goal_oracle','neural_1051')])
    W=450;H=475
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*W}" height="{3*H}" viewBox="0 0 {2*W} {3*H}">',
         '<rect width="100%" height="100%" fill="#121923"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e8eef7}.head{font-size:17px;font-weight:600}.small{font-size:12px;fill:#aab8c8}</style>']
    for i,tr in enumerate(selected):
        col=i%2;row=i//2; x0=col*W+30;y0=row*H+55;scale=3.8
        s=tr['summary'];sp=Spec(**tr['world']);world=World.create(sp)
        out.extend([f'<rect x="{col*W+10}" y="{row*H+10}" width="{W-20}" height="{H-18}" rx="9" fill="#1d2936"/>',
            f'<text class="head" x="{col*W+28}" y="{row*H+34}">{html.escape(s["family"])} | {html.escape(s["arm"])}</text>',
            f'<rect x="{x0}" y="{y0}" width="{96*scale}" height="{96*scale}" fill="#17212c"/>'])
        for (x,y),kind in world.materials.items():
            color='#52718b' if kind==MATERIAL_Z else '#6a7582'
            out.append(f'<rect x="{x0+x*scale:.1f}" y="{y0+y*scale:.1f}" width="{scale:.1f}" height="{scale:.1f}" fill="{color}"/>')
        for x,y in world.protected:
            out.append(f'<rect x="{x0+x*scale:.1f}" y="{y0+y*scale:.1f}" width="{scale:.1f}" height="{scale:.1f}" fill="#b36bb5"/>')
        pts=[[48,48]] + [e['cursor_after'] for e in tr['events'] if e['chosen']!='STOP']
        out.append('<polyline fill="none" stroke="#ffc14a" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round" points="'+ ' '.join(f'{x0+scale*(x+.5):.1f},{y0+scale*(y+.5):.1f}' for x,y in pts) +'"/>')
        for j,(x,y) in enumerate(tr['waypoints']):
            attained= j<s['goals_reached']
            out.append(f'<circle cx="{x0+scale*(x+.5):.1f}" cy="{y0+scale*(y+.5):.1f}" r="{3.5 if attained else 2.2}" fill="{ "#5fe1ae" if attained else "#fb6e88" }"/>')
        ax,ay=pts[-1];out.append(f'<circle cx="{x0+scale*(ax+.5):.1f}" cy="{y0+scale*(ay+.5):.1f}" r="5" fill="#5bd6f6"/>')
        out.append(f'<text class="small" x="{col*W+28}" y="{row*H+454}">Reached {s["goals_reached"]}/{s["goals_total"]} goals | {s["decisions"]} decisions | {s["logical_moves"]}/64 logical moves</text>')
    out.append('</svg>')
    Path(output).write_text('\n'.join(out)+'\n')
    return len(selected)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--traces',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();print(draw(json.loads(Path(args.traces).read_text()),args.out))
