"""Actual executed receipt trajectories only; no interpolated or inferred actions."""
from pathlib import Path
import argparse
import gzip
import json
from html import escape


def draw(receipts: Path, dest: Path):
    with gzip.open(receipts,'rt',encoding='utf-8') as f:
        traces = [json.loads(x) for x in f]
    arms=('blind','intersect','lazy','neural')
    modes=('early_reopen','late_reopen')
    items=[]
    for mode in modes:
        for arm in arms:
            tr=next(t for t in traces if t['row']['spec']['family']=='pickets'
                    and t['row']['goal_idx']==0 and t['row']['mode']==mode
                    and t['row']['arm']==arm and (arm!='neural' or t['row']['seed']==711))
            items.append((mode,arm,tr))
    s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 530">',
       '<rect width="1000" height="530" fill="#111620"/>',
       '<text x="20" y="28" fill="#e7edf5" font-size="20" font-family="sans-serif">DRIV3R / GATE-OSCILLATION-001 · executed synthetic trajectories</text>']
    for idx,(mode,arm,tr) in enumerate(items):
        col=idx%4;row=idx//4
        bx=16+col*247;by=44+row*238
        s.append(f'<g transform="translate({bx},{by})">')
        s.append('<rect width="236" height="222" rx="8" fill="#1c2430" stroke="#3f5062"/>')
        outcome='ARRIVED' if tr['row']['arrived'] else 'FAILED'
        s.append(f'<text x="10" y="19" fill="#fff" font-size="12" font-family="sans-serif">{escape(mode)} / {arm} · {outcome}</text>')
        assert tr['entries'] and 'cursor_before' in tr['entries'][0]['receipt']
        points=[tr['entries'][0]['receipt']['cursor_before']]+[e['cursor'] for e in tr['entries']]
        cx,cy=tr['gate']
        xs=[p[0] for p in points]+[cx];ys=[p[1] for p in points]+[cy]
        xmin=min(xs)-2;xmax=max(xs)+2;ymin=min(ys)-2;ymax=max(ys)+2
        scale=min(200/(xmax-xmin),167/(ymax-ymin))
        x0=18-xmin*scale;y0=32-ymin*scale
        def coord(p):return f'{x0+p[0]*scale:.2f},{y0+p[1]*scale:.2f}'
        s.append(f'<polyline points="{" ".join(map(coord,points))}" fill="none" stroke="#5dc6f5" stroke-width="1.5" stroke-linejoin="round"/>')
        gx=x0+cx*scale;gy=y0+cy*scale
        s.append(f'<rect x="{gx-2.5:.2f}" y="{gy-2.5:.2f}" width="5" height="5" fill="#fc7968"/>')
        a,b=points[-1];s.append(f'<circle cx="{x0+a*scale:.2f}" cy="{y0+b*scale:.2f}" r="3" fill="#c7f77a"/>')
        s.append(f'<text x="10" y="211" fill="#a1b2c7" font-size="10" font-family="sans-serif">{tr["row"]["decisions"]} decisions · {tr["row"]["planner_calls"]} searches</text>')
        s.append('</g>')
    s.append('<text x="19" y="524" fill="#9cabbf" font-size="10" font-family="sans-serif">Blue: executed cursor path · red: material gate · lime: final cursor · synthetic DEVELOPMENT only</text>')
    s.append('</svg>')
    dest.write_text('\n'.join(s)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipts',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();draw(a.receipts,a.out)
