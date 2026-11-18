"""Source-constructed SVG: exact historic cursors + native-fixture action receipts.

No image generation, no interpolated movements, no reconstructed invented paths.
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path
from . import budget_calibration_lab as prior

W,H=1380,825
C = {'bg':'#0d1420','panel':'#172334','muted':'#91a6c1','grid':'#314157',
     'white':'#e8edf8','stroke':'#aec1d3','goal':'#ffcc66','old':'#4e6380',
     'active':'#6aa9ff','action':'#4bddab','stop':'#f79390'}

def construct(e,ref):
    k=(e['case'],e['rot'],e['kind'],e['original_budget'])
    case=ref[k]
    # Every cursor position is an actual host-legal receipt in the synthetic fixture.
    history=[(48,48)]+[tuple(x['cursor_after']) for x in case['state'].decisions]
    if len(history)!=len(case['state'].decisions)+1:raise AssertionError('bad history')
    after=[history[-1]]+[tuple(x['receipt']['cursor_after']) for x in e['rows']]
    if tuple(history[-1])!=case['state'].cursor:raise AssertionError('incorrect start')
    return history,after

def linepath(points,xy):
    return ' '.join(f'{"M" if i==0 else "L"}{xy(p)[0]:.2f},{xy(p)[1]:.2f}' for i,p in enumerate(points))

def image(samples):
    indices=[('8850699a','former_positive',1),('9717e8a5','former_positive',5),('8850699a','former_negative',1)]
    picked=[]
    for partial,kind,remain in indices:
        for arm in ('standard','recover4'):
            candidates=[e for e in samples if e['case'].startswith(partial) and e['kind']==kind and e['remaining']==remain and e['arm']==arm]
            if len(candidates)!=1:raise ValueError((partial,kind,remain,arm,len(candidates)))
            picked.append(candidates[0])
    ref={(c['case'],c['rot'],c['kind'],c['budget']):c for c in prior.cases('development')}
    chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        '<text x="28" y="40" fill="#eaf1ff" font-family="sans-serif" font-size="27" font-weight="bold">DRIV3R / recovery cost: actual receipt paths</text>',
        '<text x="28" y="65" fill="#91a6c1" font-family="sans-serif" font-size="15">Left: standard 1× supervision  ·  Right: 4× reachable-target emphasis  ·  Synthetic DEVELOPMENT, seed 941</text>']
    for i,e in enumerate(picked):
        row=i//2;col=i%2; x0=28+col*690;y0=90+row*240
        history,after=construct(e,ref)
        goals=[tuple(e['goal'])]
        coords=history+after+goals
        minx,maxx=min(x for x,y in coords),max(x for x,y in coords)
        miny,maxy=min(y for x,y in coords),max(y for x,y in coords)
        scale=min(31, (624-24)/max(1,maxx-minx+2), (156-24)/max(1,maxy-miny+2))
        def xy(p):return (x0+28+(p[0]-minx+1)*scale,y0+55+(p[1]-miny+1)*scale)
        status='ARRIVED' if e['hit'] else ('STOPPED' if e['stopped'] else 'NO ARRIVAL')
        cstat=C['action'] if e['hit'] else C['stop']
        chunks.append(f'<rect x="{x0}" y="{y0}" width="655" height="227" rx="12" fill="{C["panel"]}"/>')
        title=f'{e["case"][:8]} · {e["kind"].replace("former_", "")} · cap {e["remaining"]} · {e["arm"]}'
        chunks.append(f'<text x="{x0+16}" y="{y0+26}" fill="{C["white"]}" font-size="17" font-family="sans-serif">{html.escape(title)}</text>')
        chunks.append(f'<text x="{x0+16}" y="{y0+47}" fill="{cstat}" font-size="14" font-family="sans-serif">{status} · {e["steps"]} executed actions · goal depth {e["depth"]}</text>')
        chunks.append(f'<path d="{linepath(history,xy)}" fill="none" stroke="{C["old"]}" stroke-width="2.2" stroke-linejoin="round" stroke-dasharray="4 3"/>')
        state=ref[(e['case'],e['rot'],e['kind'],e['original_budget'])]['state']
        chunks.append(f'<path d="{linepath(state.path,xy)}" fill="none" stroke="{C["active"]}" stroke-width="3.5" stroke-linejoin="round"/>')
        if len(after)>1:
            chunks.append(f'<path d="{linepath(after,xy)}" fill="none" stroke="{C["action"]}" stroke-width="5" stroke-linecap="round"/>')
        goalx,goaly=xy(tuple(e['goal']))
        chunks.append(f'<circle cx="{goalx}" cy="{goaly}" r="7" stroke="{C["goal"]}" stroke-width="2.5" fill="none"/>')
        sx,sy=xy(history[-1]);chunks.append(f'<circle cx="{sx}" cy="{sy}" r="4" fill="{C["white"]}"/>')
        ex,ey=xy(after[-1]);chunks.append(f'<circle cx="{ex}" cy="{ey}" r="5" fill="{C["action"] if e["hit"] else C["stop"]}"/>')
        if e['rows']:
            choices=', '.join(f"{count}×{action}" for action,count in sorted(e['actions'].items()))
        else:choices='none'
        chunks.append(f'<text x="{x0+455}" y="{y0+195}" fill="{C["muted"]}" font-size="12" font-family="monospace">{html.escape(choices)}</text>')
    chunks.extend(['<text x="28" y="814" fill="#91a6c1" font-family="sans-serif" font-size="13">Dashed: all historical cursor receipts  /  blue: active history stack  /  green: executed continuation  /  gold circle: target</text>','</svg>'])
    return '\n'.join(chunks)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();samples=json.loads((a.evidence/'trajectory_samples.json').read_text());a.out.write_text(image(samples))
if __name__=='__main__':main()
