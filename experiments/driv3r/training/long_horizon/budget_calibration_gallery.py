"""Original vector trajectory gallery from exact saved source-replayed receipts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from . import budget_calibration_lab as lab

COLORS={'calibrated':'#12a89d','fixed3':'#df714c','budget_blind':'#8b73bb'}

def esc(s):
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def build(evidence,out):
    rows=json.loads((evidence/'samples.json').read_text())
    # Six panels: 3 matched fork states, calibrated vs fixed-3.
    bykey=defaultdict(dict)
    for r in rows:
        if r['seed']==731:
            bykey[(r['case'],r['rotation'],r['budget'],r['kind'])][r['arm']]=r
    pairlist=[]
    for group in sorted(bykey.values(),key=lambda d:(d.get('calibrated',{}).get('budget',0),
                                                    d.get('calibrated',{}).get('case',''))):
        if 'calibrated' in group and 'fixed3' in group:
            a,b=group['calibrated'],group['fixed3']
            if a['first']!=b['first']:pairlist.append((a,b))
    chosen=[]
    for target in (1,5,3):
        options=[pair for pair in pairlist if pair[0]['budget']==target and
                 pair[0]['label']==pair[0]['first'] and pair[1]['label']!=pair[1]['first']]
        if not options:options=[pair for pair in pairlist if pair[0]['budget']==target]
        if options:chosen.append(options[0])
    if len(chosen)<3:raise ValueError('Too few informative matched groups')
    lookup={(c['case'],c['rot']*90,c['budget'],c['kind']):c for c in lab.cases('development')}
    W,H=1120,885
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
      '<rect width="100%" height="100%" fill="#0d1725"/>',
      '<text x="28" y="39" fill="#edf6f7" font-family="monospace" font-size="22">DRIV3R  /  BUDGET-CALIBRATION-001</text>',
      '<text x="28" y="62" fill="#9cb4bd" font-family="monospace" font-size="12">DEVELOPMENT synthetic World  |  source-replayed host receipts  |  same state, different trained policy</text>']
    for idx,(cal,fix) in enumerate(chosen):
        c=lookup[(cal['case'],cal['rotation'],cal['budget'],cal['kind'])]
        # Full prehistory already recorded: show ink as muted cells and actual
        # continuation as bright centerline, no implied animation/physics.
        for col,r in enumerate((cal,fix)):
            px=28+col*548;py=93+idx*253
            parts.append(f'<rect x="{px}" y="{py}" width="532" height="240" rx="12" fill="#152436" stroke="#385165"/>')
            name=r['arm'];good=r['first']==r['label']
            parts.append(f'<text x="{px+15}" y="{py+23}" fill="{COLORS[name]}" font-family="monospace" font-size="15" font-weight="bold">{esc(name.upper())}</text>')
            parts.append(f'<text x="{px+171}" y="{py+22}" fill="#e9f0ef" font-family="monospace" font-size="12">budget {r["budget"]}, depth {r["depth"]}, rot {r["rotation"]}°</text>')
            parts.append(f'<text x="{px+15}" y="{py+43}" fill="{"#a9eac6" if good else "#fca8a8"}" font-family="monospace" font-size="12">target: {r["label"]}  |  first: {r["first"]}  |  {"CORRECT" if good else "INCORRECT"}</text>')
            scene_x=px+40;scene_y=py+64;scale=16
            window=(44,44,53,53)
            # draw playfield 10x10 cropped around origin; x increase right, y down
            parts.append(f'<rect x="{scene_x}" y="{scene_y}" width="160" height="160" rx="5" fill="#0b1623" stroke="#42566d"/>')
            for x,y in sorted(c['state'].ink):
                if 44<=x<=53 and 44<=y<=53:
                    parts.append(f'<rect x="{scene_x+(x-44)*scale+2}" y="{scene_y+(y-44)*scale+2}" width="12" height="12" rx="3" fill="#3f607c"/>')
            for x,y in sorted(c['state'].protected):
                if 44<=x<=53 and 44<=y<=53:
                    parts.append(f'<rect x="{scene_x+(x-44)*scale+1}" y="{scene_y+(y-44)*scale+1}" width="14" height="14" fill="#af5960"/>')
            # each recorded receipt includes cursor_before and cursor_after
            points=[tuple(c['state'].cursor)]
            for e in r['rows']:
                rec=e['receipt'];points.append(tuple(rec['cursor_after']))
            line=' '.join(f'{scene_x+(x-44)*scale+8},{scene_y+(y-44)*scale+8}' for x,y in points)
            if len(points)>1:parts.append(f'<polyline points="{line}" fill="none" stroke="{COLORS[name]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
            sx,sy=points[0];gx,gy=c['goal']
            parts.append(f'<circle cx="{scene_x+(sx-44)*scale+8}" cy="{scene_y+(sy-45)*scale+8}" r="5" fill="#fff"/>')
            parts.append(f'<circle cx="{scene_x+(gx-44)*scale+8}" cy="{scene_y+(gy-45)*scale+8}" r="7" fill="none" stroke="#f4c46f" stroke-width="2"/>')
            parts.append(f'<text x="{px+271}" y="{py+83}" fill="#b1c9ce" font-family="monospace" font-size="12">Goal reached: {"YES" if r["hit_step"] is not None else "NO"}</text>')
            parts.append(f'<text x="{px+271}" y="{py+105}" fill="#b1c9ce" font-family="monospace" font-size="12">Actions: {r["decisions"]}</text>')
            p=[z for z in r['rows']]
            seq=' → '.join(z['action'].replace('BACKTRACK','BACK').replace('RELOCATE_START','ROOT') for z in p)
            parts.append(f'<text x="{px+271}" y="{py+129}" fill="#e1eced" font-family="monospace" font-size="11">{esc(seq[:39])}</text>')
            parts.append(f'<text x="{px+271}" y="{py+161}" fill="#819baa" font-family="monospace" font-size="10">SHA {r["state_before"][:15]}</text>')
    parts.append('<text x="28" y="868" fill="#89a4b1" font-family="monospace" font-size="11">Not native CyberSand. Cropped local ink; gold ring is goal, bright stroke is actual continuation. STOP is task-specific.</text>')
    parts.append('</svg>')
    out.write_text('\n'.join(parts)+'\n')
    return len(chosen)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();print('panels',build(args.evidence,args.output)*2)
