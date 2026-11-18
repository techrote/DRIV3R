"""Original source-backed SVG: complete legal history and sampled continuations."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path
from . import budget_calibration_lab as cal
from . import cap_conditioning_lab as cap
from . import history_alias_lab as alias

def path_points(w):
    receipts=w.decisions
    if not receipts:return []
    return [tuple(receipts[0]['cursor_before'])]+[tuple(r['cursor_after']) for r in receipts]

def line(points,ox,oy,scale=10):
    return ' '.join(f'{ox+(x-40)*scale:.1f},{oy+(y-40)*scale:.1f}' for x,y in points)

def generate(evidence,output):
    data=json.loads((Path(evidence)/'trajectories.json').read_text())
    lookup={(c['case'],c['rot']*90,c['kind'],c['budget']):c for c in cal.cases('development')}
    # Keep three arms on a matched root for every row, and select contrasting cases.
    starts={}
    for item in data:
        if item['budget']!=1:continue
        k=(item['case'],item['rotation'],item['kind'],item['original_budget'])
        starts.setdefault(k,{})[item['arm']]=item
    ordered=sorted(starts, key=lambda k:(
        -(starts[k].get('paired',{}).get('first')!=starts[k].get('static',{}).get('first')),
        -int(lookup[k]['depth']>=2),k))
    selected=ordered[:2]
    if len(selected)<2:raise ValueError('Need two source-matched cases')
    tiles=[];width=900;height=590
    for row,k in enumerate(selected):
        c=lookup[k];begin=path_points(c['state']);goal=c['goal']
        for col,arm in enumerate(cap.ARMS):
            rec=starts[k][arm];ox=70+col*300;oy=108+row*275
            tail=[tuple(z['receipt']['cursor_after']) for z in rec['rows']]
            old=line(begin,ox,oy);recent=line([begin[-1]]+tail,ox,oy)
            gx=ox+(goal[0]-40)*10;gy=oy+(goal[1]-40)*10
            ex,ey=(begin[-1] if not tail else tail[-1]);ex=ox+(ex-40)*10;ey=oy+(ey-40)*10
            title=f'{arm} | {"positive" if c["kind"]=="former_positive" else "negative"} | depth {c["depth"]} | +1 cap'
            txt=f'{"hit" if rec["hit"] else "no hit"}; {rec["steps"]} legal actions; {rec["first"] or "none"}'
            tiles.append(f'<rect x="{col*300+8}" y="{row*275+45}" width="284" height="259" fill="#131d2a" stroke="#43546a" rx="8"/>')
            tiles.append(f'<text x="{col*300+22}" y="{row*275+67}" fill="#f0f5fc" font-size="12">{html.escape(title)}</text>')
            tiles.append(f'<polyline points="{old}" fill="none" stroke="#69a7dd" stroke-width="2.4" stroke-linejoin="round" opacity=".78"/>')
            if tail:tiles.append(f'<polyline points="{recent}" fill="none" stroke="#f4a94e" stroke-width="3.3" stroke-linejoin="round"/>')
            tiles.append(f'<circle cx="{gx}" cy="{gy}" r="5.3" fill="#7ee0b0" stroke="#15212a" stroke-width="1"/>')
            tiles.append(f'<circle cx="{ex}" cy="{ey}" r="3.4" fill="#f4a94e"/>')
            tiles.append(f'<text x="{col*300+22}" y="{row*275+283}" fill="#a8b4c8" font-size="12">{html.escape(txt)}</text>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0c1420"/>
<text x="18" y="25" fill="#f0f5fc" font-family="sans-serif" font-size="19" font-weight="bold">DRIV3R • CAP-CONDITIONING — synthetic legal history</text>
<text x="18" y="42" fill="#9db0c5" font-family="sans-serif" font-size="11">Blue: complete legal prehistory • Orange: autonomous continuation • Green: former goal • no native game acceptance</text>
<g font-family="sans-serif">{''.join(tiles)}</g></svg>'''
    Path(output).write_text(svg)
    return {'tiles':len(tiles),'matched_root_states':len(selected),'path_source':'world decision receipts'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(generate(a.run,a.out))
