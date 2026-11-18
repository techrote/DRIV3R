"""Render real source-trace witness paths against three counterfactual budget caps."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path
from . import budget_cap_intervention_lab as lab


def svg(report_path, out):
    data=json.loads(Path(report_path).read_text());cases={}
    for c in lab.select_cases():
        cases[(c['case'],c['rot']*90,c['kind'],c['budget'])]=c
    seen=set();samples=[]
    for arm in ('calibrated','fixed3'):
        for w in (x for x in data['witnesses'] if x['arm']==arm):
            key=(w['case'],w['rotation'],w['kind'],w['original_budget'])
            if key[0] not in seen:
                samples.append((w,cases[key]));seen.add(key[0])
            if len([t for t,_ in samples if t['arm']==arm])>=3:break
    if len(samples)<6:raise ValueError('Not enough independent witnesses for gallery')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="770" viewBox="0 0 1200 770">',
           '<rect width="1200" height="770" fill="#0b1521"/>',
           '<text x="35" y="43" fill="#f0f4f8" font-family="sans-serif" font-size="26" font-weight="bold">DRIV3R · Counterfactual recovery caps</text>',
           '<text x="35" y="68" fill="#a9bfd0" font-family="sans-serif" font-size="14">Each panel holds the exact host state and path fixed. Only the synthetic maximum recovery allowance changes. NOT native gameplay.</text>']
    for i,(w,c) in enumerate(samples):
        col=i%2;row=i//2;bx=30+col*590;by=98+row*216
        parts.append(f'<rect x="{bx}" y="{by}" width="564" height="198" rx="10" fill="#172638" stroke="#36526b"/>')
        arm=html.escape(w['arm']);case=html.escape(w['case'][:12]);kind=html.escape(w['kind'])
        parts.append(f'<text x="{bx+17}" y="{by+25}" fill="white" font-family="monospace" font-size="16">{arm} · {kind} · {case}</text>')
        pts=[p for p in c['state'].path if 47<=p[0]<=52 and 47<=p[1]<=52]
        if pts:
            path=' '.join(f'{bx+27+(p[0]-47)*24},{by+58+(p[1]-47)*24}' for p in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="#6fc5c0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        gx,gy=c['goal'];cx,cy=c['state'].cursor
        parts.append(f'<circle cx="{bx+27+(gx-47)*24}" cy="{by+58+(gy-47)*24}" r="6" fill="#f7c56e"/>')
        parts.append(f'<circle cx="{bx+27+(cx-47)*24}" cy="{by+58+(cy-47)*24}" r="5" fill="#ff777c"/>')
        for j,(budget,choice,prob) in enumerate(zip(lab.BUDGETS,w['choices'],w['backtrack_probs'])):
            parts.append(f'<text x="{bx+209}" y="{by+70+j*37}" font-family="monospace" font-size="15" fill="#f1f4f9">cap→ {budget} left: {html.escape(choice)} · P(B)={prob:.3f}</text>')
        parts.append(f'<text x="{bx+209}" y="{by+178}" fill="#adc4d8" font-family="monospace" font-size="13">history depth {w["depth"]} · start credit {w["original_budget"]}</text>')
    parts.append('<text x="30" y="760" font-family="sans-serif" font-size="13" fill="#95aec2">Cyan: actual legal prehistory suffix · amber: history goal · coral: current cursor · numbers: inference only, no counterfactual execution</text>')
    parts.append('</svg>');Path(out).write_text('\n'.join(parts)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report_json');p.add_argument('output_svg')
    a=p.parse_args();svg(a.report_json,a.output_svg)
