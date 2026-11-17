"""Render eight selected executed synthetic trajectories from compact JSON to SVG HTML."""
from pathlib import Path
import argparse, html, json

def render(source, output):
    data=json.loads(Path(source).read_text())
    cards=[]
    for r in data['traces']:
        coords=[r['start']]+r['positions']
        pts=' '.join(f'{(p%15)*20+10},{(p//15)*20+10}' for p in coords)
        grid=' '.join(f'M{i},0V300 M0,{i}H300' for i in range(0,301,20))
        title=' · '.join((r['c'],r['family'],r['model']))
        cards.append('<article><h3>'+html.escape(title)+'</h3>'+
          '<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000">'+
          '<rect width="300" height="300" fill="#10212e"/>'+
          f'<path d="{grid}" stroke="#324759" stroke-width=".35"/>'+
          f'<polyline points="{pts}" fill="none" stroke="#34c8d5" stroke-width="2"/>'+
          f'<circle cx="{(coords[0]%15)*20+10}" cy="{(coords[0]//15)*20+10}" r="4" fill="#ffd26a"/>'+
          f'<circle cx="{(coords[-1]%15)*20+10}" cy="{(coords[-1]//15)*20+10}" r="4" fill="#fa7a8b"/>'+
          '</svg><p>'+str(r['goals'])+' goals · '+str(r['steps'])+' legal actions</p></article>')
    style='<style>body{background:#09131c;color:#e2e9f1;font:15px system-ui;margin:1.5rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}article{background:#182633;padding:12px;border-radius:9px}svg{width:100%;aspect-ratio:1}h3{font-size:15px}p{color:#b3c0cc;font-size:13px}</style>'
    page='<!doctype html><meta charset="utf-8"><title>DRIV3R C41 — Executed paths</title>'+style+'<h1>C41 · Prim transfer, executed paths</h1><p>Orange = start, red = end. Grid denotes coordinates only, not protected cells. These are synthetic-host paths, not CyberSand native capture.</p><main>'+''.join(cards)+'</main>'
    Path(output).write_text(page)
    return len(cards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output')
    args=p.parse_args();print(render(args.source,args.output),'rendered panels')
