"""Render a source-linked gallery without altering model-produced W0 PNGs.

The optional magenta crosshatch is an *analytic overlay* of hypothetical
protected cells, not a pixel from the game or a native material readback.
"""
from pathlib import Path
import argparse
import csv
import html
import json
from PIL import Image, ImageDraw, ImageFont
import p14_protected_overlay as exp


def render(inp,layout,seed,output_size=(240,240)):
    image=Image.open(inp).convert('RGB')
    # Original 96x96 host cells are painted at 4x scale, independently decoded.
    over=Image.new('RGBA',image.size,(0,0,0,0))
    draw=ImageDraw.Draw(over)
    for x,y in sorted(exp.protected_cells(seed,layout)):
        a,b=x*4,y*4
        draw.rectangle((a,b,a+3,b+3),fill=(230,40,120,70))
        if (x+y)%2==0:draw.line((a,b,a+3,b+3),fill=(255,80,160,160))
    image=Image.alpha_composite(image.convert('RGBA'),over).convert('RGB')
    return image.resize(output_size,Image.Resampling.NEAREST)


def build(directory):
    directory=Path(directory)
    idx=json.loads((directory/'P14-rollout-index.json').read_text())
    by={(r['family'],r['seed'],r['layout'],r['policy']):r for r in idx}
    width,height=4*286+24,3*300+98
    sheet=Image.new('RGB',(width,height),(24,29,41));d=ImageDraw.Draw(sheet)
    d.text((16,12),'P14 / REAL W16-AUX TRAJECTORIES + HYPOTHETICAL PROTECTED-CELL OVERLAY',fill='white')
    d.text((16,38),'Source PNGs are unchanged in the evidence directory; magenta cells are analytical annotations',fill=(220,208,218))
    for row,family in enumerate(exp.FAMILIES):
        for col,policy in enumerate(exp.POLICIES):
            rec=by[family,2711,'near_origin',policy]
            x=16+col*286;y=75+row*300
            img=render(directory/f'{family}-2711-near_origin-{policy}.png','near_origin',2711)
            sheet.paste(img,(x,y+25))
            d.text((x,y),f'{family} | {policy}',fill='white')
            d.text((x,y+269),f"novel {rec['novel_positions']}  contacts {rec['protected_contacts']}  goal {int(rec['goal_completed'])}",fill=(220,220,225))
    out=directory/'P14-12-original-traces-protected-overlay-ANNOTATED.png'
    sheet.save(out,optimize=True)
    lines=['<!doctype html><html lang="en"><meta charset="utf-8"><title>P14 W16 432-trace protected overlay bestiary</title>',
        '<style>body{background:#161b26;color:#eff2f7;font:14px system-ui;margin:20px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:12px}article{border:1px solid #485064;border-radius:9px;padding:10px;background:#202738}img{width:100%;image-rendering:pixelated}select{padding:6px;margin:5px}small{color:#c2cbdd}a{color:#b7d4ff} .title{font-weight:bold} </style>',
        '<h1>P14 — 432 closed-loop W16 auxiliary-head rollouts</h1>',
        '<p><b>Research-only:</b> W0 synthetic host; protected rectangles are a hypothetical position overlay, NOT native protected paint/materials. Source PNGs below are <b>unannotated original W0 outputs</b>. All preference labels are algorithmic. No owner/human ratings.</p>',
        '<div><label>Family <select id="family"><option value="">all</option>'+''.join(f'<option>{x}</option>' for x in exp.FAMILIES)+'</select></label>',
        '<label>Policy <select id="policy"><option value="">all</option>'+''.join(f'<option>{x}</option>' for x in exp.POLICIES)+'</select></label>',
        '<label>Mask <select id="layout"><option value="">all</option>'+''.join(f'<option>{x}</option>' for x in exp.LAYOUTS)+'</select></label></div>',
        '<p><a href="P14-results.json">Summary</a> · <a href="P14-rollout-index.json">Hash/index</a> · <a href="P14-pairs.json">Traceable weak-supervision pairs</a></p><div class="grid" id="gallery">']
    for r in idx:
        name=f"{r['family']}-{r['seed']}-{r['layout']}-{r['policy']}"
        lines.extend([f'<article data-family="{r["family"]}" data-layout="{r["layout"]}" data-policy="{r["policy"]}">',
            f'<div class="title">{html.escape(name)}</div>',
            f'<a href="{name}.png"><img loading="lazy" alt="real W0 model-produced material drawing" src="{name}.png"></a>',
            f'<div>New cells {r["novel_positions"]} · protected contacts {r["protected_contacts"]} · goal completed {r["goal_completed"]} · steps {r["steps"]}</div>',
            f'<small>Source PNG SHA-256: {r["png_sha256"]}<br>Trace SHA-256: {r["trace_sha256"]}</small><p><a href="{name}.json">Trace/host receipts</a></p></article>'])
    lines+=['</div><script>for(const id of ["family","layout","policy"]) document.getElementById(id).addEventListener("change",()=>{for(const card of document.querySelectorAll("article")){card.hidden=["family","layout","policy"].some(k=>document.getElementById(k).value && card.dataset[k]!==document.getElementById(k).value);}});</script></html>']
    (directory/'P14-432-model-bestiary.html').write_text('\n'.join(lines))
    with (directory/'P14-432-trajectory-metrics.csv').open('w',newline='') as f:
        keys=('family','seed','cohort','layout','policy','source_trace_sha256','trace_sha256','png_sha256','mask_sha256','novel_positions','goal_completed','protected_contacts','protected_unique','branches','retraces','steps','stopped','invalid')
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(idx)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rollouts',type=Path,required=True)
    args=parser.parse_args();print(build(args.rollouts))
