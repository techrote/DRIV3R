"""D022 read-only replay audit against independently emitted evidence receipts.

This uses the same archived synthetic Game fixture, but does not use the
experiment's rollout/summary functions or retrain any student checkpoint.
It is not independent native-game verification.
"""
from __future__ import annotations
import argparse, hashlib, io, json, collections
from pathlib import Path
import numpy as np
from .lab import STYLES, Game
from .repair import load, adverse
from .source_parent_branch_gate import Controller, Gate, parents, sha, js
from .iterated_onpolicy_022 import CASES, TEST, CONFIRM, STRESS

def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]

def replay(out,d007,d017,q8,d021):
 m=json.loads((out/'d022_metrics.json').read_text())
 selected=m['selected_for_diagnostic']
 models={'d017':Controller(load(d007),parents(q8),Gate.read(d017),.5,.58),
         'd021':load(d021),'d022':load(out/f'd022_{selected}.npz')}
 receipts={(r['seed'],r['style'],r['case'],r['policy'],r['tick']):r
           for r in rows(out/'d022_action_receipts.jsonl')}
 audited=0;actions=0;matched_receipts=0;png_match=0;illegal=0;observed=set()
 for split,seeds in (('test',TEST),('confirmation',CONFIRM),('stress',STRESS)):
  expected={(r['seed'],r['style'],r['case'],r['model']):r
            for r in rows(out/f'd022_{split}.jsonl')}
  actual=[]
  for seed in seeds:
   for style in STYLES:
    for case in CASES:
     for name,model in models.items():
      g=Game(seed,style)
      for tick in range(80):
       adverse(g,case)
       o=g.obs();act=model.predict(o)
       if act not in o['legal']:illegal+=1;raise AssertionError('illegal action')
       k=(seed,style,case,name,tick)
       if k in receipts:
        r=receipts[k];observed.add(k);matched_receipts+=1
        if r['action']!=act or r['legal']!=o['legal'] or r['position']!=[o['x'],o['y']]:
         raise AssertionError('action receipt mismatch '+str(k))
       g.step(act);actions+=1
       if g.stopped:break
      row={'seed':seed,'style':style,'case':case,'model':name,
           'cells':len(g.painted),'branches':g.branch_count,
           'retraces':g.retraces,'revisits':g.revisits,'decisions':g.tick,
           'stopped':int(g.stopped),'illegal':g.invalid}
      kk=(seed,style,case,name)
      if row!=expected[kk]:raise AssertionError('episode mismatch '+str(kk))
      actual.append(row);audited+=1
      if split=='test' and seed in (67001,67005,67009,67013) and style=='branchwitch':
       buf=io.BytesIO();g.png().save(buf,format='PNG')
       f=out/'pngs'/f'{case}_{seed}_{name}.png'
       if hashlib.sha256(buf.getvalue()).digest()!=hashlib.sha256(f.read_bytes()).digest():
        raise AssertionError('PNG mismatch '+str(f))
       png_match+=1
  for case in CASES:
   for name in models:
    for key in ('cells','branches','retraces','revisits','decisions','stopped','illegal'):
     vals=[r[key] for r in actual if r['case']==case and r['model']==name]
     if abs(float(np.mean(vals))-m[split]['summary'][f'{case}/{name}'][key])>0.00011:
      raise AssertionError('metric mismatch '+str((split,case,name,key)))
 if set(receipts)!=observed:raise AssertionError('missing receipts')
 for line in (out/'SHA256SUMS.txt').read_text().splitlines():
  digest,relative=line.split('  ',1)
  if sha(out/relative)!=digest:raise AssertionError('evidence hash mismatch '+relative)
 return {'schema':'driv3r.distillation.d022-replay-audit/1','auditor':'independent_loop_same_archived_synthetic_fixture',
         'episodes_replayed':audited,'decisions_replayed':actions,
         'matched_action_receipts':matched_receipts,'matched_pngs':png_match,
         'illegal_actions':illegal,'all_metrics_reproduced':True,
         'manifest_entries':len((out/'SHA256SUMS.txt').read_text().splitlines()),
         'source_sha256':m['source_sha256'],
         'native_cybersand_executed':False}

def main(argv=None):
 ap=argparse.ArgumentParser()
 for k in ('out','d007','d017','q8','d021'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args(argv);result=replay(a.out,a.d007,a.d017,a.q8,a.d021)
 (a.out/'d022_audit.json').write_text(js(result)+'\n');print(js(result))
if __name__=='__main__':main()
