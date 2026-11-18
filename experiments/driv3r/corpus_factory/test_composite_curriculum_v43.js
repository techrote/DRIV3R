'use strict';
// Independent full-row audit. Checks manifest pinning, no teacher leakage, split/geometry owners,
// source and provenance replay, train task balance, and two output run byte identity.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const src=require('./composite_curriculum_v43.js');
const ok=(x,m)=>{if(!x)throw Error(m);},sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const rows=(d,n)=>fs.readFileSync(path.join(d,n),'utf8').trimEnd().split('\n').map(JSON.parse);
function audit(a,b,v38,v42){
 const A=fs.readFileSync(path.join(a,'manifest.json')),B=fs.readFileSync(path.join(b,'manifest.json'));
 ok(A.equals(B),'manifest divergence');const m=JSON.parse(A);
 ok(m.schema==='driv3r/corpus_factory/composite-v43/manifest','wrong version');
 const sources={v38,v42},manifests={v38:src.verify(v38,'v38'),v42:src.verify(v42,'v42')};
 let counted=0,zero=0,records=0,challenge=0,dup=0;
 const owner=new Map(),geometry=new Map();
 for(const [name,stat] of Object.entries(m.shards)){
  const x=fs.readFileSync(path.join(a,name)),y=fs.readFileSync(path.join(b,name));
  ok(x.equals(y)&&sha(x)===stat.sha256&&x.length===stat.bytes,'bad shard '+name);
  ok(x.toString().trimEnd().split('\n').length===stat.rows,'bad rows '+name);
 }
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const policy=rows(a,split+'_policy.jsonl'),meta=rows(a,split+'_provenance_EVALUATOR_ONLY.jsonl');
  ok(policy.length===meta.length,'missing source refs');
  const counts={v38:0,v42:0};
  // Recheck source rows once only, preserving source-row offsets.
  const parent={v38:rows(v38,split+'_policy.jsonl'),v42:rows(v42,split+'_policy.jsonl')};
  for(let i=0;i<policy.length;i++){
   const r=policy[i],p=meta[i],orig=parent[p.suite][p.source_row];
   ok(JSON.stringify(Object.keys(r).sort())===JSON.stringify(['action','input','weight']),'teacher leak');
   ok(orig.action===r.action&&JSON.stringify(orig.input)===JSON.stringify(r.input),'source manipulation '+i);
   ok(r.input.legal.includes(r.action)&&[0,1].includes(r.weight),'illegal '+i);
   ok(p.geometry_sha256===src.geom(r.input),'provenance geometry mismatch');
   ok(p.durable_receipts===r.input.durable.length,'provenance history mismatch');
   const h=sha(JSON.stringify(r.input)),g=p.geometry_sha256;
   ok(!owner.has(h)||owner.get(h).split===split,'cross-split model input leak');
   ok(!geometry.has(g)||geometry.get(g)===split,'cross-split topology leak');
   if(owner.has(h))ok(owner.get(h).action===r.action,'contradictory action');
   owner.set(h,{split,action:r.action});geometry.set(g,split);
   ok(p.suite==='v38'||p.suite==='v42','invalid source');
   if(split==='TRAIN')ok(r.weight===1,'unnecessary duplicate in TRAIN');
   counts[p.suite]++;zero+=!r.weight;records++;
  }
  ok(counts.v38===m.splits[split].selected_rows.v38&&counts.v42===m.splits[split].selected_rows.v42,'task count changed');
  if(split==='TRAIN'){
   ok(counts.v38===counts.v42,'unbalanced TRAIN');
   const selected=meta.filter(x=>x.suite==='v38');const selectedNums=new Set(selected.map(x=>x.source_row));
   for(let j=0;j<parent.v38.length;j++)if(parent.v38[j].weight===1 && ['ERASE','STOP','PAINT_A','PAINT_B'].includes(parent.v38[j].action))
     ok(selectedNums.has(j),'rare supervised v38 action omitted '+j);
  } else {
   for(const suite of ['v38','v42'])ok(counts[suite]===manifests[suite].shards[split+'_policy.jsonl'].rows,'evaluation sampled '+suite);
   for(const suite of ['v38','v42'])for(const kind of suite==='v38'?['challenge','memory']:['challenge']){
    const n=split+'_'+suite+'_'+kind+'.jsonl';const orig=split+'_'+kind+'.jsonl';
    const x=fs.readFileSync(path.join(a,n)),y=fs.readFileSync(path.join(sources[suite],orig));
    ok(x.equals(y),'evaluator challenge changed '+n);challenge+=rows(a,n).length;
   }
  }
 }
 return {policy_rows:records,source_verifiable:records,output_shards:Object.keys(m.shards).length,
  evaluator_only_rows:challenge,original_weight_zero:zero,topologies:geometry.size,
  source_manifests_pinned:Object.fromEntries(Object.keys(src.PIN).map(s=>[s,src.PIN[s].manifest])),
  manifest_sha256:sha(A)};
}
if(require.main===module){const [a,b,v38,v42]=process.argv.slice(2);ok(a&&b&&v38&&v42,'usage: node test_composite_curriculum_v43.js GEN_A GEN_B SRC38 SRC42');
 console.log(JSON.stringify(audit(a,b,v38,v42),null,2));}
module.exports={audit};
