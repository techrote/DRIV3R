'use strict';
/* DRIV3R Corpus Factory v43: reproducible, split-guarded multimission export.
 * Inputs: full generated v38 & v42 synthetic-host corpora (not native CyberSand).
 * Teacher fields stay outside model input; no new oracle policy or model training.
 */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const assert = (ok,m) => {if(!ok) throw Error(m);};
const sha = b => crypto.createHash('sha256').update(Buffer.isBuffer(b)||typeof b==='string'?b:JSON.stringify(b)).digest('hex');
const PIN = Object.freeze({
 v38:{manifest:'0d0c3c8f821dc9900863d45c9aac916014e4d66e644f99799c491a82f6554f5f',schema:'driv3r/corpus_factory/host-live-law-v38/manifest',generator:'599aaaa8b70007cd8a50be6e569d7e7a2d0e40ef7119b3cb40c889950fe95976'},
 v42:{manifest:'fce49195cdfae48ec02a71928a85fecfe398f4e3c93e8b2aefa0fa0ff5af2834',schema:'driv3r/corpus_factory/dual-interlock-v42/manifest',generator:'e08d8c186b54b4680a59c33169ec150f0873b4f309303a57ef909ad01449d993'}
});
const SPLITS=['TRAIN','DEV','HELDOUT'];
function verify(root,suite){
 const b=fs.readFileSync(path.join(root,'manifest.json')),m=JSON.parse(b.toString());
 assert(sha(b)===PIN[suite].manifest&&m.schema===PIN[suite].schema,'unrecognized/corrupt '+suite+' parent manifest');
 assert(m.generator_sha256===PIN[suite].generator&&m.quick===false,'wrong source/quick parent '+suite);
 for(const [name,x] of Object.entries(m.shards)){
   const b=fs.readFileSync(path.join(root,name));
   assert(b.length===x.bytes&&sha(b)===x.sha256,'parent shard mismatch '+suite+'/'+name);
   assert(b.at(-1)===10,'missing JSONL final newline '+name);
 }
 return m;
}
function sourceRows(root,split){
 return fs.readFileSync(path.join(root,split+'_policy.jsonl'),'utf8').trimEnd().split('\n').map((s,i)=>{
  const r=JSON.parse(s);
  assert(Object.keys(r).sort().join('|')==='action|input|weight','teacher metadata contaminates source '+split+'/'+i);
  assert(r.input&&typeof r.input==='object'&&r.input.legal.includes(r.action),'illegal label '+split+'/'+i);
  assert([0,1].includes(r.weight),'invalid weight '+split+'/'+i);
  return {input:r.input,action:r.action,weight:r.weight,from:i};
 });
}
function geom(input){return sha([input.cells,input.edges]);}
function entry(r,suite){
 const x=r.input;
 return {suite,source_row:r.from,geometry_sha256:geom(x),durable_receipts:x.durable.length,
  source_version:x.v,action_kind:r.action.startsWith('PAINT')?'PAINT':r.action==='ERASE'?'ERASE':r.action==='STOP'?'STOP':'NAV'};
}
const score=r=>sha('v43-balanced-train-select/sha256/v1|'+JSON.stringify(r.input));
const order=r=>sha('v43-final-shuffle/sha256/v1|'+JSON.stringify(r.input));
function balanced(a,b){
 // All v42 TRAIN rows (no zero weight); sample the same number of v38
 // unique-weight rows, retaining all rare PAINT/ERASE/STOP labels first.
 assert(b.every(r=>r.weight===1),'unexpected v42 zero-weight');
 const good=a.filter(r=>r.weight===1),critical=good.filter(r=>r.action.startsWith('PAINT_')||r.action==='ERASE'||r.action==='STOP');
 const nav=good.filter(r=>!critical.includes(r)).map(r=>({r,key:score(r)})).sort((x,y)=>x.key.localeCompare(y.key)||x.r.from-y.r.from).map(x=>x.r);
 assert(critical.length<=b.length && good.length>=b.length,'cannot construct balanced TRAIN');
 const selected=[...critical,...nav.slice(0,b.length-critical.length)];
 assert(selected.length===b.length&&selected.every(x=>x.weight===1),'balance failed');
 return {v38:selected,v42:b,excluded_v38:good.length-selected.length,source_zero_excluded:a.length-good.length,
  kept_critical:critical.length,kept_navigation:b.length-critical.length};
}
function writeRows(out,file,rows){
 const buf=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');
 fs.writeFileSync(path.join(out,file),buf,{flag:'wx'});
 return {rows:rows.length,bytes:buf.length,sha256:sha(buf)};
}
function generate(v38dir,v42dir,out){
 assert(!fs.existsSync(out),'refuse overwriting '+out);
 const roots={v38:v38dir,v42:v42dir},parents={};
 for(const suite of ['v38','v42'])parents[suite]=verify(roots[suite],suite);
 fs.mkdirSync(out,{recursive:true});
 const splitStats={},shards={},globalInput=new Map(),globalGeom=new Map();
 for(const split of SPLITS){
  const a=sourceRows(v38dir,split),b=sourceRows(v42dir,split);
  const choice=split==='TRAIN'?balanced(a,b):{v38:a,v42:b};
  const selected=['v38','v42'].flatMap(s=>choice[s].map(r=>({r,s,sortKey:order(r)})));
  // Fix stable balanced ordering without including action labels or provenance in the model input.
  selected.sort((x,y)=>x.sortKey.localeCompare(y.sortKey)||x.s.localeCompare(y.s)||x.r.from-y.r.from);
  const outputs=[],provenance=[],counts={v38:0,v42:0},actions={},geos={v38:new Set(),v42:new Set()};
  let zero=0,conflicts=0,nearAliases=0;
  const seen=new Map(),localAliases=new Map();
  for(const {r,s} of selected){
   const x=r.input,key=sha(x),g=geom(x),previous=seen.get(key),prior=globalInput.get(key),priorG=globalGeom.get(g);
   assert(!prior||prior.split===split,'CROSS-SPLIT EXACT INPUT LEAK '+split+'/'+s);
   assert(!priorG||priorG.split===split,'CROSS-SPLIT TOPOLOGY LEAK '+split+'/'+s);
   assert(previous===undefined||previous===r.action,'conflicting label '+split+'/'+s);
   const normalized={input:x,action:r.action,weight:r.weight};
   if(previous!==undefined&&split==='TRAIN')normalized.weight=0;
   if(previous!==undefined)nearAliases++;
   if(!normalized.weight)zero++;
   seen.set(key,r.action);globalInput.set(key,{split,s});globalGeom.set(g,{split,s});
   geos[s].add(g);counts[s]++;actions[r.action]=(actions[r.action]||0)+1;
   outputs.push(normalized);provenance.push(entry(r,s));
  }
  assert(outputs.every(r=>Object.keys(r).sort().join('|')==='action|input|weight'),'unsafe output schema');
  assert(counts.v38>0&&counts.v42>0,'empty suite');
  if(split==='TRAIN')assert(counts.v38===counts.v42,'unequal TRAIN tasks');
  const policy=split+'_policy.jsonl',index=split+'_provenance_EVALUATOR_ONLY.jsonl';
  shards[policy]=writeRows(out,policy,outputs);shards[index]=writeRows(out,index,provenance);
  if(split!=='TRAIN')for(const [s,names] of [['v38',[split+'_challenge.jsonl',split+'_memory.jsonl']],['v42',[split+'_challenge.jsonl']]]){
    for(const name of names){const file=split+'_'+s+'_'+name.slice(split.length+1);
      const data=fs.readFileSync(path.join(roots[s],name));fs.writeFileSync(path.join(out,file),data,{flag:'wx'});
      shards[file]={rows:parents[s].shards[name].rows,bytes:data.length,sha256:sha(data)};
    }
  }
  splitStats[split]={source_rows:{v38:a.length,v42:b.length},selected_rows:counts,policy_rows:outputs.length,
    weight_zero:zero,full_input_duplicates:nearAliases,geometry_groups:{v38:geos.v38.size,v42:geos.v42.size},
    action_counts:actions,training_sampling:split==='TRAIN'?{
      method:'SHA256 deterministic, all v38 PAINT/ERASE/STOP retained; hashed NAV subset to match v42 weight-one count',
      excluded_v38:choice.excluded_v38,excluded_original_zero_v38:choice.source_zero_excluded,
      kept_v38_critical:choice.kept_critical,kept_v38_navigation:choice.kept_navigation}:null};
 }
 const manifest={schema:'driv3r/corpus_factory/composite-v43/manifest',synthetic_not_native_CyberSand:true,
  generator:'composite_curriculum_v43.js',generator_sha256:sha(fs.readFileSync(__filename)),
  purpose:'Balanced TRAIN from independently qualified v38 dynamic material law and v42 multi-interlock delayed obstruction. Full unsampled DEV/HELDOUT with split and parent-shard guards.',
  upstream:Object.fromEntries(Object.entries(PIN).map(([k,v])=>[k,{...v,shards:parents[k].shards}])),
  model_contract:'Only policy row.input is model-visible. action is teacher-only label; weight 0 excluded. provenance_EVALUATOR_ONLY and challenge/memory shards MUST NEVER be concatenated into prompts, training or model-facing runtime.',
  source_safety:'Original row.input preserved byte-for-byte semantically (JSON parsed then serialized); full host-observed snapshot, synthetic-only. Separate source index is NOT a host signal.',
  curriculum_limitations:['This is a guarded remix of v38 and v42, NOT a newly executed trajectory family','TRAIN is sampled per decision; original host-accepted trajectory order cannot be recovered from this export','Mixed training remains susceptible to symbolic rule parsing and does not demonstrate learned skill transfer','No native CyberSand evaluation, no SLM training or inference, no immutable Actions artifact publication'],
  splits:splitStats,shards};
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
module.exports={generate,verify,balanced,PIN,sha,geom};
if(require.main===module){
 const [a,b,out]=process.argv.slice(2);assert(a&&b&&out,'usage: node composite_curriculum_v43.js FULL_V38_DIR FULL_V42_DIR NEW_OUTPUT_DIR');
 console.log(JSON.stringify(generate(path.resolve(a),path.resolve(b),path.resolve(out)).splits,null,2));
}
