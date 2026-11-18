'use strict';
// Independent full-shard manifest, split, mapping and paired-effect audit.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
function assert(x,m){if(!x)throw Error(m);}
async function* records(p){const s=fs.createReadStream(p),r=readline.createInterface({input:s,crlfDelay:Infinity});try{for await(const line of r)if(line)yield JSON.parse(line);}finally{r.close();s.destroy();}}
function lawEffect(a,law){return law==='inverted'?(a==='PAINT_A'?'PAINT_B':a==='PAINT_B'?'PAINT_A':a):a;}
function original(i){const {law,...rest}=i;return rest;}
function geom(i){return sha([i.cells,i.edges,i.protected,i.gate.slice(0,2)]);}
async function audit(dir){const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 assert(m.schema==='driv3r/corpus_factory/material-law-forks-v37/manifest','wrong schema');
 const sources=new Map(),geoms=new Map();let decisions=0,positive=0,controls=0,weight1=0,zero=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  let n=0,pairs=0,changes=0,stables=0,nonzero=0,zeros=0,duplicates=0;
  let previous=null;
  for await(const row of records(path.join(dir,split+'_policy.jsonl'))){
   n++;decisions++;const i=row.input;assert(row&&i&&typeof i==='object','bad row');
   assert(Object.keys(row).sort().join(',')==='action,input,weight','teacher metadata in policy row');
   assert(!Object.hasOwn(i,'action')&&!Object.hasOwn(i,'weight')&&!Object.hasOwn(i,'cost')&&!Object.hasOwn(i,'regret')&&!Object.hasOwn(i,'episode_id'),'target leakage');
   assert(row.weight===0||row.weight===1,'bad weight');if(row.weight){nonzero++;weight1++;}else{zeros++;zero++;}
   assert(i.legal.includes(row.action),'illegal action');
   const law=i.law.paint_A_yields==='A'?'normal':'inverted';
   assert(i.law.kind==='public_tool_output_mapping'&&i.law.scope==='FROM_THIS_DECISION_ONWARD','bad law context');
   assert(i.law.paint_B_yields===(law==='normal'?'B':'A'),'inconsistent law');
   const key=sha(i);const owner=sources.get(key);assert(!owner||owner.split===split,'cross split exact input leakage');
   if(owner){assert(owner.action===row.action,'conflicting teacher action');duplicates++;}
   else sources.set(key,{split,action:row.action});
   const g=geom(i);const old=geoms.get(g);assert(!old||old===split,'cross split graph geometry leakage');geoms.set(g,split);
   if(n%2===1){assert(law==='normal','wrong pair order');previous=row;continue;}
   assert(law==='inverted','second row not inverted');
   assert(sha(original(i))===sha(original(previous.input)),'pair inputs differ beyond law');
   const baseline=previous.action;
   assert(lawEffect(row.action,'inverted')===baseline,'law inversion breaks physical effect');
   const isPaint=baseline==='PAINT_A'||baseline==='PAINT_B';
   assert((baseline!==row.action)===isPaint,'unexpected change/non-change');
   if(isPaint){changes++;positive++;
     const t=i.targets.find(x=>x[0]===i.p);assert(t&&baseline==='PAINT_'+t[1],'wrong paint law/target');
     assert(i.local[0][4]==='EMPTY','non-empty source before paint');
   }else{stables++;controls++;}
   pairs++;
  }
  assert(n===m.splits[split].policy_rows&&pairs===m.splits[split].paired_decisions,'policy count mismatch');
  assert(changes===m.splits[split].changed_action_pairs&&stables===m.splits[split].unchanged_action_controls,'effect count mismatch');
  assert(nonzero===m.splits[split].weight_one&&zeros===m.splits[split].weight_zero,'loss weight count mismatch');
  assert(duplicates===m.splits[split].exact_duplicate_inputs,'duplicate count mismatch');
  let ch=0;
  for await(const pair of records(path.join(dir,split+'_counterfactual.jsonl'))){ch++;
   assert(split!=='TRAIN','training must not expose eval paired labels');
   assert(sha(original(pair.normal.input))===pair.law_free_input_hash,'challenge source hash mismatch');
   assert(sha(original(pair.inverted.input))===pair.law_free_input_hash,'challenge other source mismatch');
   assert(pair.changes_optimal_action===(pair.normal.action!==pair.inverted.action),'challenge impact mismatch');
  }
  assert(ch===m.shards[split+'_counterfactual.jsonl'].rows,'challenge count mismatch');
  for(const label of ['policy','counterfactual']){
   const file=path.join(dir,split+'_'+label+'.jsonl'),stat=m.shards[split+'_'+label+'.jsonl'];const data=fs.readFileSync(file);
   assert(data.length===stat.bytes&&sha(data)===stat.sha256,'shard integrity '+file);
  }
 }
 assert(positive>0&&controls>0&&decisions===positive*2+controls*2,'total pair mismatch');
 const raw=fs.readFileSync(path.join(dir,'manifest.json'));
 assert(m.generator_sha256===sha(fs.readFileSync(path.join(__dirname,'material_law_forks_v37.js'))),'generator drift');
 return {decision_rows:decisions,paint_counterfactual_pairs:positive,unchanged_action_control_pairs:controls,weighted_rows:weight1,zero_weight_rows:zero,unique_topologies:geoms.size,manifest_sha256:sha(raw)};
}
if(require.main===module){const [a,b]=process.argv.slice(2);assert(a,'usage: node test_material_law_forks_v37.js OUT_A [OUT_B]');
 (async()=>{const x=await audit(a);if(b){const y=await audit(b);assert(x.manifest_sha256===y.manifest_sha256,'manifest not reproducible');const m=JSON.parse(fs.readFileSync(path.join(a,'manifest.json')));for(const file of Object.keys(m.shards))assert(sha(fs.readFileSync(path.join(a,file)))===sha(fs.readFileSync(path.join(b,file))),'shard not byte-identical '+file);}console.log(JSON.stringify(x,null,2));})().catch(e=>{console.error(e.stack);process.exitCode=1;});}
module.exports={audit};
