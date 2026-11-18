'use strict';
// Corpus Factory v30: independent verification of emitted compact traces and
// counterfactual controls; synthetic host, not native CyberSand.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const gen=require('./dual_history_v30.js');
const bounded=require('./bounded_view_v30.js');
const hash=v=>crypto.createHash('sha256').update(Buffer.isBuffer(v)||typeof v==='string'?v:JSON.stringify(v)).digest('hex');
function read(dir,name){return fs.readFileSync(path.join(dir,name),'utf8').trimEnd().split('\n').map(JSON.parse);}
function verify(dir){
 const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 assert.equal(manifest.schema,'driv3r/corpus_factory/dual-history-v30/manifest');
 assert.equal(manifest.generator_sha256,hash(fs.readFileSync(path.join(__dirname,'dual_history_v30.js'))));
 assert.equal(manifest.bounded_view_source_sha256,hash(fs.readFileSync(path.join(__dirname,'bounded_view_v30.js'))));
 const allInputs=new Map(),allGeometry=new Map();let nEp=0,nDec=0,nErase=0;
 for(const [split,m] of Object.entries(manifest.splits)){
  const eps=read(dir,split+'_episodes.jsonl'),dec=read(dir,split+'_decisions.jsonl');
  assert.equal(eps.length,m.episodes);assert.equal(dec.length,m.decisions);assert.equal(eps.length,m.pairs*2);
  assert.equal(m.missing_strict_worlds.length,0);assert.equal(m.exact_duplicates,0);
  const byEpisode=new Map(),pairs=new Map(),first=new Map();
  for(const e of eps){
   assert.equal(e.terminal_success,true);assert.equal(e.erases,2);assert.ok(e.first_action_regret>=2);
   assert.ok(e.difficulty.post_paint_distractor_moves>=160);
   assert.ok(e.difficulty.remote_distance>=3);
   assert.equal(e.hidden_interventions.target_indices.length,2);
   if(!pairs.has(e.pair_id))pairs.set(e.pair_id,[]);pairs.get(e.pair_id).push(e);
   const prev=allGeometry.get(e.geometry_sha256);assert.ok(!prev||prev===split,'cross split geometry leak');allGeometry.set(e.geometry_sha256,split);
   byEpisode.set(e.episode_id,{head:e,history:structuredClone(e.initial_history),count:0,erases:0});
  }
  for(const r of dec){
   const p=byEpisode.get(r.episode_id);assert.ok(p,'decision episode exists');
   p.history.events.push(...r.events_added);p.history.positions.push(...r.positions_added);
   const obs={...r.state,history:p.history};
   assert.equal(hash(obs),r.model_input_sha256,'full observation reconstruction');
   assert.equal(hash(r.bounded_model_input),r.bounded_model_input_sha256,'bounded hash');
   assert.deepEqual(bounded.compress(obs),r.bounded_model_input,'public-only bounded encoding');
   assert.ok(JSON.stringify(r.bounded_model_input).length<=2048);
   const old=allInputs.get(r.model_input_sha256);assert.ok(!old||old.split===split&&old.action===r.target,'no cross-split input leak or label conflict');
   allInputs.set(r.model_input_sha256,{split,action:r.target});
   if(r.step===0)first.set(r.episode_id,{obs:structuredClone(obs),action:r.target});
   if(r.target==='ERASE')p.erases++;
   assert.equal(r.step,p.count++,'step monotonic');
   assert.equal(r.receipt.accepted,true,'host applied legal oracle action');
  }
  for(const [key,arms] of pairs){
   assert.equal(arms.length,2,'paired counterfactuals');const a=first.get(arms[0].episode_id),b=first.get(arms[1].episode_id);
   assert.ok(a&&b);assert.notEqual(a.action,b.action,'opposite first action');
   assert.equal(hash(bounded.compress(a.obs,{memory:false})),hash(bounded.compress(b.obs,{memory:false})),'no-memory controls tied');
   assert.equal(hash(bounded.compress(a.obs,{lastPaintOnly:true})),hash(bounded.compress(b.obs,{lastPaintOnly:true})),'last-paint controls tied');
   assert.equal(hash(a.obs.history.events.slice(-64)),hash(b.obs.history.events.slice(-64)),'recent-64 tied');
   assert.notEqual(hash(a.obs),hash(b.obs),'full history identifies case');
  }
  for(const p of byEpisode.values()){assert.equal(p.count,p.head.steps);assert.equal(p.erases,2);}
  const tally=eps.reduce((o,e)=>(o[e.first_action]=(o[e.first_action]||0)+1,o),{});
  assert.deepEqual(tally,m.first_action);
  nEp+=eps.length;nDec+=dec.length;nErase+=eps.reduce((n,e)=>n+e.erases,0);
  for(const suffix of ['episodes.jsonl','decisions.jsonl']){
   const name=split+'_'+suffix,data=fs.readFileSync(path.join(dir,name));
   assert.equal(hash(data),manifest.shards[name].sha256,'shard hash');
   assert.equal(data.length,manifest.shards[name].bytes,'shard bytes');
  }
 }
 console.log('PASS independent strict counterfactual controls / exact trace reconstruction / source & shard hashes',JSON.stringify({episodes:nEp,decisions:nDec,erases:nErase}));
 return manifest;
}
if(require.main===module){
 const dir=process.argv[2]||fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v30-'))+'/dataset';
 if(!process.argv[2])gen.generate(dir,{quick:true});
 const m=verify(dir);
 if(!m.quick){assert.equal(m.splits.TRAIN.episodes,128);assert.equal(m.splits.DEV.episodes,64);assert.equal(m.splits.HELDOUT.episodes,64);}
}
module.exports={verify};
