'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const v=require('./compound_remote_v29.js'),base=require('./causal_gate_v26.js');
const sha=base.sha;
const jsonl=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
function check(dir,quick){
 const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'))),inputOwners=new Map(),worldOwners=new Map(),geomOwners=new Map();
 let episodes=0,decisions=0,pairs=0,erasures=0,duplicates=0;
 assert.equal(m.restricted_first_action_theoretical_ceiling,.5);
 for(const [split,stat] of Object.entries(m.splits)){
  const eps=jsonl(path.join(dir,split+'_episodes.jsonl')),ds=jsonl(path.join(dir,split+'_decisions.jsonl'));
  assert.equal(eps.length,stat.episodes);assert.equal(ds.length,stat.decisions);
  assert.equal(eps.length,2*stat.pairs);assert.ok(stat.pairs>=3);assert.ok(stat.min_wrong_choice_regret>=2);
  assert.equal(stat.erases,eps.length*2);
  for(const kind of ['episodes','decisions']){const fn=split+'_'+kind+'.jsonl',b=fs.readFileSync(path.join(dir,fn));
   assert.equal(sha(b),m.shards[fn].sha256);assert.equal(b.byteLength,m.shards[fn].bytes);
  }
  const byEp=new Map();for(const d of ds){if(!byEp.has(d.episode_id))byEp.set(d.episode_id,[]);byEp.get(d.episode_id).push(d);}
  const byPair=new Map();for(const e of eps){if(!byPair.has(e.pair_id))byPair.set(e.pair_id,[]);byPair.get(e.pair_id).push(e);}
  assert.equal(byPair.size,stat.pairs);
  for(const [id,rows] of byPair){assert.equal(rows.length,2,id);const [a,b]=rows;
   assert.notDeepEqual(a.hidden_interventions.wrong_material_cells,b.hidden_interventions.wrong_material_cells);
   assert.equal(a.hidden_interventions.wrong_material_cells.length,2);
   assert.equal(b.hidden_interventions.wrong_material_cells.length,2);
   assert.notEqual(a.first_action,b.first_action);assert.ok(a.first_action_regret>=2&&b.first_action_regret>=2);
   assert.equal(a.restricted_alias_sha256,b.restricted_alias_sha256);
   assert.notEqual(a.initial_complete_observation_sha256,b.initial_complete_observation_sha256);
   pairs++;
  }
  for(const ep of eps){
   episodes++;assert.equal(ep.terminal_success,true);assert.equal(ep.erases,2);assert.equal(ep.difficulty.post_paint_distractor_moves,160);
   assert.ok(ep.difficulty.remote_distance>=3);
   for(const [map,hash] of [[worldOwners,ep.world_sha256],[geomOwners,ep.geometry_sha256]]){
    if(map.has(hash))assert.equal(map.get(hash),split);map.set(hash,split);
   }
   const rows=byEp.get(ep.episode_id);assert.equal(rows.length,ep.steps);assert.equal(rows[0].target,ep.first_action);
   const history=structuredClone(ep.initial_history);assert.ok(history.events.some(e=>e.action?.startsWith('PAINT')));
   assert.equal(history.events.slice(-64).some(e=>e.action?.startsWith('PAINT')||e.kind==='GATE_TRANSITION'),false);
   let localErasures=0;
   for(let i=0;i<rows.length;i++){
    const r=rows[i];assert.equal(r.step,i);history.events.push(...r.events_added);history.positions.push(...r.positions_added);
    const o={};for(const k of ep.observation_keys)o[k]=k==='history'?structuredClone(history):r.state[k];
    assert.equal(sha(o),r.model_input_sha256,ep.episode_id+':'+i);assert.ok(o.legal_actions.includes(r.target));
    assert.equal(r.receipt.accepted,true);
    if(r.target==='ERASE')localErasures++;
    const previous=inputOwners.get(r.model_input_sha256);if(previous)assert.equal(previous.split,split);
    if(previous)assert.equal(previous.target,r.target);
    inputOwners.set(r.model_input_sha256,{split,target:r.target});
    if(split!=='TRAIN')assert.equal(r.training_weight,1);
    if(r.training_weight===0)duplicates++;
    decisions++;
   }
   assert.equal(rows.at(-1).target,'STOP');assert.equal(localErasures,2);erasures+=localErasures;
  }
 }
 assert.equal(m.splits.TRAIN.zero_weight+m.splits.DEV.zero_weight+m.splits.HELDOUT.zero_weight,duplicates);
 return {episodes,decisions,pairs,erasures,duplicates,quick};
}
const group=base.family('TRAIN',0),p=v.choose(group[0],'TRAIN',0,0);
assert.ok(p);assert.equal(p.arms.length,2);assert.notEqual(p.arms[0].rank[0].action,p.arms[1].rank[0].action);
assert.deepEqual(v.restricted(p.arms[0].o),v.restricted(p.arms[1].o));
console.log('PASS strict two-hit aliased first state');
const root=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v29-'));
try{
 const a=v.generate(path.join(root,'a'),{quick:true}),b=v.generate(path.join(root,'b'),{quick:true});
 assert.deepEqual(a,b);
 for(const filename of Object.keys(a.shards))assert.deepEqual(fs.readFileSync(path.join(root,'a',filename)),fs.readFileSync(path.join(root,'b',filename)));
 console.log('PASS two independent byte-identical quick generations');
 console.log('PASS independent reconstruction and split checks',check(path.join(root,'a'),true));
}finally{fs.rmSync(root,{recursive:true,force:true});}
if(process.argv[2])console.log('PASS full dataset independent reconstruction',check(path.resolve(process.argv[2]),false));
