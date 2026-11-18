'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const v=require('./remote_error_alias_v28.js'),base=require('./causal_gate_v26.js');
const sha=base.sha;
function test(label,fn){fn();console.log('PASS '+label);}
function readJSONL(f){return fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);}
test('wrong target accepts paint, makes remote error invisible in latest64',()=>{
 const c=base.family('TRAIN',0)[0],p=v.pairFor(c,'TRAIN',0,0);
 assert.ok(p);const [a,b]=p.cases;
 assert.notEqual(a.paint_cell,b.paint_cell);assert.notEqual(a.rank[0].action,b.rank[0].action);
 assert.ok(p.regrets.every(x=>x>=2));assert.ok(a.remote_distance>=3&&b.remote_distance>=3);
 assert.ok(a.aged_out&&b.aged_out);assert.equal(a.o.painted_correct,0);assert.equal(b.o.painted_correct,0);
 assert.deepEqual(v.restricted(a.o),v.restricted(b.o));assert.notEqual(sha(a.o),sha(b.o));
 assert.ok(a.o.history.events.some(x=>x.action?.startsWith('PAINT')));
 assert.ok(b.o.history.events.some(x=>x.action?.startsWith('PAINT')));
 const ca=a.o.history.events.find(x=>x.action?.startsWith('PAINT')),cb=b.o.history.events.find(x=>x.action?.startsWith('PAINT'));
 assert.notEqual(ca.at,cb.at);
});
function inspect(dir,quick){
 const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'))),bySplit=new Map(),worldMap=new Map();
 let total=0,pairs=0,erasures=0,zeroWeight=0;
 for(const [split,q] of Object.entries(m.splits)){
  const episodes=readJSONL(path.join(dir,split+'_episodes.jsonl')),decisions=readJSONL(path.join(dir,split+'_decisions.jsonl'));
  assert.equal(q.pairs*2,episodes.length);assert.equal(episodes.length,q.episodes);assert.equal(decisions.length,q.decisions);assert.ok(q.pairs>=4);
  assert.equal(q.min_regret>=2,true);assert.ok(q.min_remote_distance>=3);
  for(const kind of ['episodes','decisions']){const name=split+'_'+kind+'.jsonl',b=fs.readFileSync(path.join(dir,name));assert.equal(sha(b),m.shards[name].sha256);assert.equal(b.byteLength,m.shards[name].bytes);}
  const episodeRows=new Map();for(const d of decisions){if(!episodeRows.has(d.episode_id))episodeRows.set(d.episode_id,[]);episodeRows.get(d.episode_id).push(d);}
  const groups=new Map();for(const ep of episodes){if(!groups.has(ep.pair_id))groups.set(ep.pair_id,[]);groups.get(ep.pair_id).push(ep);}
  assert.equal(groups.size,q.pairs);
  for(const [id,es] of groups){assert.equal(es.length,2,id);const [a,b]=es;
   assert.notEqual(a.hidden_intervention.target_cell,b.hidden_intervention.target_cell);
   assert.notEqual(a.first_action,b.first_action);
   assert.equal(a.restricted_alias_sha256,b.restricted_alias_sha256);
   assert.notEqual(a.initial_complete_observation_sha256,b.initial_complete_observation_sha256);
   assert.ok(a.first_action_regret>=2&&b.first_action_regret>=2);
   assert.equal(a.difficulty.scripted_prelabel_actions,b.difficulty.scripted_prelabel_actions);
   assert.equal(a.difficulty.older_than_64,true);assert.equal(b.difficulty.older_than_64,true);
   pairs++;
  }
  for(const ep of episodes){assert.equal(ep.terminal_success,true);assert.equal(ep.erases,1);erasures++;
   assert.ok(ep.difficulty.remote_distance>=3);
   const prior=worldMap.get(ep.world_sha256);if(prior)assert.equal(prior,split);worldMap.set(ep.world_sha256,split);
   const rows=episodeRows.get(ep.episode_id);assert.equal(rows.length,ep.steps);assert.equal(rows[0].target,ep.first_action);
   let history=structuredClone(ep.initial_history),erase=0;
   assert.equal(history.events.slice(-64).some(x=>x.action?.startsWith('PAINT')||x.kind==='GATE_TRANSITION'),false);
   for(let i=0;i<rows.length;i++){
    const row=rows[i];assert.equal(row.step,i);history.events.push(...row.events_added);history.positions.push(...row.positions_added);
    const observation={};for(const key of ep.observation_keys)observation[key]=key==='history'?structuredClone(history):row.state[key];
    assert.equal(sha(observation),row.model_input_sha256,ep.episode_id+'/'+i);
    assert.ok(observation.legal_actions.includes(row.target));assert.equal(row.receipt.accepted,true);
    if(row.target==='ERASE')erase++;
    const owner=bySplit.get(row.model_input_sha256);if(owner)assert.equal(owner.split,split);bySplit.set(row.model_input_sha256,{split,target:row.target});
    if(split!=='TRAIN')assert.equal(row.training_weight,1);
    if(row.training_weight===0)zeroWeight++;
    total++;
   }
   assert.equal(rows.at(-1).target,'STOP');assert.equal(erase,1);
  }
 }
 assert.ok(total>0);assert.equal(erasures,m.splits.TRAIN.episodes+m.splits.DEV.episodes+m.splits.HELDOUT.episodes);
 return {total,pairs,erasures,zeroWeight,quick};
}
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v28-'));
try{
 const a=v.generate(path.join(tmp,'A'),{quick:true}),b=v.generate(path.join(tmp,'B'),{quick:true});
 test('byte-identical quick independent generations',()=>assert.deepEqual(a,b));
 test('compact replay, paired aliases, regret, split leakage, terminal recovery',()=>console.log(inspect(path.join(tmp,'A'),true)));
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
if(process.argv[2])test('complete corpus independent reconstruction',()=>console.log(inspect(path.resolve(process.argv[2]),false)));
