'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const v=require('./causal_gate_v26.js');
const eq=(a,b)=>assert.deepEqual(a,b);
function test(label,fn){fn();console.log('PASS '+label);}
function inspect(root,quick){const m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json')));const inputSplits=new Map(),worldSplits=new Map();let total=0,causal=0;
 for(const [split,s] of Object.entries(m.splits)){
  const epPath=path.join(root,split+'_episodes.jsonl'),decPath=path.join(root,split+'_decisions.jsonl'),episodes=fs.readFileSync(epPath,'utf8').trim().split('\n').map(JSON.parse),rows=fs.readFileSync(decPath,'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(episodes.length,s.episodes);assert.equal(rows.length,s.decisions);assert.equal(episodes.length,s.groups*16);
  for(const kind of ['episodes','decisions']){let filename=`${split}_${kind}.jsonl`,raw=fs.readFileSync(path.join(root,filename));assert.equal(v.sha(raw),m.shards[filename].sha256);assert.equal(raw.length,m.shards[filename].bytes);}
  eq(s.first_action,{BACKTRACK:s.groups*4,MOVE_N:s.groups*4});eq(s.gate_initial,{open:s.groups*4,closed:s.groups*4});
  const byId=new Map();for(const r of rows){if(!byId.has(r.episode_id))byId.set(r.episode_id,[]);byId.get(r.episode_id).push(r);}
  for(const ep of episodes){assert.equal(ep.synthetic_host,true);assert.equal(ep.terminal_success,true);assert.equal(ep.gate_changes,1);assert.ok(ep.first_action_regret>=2);
   assert.equal(ep.steps,byId.get(ep.episode_id)?.length);assert.equal(ep.episode_mode==='wrong-first-recovery'?ep.accepted_predecision_interventions:0,ep.accepted_predecision_interventions);
   const owner=worldSplits.get(ep.world_sha256);if(owner)assert.equal(owner,split);worldSplits.set(ep.world_sha256,split);
   if(ep.episode_mode==='clean'&&ep.post_gate_policy_differences_vs_frozen_gate>0)causal++;
   const history=structuredClone(ep.initial_history),rs=byId.get(ep.episode_id),seenGates=[];
   for(let i=0;i<rs.length;i++){
    const r=rs[i];assert.equal(r.step,i);assert.equal(r.schema,'driv3r/corpus_factory/causal-gate-v26/decision');
    history.events.push(...r.events_added);history.positions.push(...r.positions_added);
    const observation={};for(const k of ep.observation_keys)observation[k]=k==='history'?structuredClone(history):r.state[k];
    assert.equal(v.sha(observation),r.model_input_sha256,ep.episode_id+':'+i);assert.ok(observation.legal_actions.includes(r.target));assert.equal(r.receipt.accepted,true);
    assert.equal(r.receipt.action,r.target);assert.equal(r.receipt.kind,'ACTION_ACK');
    const prior=inputSplits.get(r.model_input_sha256);if(prior)assert.equal(prior,split);inputSplits.set(r.model_input_sha256,split);
    assert.ok(r.training_weight===0||r.training_weight===1);
    seenGates.push(...r.events_added.filter(e=>e.kind==='GATE_TRANSITION'));
   }
   // Transition after first accepted paint is received in following decision's incremental history.
   assert.equal(seenGates.length,1);assert.equal(seenGates[0].cause,'FIRST_ACCEPTED_PAINT');
  }
  total+=episodes.length;
  assert.equal(s.clean_episodes,s.groups*8);assert.equal(s.wrong_first_recovery_episodes,s.groups*8);
  assert.equal(s.gate_transitions,s.episodes);
 }
 assert.equal(total,quick?48:256);
 if(!quick)assert.ok(causal>0,'no causal reroutes anywhere');
 return m;
}
test('paint-triggered gate is host-legal and changes local traversability',()=>{
 const w=v.world('TRAIN',0,{blocked:false,upper_link:false,target_mask:21,phase:0,initial_open:true});
 const h=new v.Host(w,'TRAIN',0,0),[a,b]=w.gate;
 const first=v.options(w,[a,b,0,0]).map(x=>x.action),after=v.options(w,[a,b,0,1]).map(x=>x.action);
 assert.ok(first.includes('BACKTRACK'));assert.ok(!after.includes('BACKTRACK'));
 assert.equal(v.gateOpen(w,0),true);assert.equal(v.gateOpen(w,1),false);
 h.pos=w.nidx.get(w.target[0]);h.prev=w.adjacency[h.pos][0];
 assert.ok(v.apply(h,'PAINT_A').accepted);assert.equal(h.latched,1);assert.equal(h.history.events.at(-1).kind,'GATE_TRANSITION');
 assert.ok(v.apply(h,'INVALID').accepted===false);assert.equal(h.history.events.at(-1).kind,'REFUSED');
});
test('each family retains balanced counterfactual first labels and old-history alias',()=>{
 for(const [split,g] of [['TRAIN',0],['DEV',1],['HELDOUT',1]]){
  const cs=v.family(split,g),o=cs.map((c,i)=>v.observe(new v.Host(c.w,split,g,i)));
  assert.equal(cs.length,8);eq(cs.map(c=>c.first).sort(),Array(4).fill('BACKTRACK').concat(Array(4).fill('MOVE_N')));
  eq(cs.map(c=>c.config.initial_open).sort(),Array(4).fill(false).concat(Array(4).fill(true)));
  for(const x of o){eq(x.local_view,o[0].local_view);eq(x.legal_actions,o[0].legal_actions);eq(x.history.events.slice(-64),o[0].history.events.slice(-64));eq(x.history.positions.slice(-64),o[0].history.positions.slice(-64));}
 }
});
test('alternative accepted first action yields independently replayed optimal recovery suffix',()=>{
 for(const [split,g] of [['TRAIN',0],['DEV',1],['HELDOUT',1]])for(const c of v.family(split,g)){
  const h=new v.Host(c.w,split,g,0),wrong=c.first==='BACKTRACK'?'MOVE_N':'BACKTRACK';assert.ok(v.apply(h,wrong).accepted);
  let steps=1;while(!h.done){const s=[h.pos,h.prev,h.status,h.latched],r=c.p.rank(s)[0];assert.ok(Number.isFinite(r.cost));assert.ok(v.apply(h,r.action).accepted);steps++;assert.ok(steps<120);}
  assert.equal(steps,c.alternative);assert.ok(c.alternative-c.optimal>=2);
 }
});
test('wrong material can be erased and protected cells are not traversable',()=>{
 const c=v.family('TRAIN',0)[0],h=new v.Host(c.w,'TRAIN',0,0),t=c.w.target[0];h.pos=c.w.nidx.get(t);h.prev=c.w.adjacency[h.pos][0];
 const bad=c.w.recipes[0]==='A'?'PAINT_B':'PAINT_A';assert.ok(v.apply(h,bad).accepted);assert.equal(v.correct(c.w,h.status).wrong,1);
 assert.ok(v.apply(h,'ERASE').accepted);assert.equal(v.correct(c.w,h.status).wrong,0);
 const prot=h.history.events[0].protected_cells;for(const tile of prot)assert.ok(!c.w.nidx.has(tile));
});
const root=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v26-'));
try{
 const m0=v.generate(path.join(root,'quickA'),{quick:true}),m1=v.generate(path.join(root,'quickB'),{quick:true});eq(m0,m1);inspect(path.join(root,'quickA'),true);inspect(path.join(root,'quickB'),true);
 console.log('PASS two deterministic quick generations and 96 compact replays');
}finally{fs.rmSync(root,{recursive:true,force:true});}
if(process.argv[2]){inspect(path.resolve(process.argv[2]),false);console.log('PASS all 256 full-corpus episodes, SHA-256 shards, and compact decisions');}
