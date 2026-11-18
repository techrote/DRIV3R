'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const v=require('./outcome_forks_v25.js');
const eq=(a,b)=>assert.deepEqual(a,b);
function test(name,f){f();console.log('PASS '+name);}
function reconstruct(h,rows){
 const history=JSON.parse(JSON.stringify(h.initial_history));let prev=-1;
 for(const r of rows){assert.equal(r.episode_id,h.episode_id);assert.equal(r.step,prev+1);prev=r.step;history.events.push(...r.events_added);history.positions.push(...r.positions_added);
  const model={};for(const k of h.observation_keys)model[k]=k==='history'?JSON.parse(JSON.stringify(history)):r.state[k];
  assert.equal(v.sha(model),r.model_input_sha256,'invalid reconstruction at '+r.episode_id+':'+r.step);
  assert.ok(model.legal_actions.includes(r.target),'illegal oracle');assert.equal(r.receipt.accepted,true);
 }return prev+1;
}
test('exhaustive terminal-suffix first action is unique, regret positive, geometry obstruction not proxy',()=>{
 for(const split of Object.keys(v.SPLITS)){
  const c=v.family(split,split==='TRAIN'?0:1);
  eq(c.map(x=>x.first).sort(),Array(4).fill('BACKTRACK').concat(Array(4).fill('MOVE_N')));
  for(const x of c){const r=x.p.rank([x.w.root,x.w.previous,0]);assert.equal(r[0].cost,x.optimal_cost);assert.ok(r[1].cost>r[0].cost);assert.ok(x.regret>=2);}
  for(const blocked of [false,true]){const a=c.filter(x=>x.config.blocked===blocked);eq(a.map(x=>x.first).sort(),['BACKTRACK','BACKTRACK','MOVE_N','MOVE_N']);}
 }
});
test('false suffix labels are rejected; complete sensor histories rather than 64-event suffix distinguish outcomes',()=>{
 const fam=v.family('TRAIN',0),base=fam[0];let witness=[];
 for(let i=0;i<fam.length;i++){let o=v.observe(new v.Host(fam[i].w,'TRAIN',0,i));witness.push(o);
  eq(o.legal_actions,witness[0].legal_actions);eq(o.local_view,witness[0].local_view);eq(o.history.events.slice(-64),witness[0].history.events.slice(-64));eq(o.history.positions.slice(-64),witness[0].history.positions.slice(-64));}
 assert.notDeepEqual(witness[0].history.events[0],witness[2].history.events[0]);
});
test('explicit alternative first-choice suffix cost matches independent replay, even for blocked spur',()=>{
 for(const [split,g] of [['TRAIN',0],['DEV',1],['HELDOUT',1]])for(const c of v.family(split,g)){
  const alt=c.first==='MOVE_N'?'BACKTRACK':'MOVE_N',h=new v.Host(c.w,split,g,0);let steps=0;
  assert.ok(v.apply(h,alt).accepted);steps++;
  while(!h.done){const r=c.p.rank([h.pos,h.prev,h.status])[0];assert.ok(Number.isFinite(r.cost));assert.ok(v.apply(h,r.action).accepted);steps++;if(steps>=100)throw Error('nonterminating suffix');}
  assert.equal(steps,c.alternative_cost);assert.equal(c.alternative_cost-c.optimal_cost,c.regret);
 }
});
test('wrong materials are accepted but can be repaired by erasure without touching protected cells',()=>{
 const c=v.family('TRAIN',0)[0],h=new v.Host(c.w,'TRAIN',0,0);
 const idx=c.w.nidx.get(c.w.target[0]);h.pos=idx;h.prev=c.w.adjacency[idx][0];
 const wrong=c.w.recipes[0]==='A'?'PAINT_B':'PAINT_A';assert.ok(v.apply(h,wrong).accepted);
 assert.equal(v.correctCount(c.w,h.status).wrong,1);assert.ok(v.observe(h).legal_actions.includes('ERASE'));
 assert.ok(v.apply(h,'ERASE').accepted);assert.equal(v.correctCount(c.w,h.status).wrong,0);
 const right=c.w.recipes[0]==='A'?'PAINT_A':'PAINT_B';assert.ok(v.apply(h,right).accepted);
 assert.equal(v.correctCount(c.w,h.status).count,1);
});
const root=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v25-'));
try{
 const results=[];for(let i=0;i<2;i++){
  const out=path.join(root,'out'+i),m=v.generate(out,{quick:true});results.push(m);
  for(const split of Object.keys(v.SPLITS)){
   const eps=fs.readFileSync(path.join(out,split+'_episodes.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
   const rows=fs.readFileSync(path.join(out,split+'_decisions.jsonl'),'utf8').trim().split('\n').map(JSON.parse),byId=new Map();
   for(const r of rows){if(!byId.has(r.episode_id))byId.set(r.episode_id,[]);byId.get(r.episode_id).push(r);}
   for(const ep of eps){assert.equal(reconstruct(ep,byId.get(ep.episode_id)),ep.steps);assert.ok(ep.terminal_success);assert.ok(ep.counterfactual_regret>=2);}
   assert.equal(eps.length,8);assert.equal(m.splits[split].first_action.BACKTRACK,4);
   assert.equal(m.shards[split+'_decisions.jsonl'].sha256,v.sha(fs.readFileSync(path.join(out,split+'_decisions.jsonl'))));
  }
 }eq(results[0],results[1]);console.log('PASS complete quick dataset deterministic twice and 24 episode lossless reconstructions');
}finally{fs.rmSync(root,{recursive:true,force:true});}
if(process.argv[2]){
 const root=path.resolve(process.argv[2]),m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
 for(const [split,s] of Object.entries(m.splits)){
  const eps=fs.readFileSync(path.join(root,split+'_episodes.jsonl'),'utf8').trim().split('\n').map(JSON.parse),rows=fs.readFileSync(path.join(root,split+'_decisions.jsonl'),'utf8').trim().split('\n').map(JSON.parse),byId=new Map();
  for(const r of rows){if(!byId.has(r.episode_id))byId.set(r.episode_id,[]);byId.get(r.episode_id).push(r);}
  for(const ep of eps){assert.equal(reconstruct(ep,byId.get(ep.episode_id)),ep.steps);assert.ok(ep.terminal_success);}
  assert.equal(rows.length,s.decisions);assert.equal(eps.length,s.episodes);
  for(const suffix of ['episodes','decisions']){const name=split+'_'+suffix+'.jsonl',buf=fs.readFileSync(path.join(root,name));assert.equal(v.sha(buf),m.shards[name].sha256);assert.equal(buf.length,m.shards[name].bytes);}
 }
 console.log('PASS complete provided corpus: all episodes, decision reconstructions, and shard SHA-256');
}
