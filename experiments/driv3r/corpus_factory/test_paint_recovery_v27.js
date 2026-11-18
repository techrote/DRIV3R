'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const v=require('./paint_recovery_v27.js'),base=require('./causal_gate_v26.js');
const sha=base.sha;
function test(name,fn){fn();console.log('PASS '+name);}
function readJsonl(p){return fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);}
function inspect(root,quick){
 const m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'))),owners=new Map(),worlds=new Map();
 let steps=0,erases=0,bad=0,consequential=0,weighted=0;
 for(const [split,s] of Object.entries(m.splits)){
  assert.equal(s.episodes,s.groups*8*v.MODES.length);assert.equal(s.corrective_erases,s.groups*16);
  assert.equal(s.accepted_bad_paints,s.groups*16);assert.equal(s.accepted_wrong_branches,s.groups*8);
  for(const mode of v.MODES)assert.equal(s.modes[mode],s.groups*8);
  if(split!=='TRAIN')assert.equal(s.zero_weight,0);
  const eps=readJsonl(path.join(root,split+'_episodes.jsonl'));
  const rows=readJsonl(path.join(root,split+'_decisions.jsonl'));
  assert.equal(eps.length,s.episodes);assert.equal(rows.length,s.decisions);
  for(const kind of ['episodes','decisions']){const f=split+'_'+kind+'.jsonl',buf=fs.readFileSync(path.join(root,f));
   assert.equal(sha(buf),m.shards[f].sha256);assert.equal(buf.byteLength,m.shards[f].bytes);}
  const byId=new Map();for(const row of rows){if(!byId.has(row.episode_id))byId.set(row.episode_id,[]);byId.get(row.episode_id).push(row);}
  for(const ep of eps){
   assert.equal(ep.synthetic_host,true);assert.equal(ep.terminal_success,true);
   const owner=worlds.get(ep.world_sha256);if(owner)assert.equal(owner,split);worlds.set(ep.world_sha256,split);
   const wrong=ep.episode_mode!=='correct-paint-control';assert.equal(ep.accepted_bad_paint,wrong?1:0);
   assert.equal(ep.corrective_erase,wrong?1:0);bad+=ep.accepted_bad_paint;erases+=ep.corrective_erase;
   if(ep.causal_frozen_gate_harmful+ep.causal_frozen_gate_illegal>0)consequential++;
   const firstPaint=ep.initial_history.events.filter(x=>x.kind==='ACTION_ACK'&&x.action.startsWith('PAINT'));
   assert.equal(firstPaint.length,1);assert.equal(ep.initial_history.events.filter(x=>x.kind==='GATE_TRANSITION').length,1);
   assert.equal(ep.intervention.accepted_wrong_paint,wrong);
   assert.ok(ep.difficulty.remote_start_distance>=3);
   assert.ok(ep.difficulty.post_paint_distractor_moves>=72);
   assert.equal(ep.initial_history.events.slice(-64).filter(x=>x.action?.startsWith('PAINT')||x.kind==='GATE_TRANSITION').length,0);
   const records=byId.get(ep.episode_id);assert.equal(records.length,ep.steps);
   let history=structuredClone(ep.initial_history),allErase=0;
   for(const [i,r] of records.entries()){
    assert.equal(r.step,i);assert.equal(r.episode_id,ep.episode_id);
    history.events.push(...r.events_added);history.positions.push(...r.positions_added);
    const obs={};for(const key of ep.observation_keys)obs[key]=key==='history'?structuredClone(history):r.state[key];
    assert.equal(sha(obs),r.model_input_sha256,ep.episode_id+':'+i);
    assert.ok(obs.legal_actions.includes(r.target),ep.episode_id+':'+i+':'+r.target);
    assert.equal(r.receipt.action,r.target);assert.equal(r.receipt.accepted,true);
    if(r.target==='ERASE')allErase++;
    assert.ok(r.training_weight===0||r.training_weight===1);
    if(r.training_weight===1)weighted++;
    const old=owners.get(r.model_input_sha256);if(old)assert.equal(old.split,split);
    owners.set(r.model_input_sha256,{split,target:r.target});steps++;
   }
   assert.equal(allErase,ep.corrective_erase);
   assert.equal(records.at(-1).target,'STOP');
  }
 }
 assert.equal(bad,quick?48:256);assert.equal(erases,bad);
 assert.ok(weighted>0);if(!quick)assert.ok(consequential>=10);
 return {steps,bad,erases,consequential,weighted};
}
test('host-accepted wrong paint toggles gate and permits erase',()=>{
 const c=base.family('TRAIN',0)[0],mode='wrong-material-recovery',p=v.prepare(c,'TRAIN',0,0,mode);
 assert.equal(base.correct(c.w,p.h.status).wrong,1);assert.equal(p.h.latched,1);
 assert.equal(p.h.history.events.filter(x=>x.kind==='GATE_TRANSITION').length,1);
 assert.ok(!base.observe(p.h).legal_actions.includes('ERASE'),'error must be remote');
 assert.equal(p.h.history.events.slice(-64).filter(x=>x.action?.startsWith('PAINT')||x.kind==='GATE_TRANSITION').length,0);
});
test('matched correct and incorrect material starts differ in a public receipt',()=>{
 const c=base.family('TRAIN',0)[0],a=v.prepare(c,'TRAIN',0,0,'correct-paint-control'),b=v.prepare(c,'TRAIN',0,0,'wrong-material-recovery');
 assert.equal(a.chosen.cell,b.chosen.cell);assert.equal(a.actions.length,b.actions.length);
 const i=a.actions.findIndex(x=>x.startsWith('PAINT'));assert.ok(i>=0);
 assert.deepEqual(a.actions.slice(0,i),b.actions.slice(0,i));assert.notEqual(a.actions[i],b.actions[i]);
 assert.deepEqual(a.actions.slice(i+1),b.actions.slice(i+1));
 const x=base.observe(a.h),y=base.observe(b.h);
 assert.deepEqual(x.local_view,y.local_view);
 const withoutCount=e=>e.map(({painted_correct,...rest})=>rest);
 assert.deepEqual(withoutCount(x.history.events.slice(-64)),withoutCount(y.history.events.slice(-64)));
 assert.notEqual(x.painted_correct,y.painted_correct); // public summary discloses *whether*, not *where*

});
const root=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v27-'));
try{
 const a=v.generate(path.join(root,'A'),{quick:true}),b=v.generate(path.join(root,'B'),{quick:true});
 assert.deepEqual(a,b);test('two byte-identical quick generations',()=>{
  for(const f of Object.keys(a.shards))assert.equal(sha(fs.readFileSync(path.join(root,'A',f))),sha(fs.readFileSync(path.join(root,'B',f))));
 });
 test('quick compact replay, no label/partition leakage, terminal correction',()=>inspect(path.join(root,'A'),true));
}finally{fs.rmSync(root,{recursive:true,force:true});}
if(process.argv[2])test('complete corpus replay, all hashes and correction proof',()=>console.log(inspect(path.resolve(process.argv[2]),false)));
