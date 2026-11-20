#!/usr/bin/env node
'use strict';
// DATA ONLY: evaluator controls. Never trainer input, never a trained-model score.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const hash=x=>require('node:crypto').createHash('sha256').update(JSON.stringify(x)).digest('hex');
function ceiling(rows,fn){const bins=new Map();for(const x of rows){const key=hash(fn(x));if(!bins.has(key))bins.set(key,new Map());const m=bins.get(key);m.set(x.action,(m.get(x.action)||0)+1);}
 let total=0,correct=0,conflicts=0;for(const m of bins.values()){total+=[...m.values()].reduce((a,b)=>a+b,0);correct+=Math.max(...m.values());if(m.size>1)conflicts++;}
 return {correct,total,accuracy:correct/total,groups:bins.size,conflicting_groups:conflicts};}
function score(root){const output={schema:'driv3r/corpus_factory/material-law-controls-v81/1',synthetic_not_native_CyberSand:true,neural_model_evaluated:false,splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){const pre=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
 const certs=fs.readFileSync(path.join(root,pre+'_material_law_certificates_EVALUATOR_ONLY.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 const eps=fs.readFileSync(path.join(root,pre+'_material_law_stream.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 assert.equal(certs.length,eps.length);const rows=[];let modelHistoryParser=0;
 certs.forEach((c,i)=>{if(c.wrong_paint_recovered)return;const ep=eps[i].steps,gi=ep.findIndex(x=>x.role==='goal_memory'),row=ep[gi];assert.ok(gi>0);const past=ep.slice(0,gi);
  const witness=past[1].input.painted_material;
  let goal=null,lawEvent=null,latestGoal=-1,latestLaw=-1;
  for(const x of past)for(const r of x.input.reports){if(r.type==='REMOTE_OBJECTIVE'&&r.sample_tick>latestGoal){goal=r.material;latestGoal=r.sample_tick;}
   if(r.type==='REMOTE_LAW_EVENT'&&r.sample_tick>latestLaw){lawEvent=r.flipped;latestLaw=r.sample_tick;}}
  assert.ok(witness==='A'||witness==='B');assert.ok(goal==='A'||goal==='B');assert.equal(typeof lawEvent,'boolean');
  let predicted=goal;if((witness==='B')!==lawEvent)predicted=predicted==='A'?'B':'A';
  if(row.action==='PAINT_'+predicted)modelHistoryParser++;
  assert.equal(row.loss_weight,1);
  assert.equal(row.input.reports.length,0);assert.equal(row.input.painted_material,null);
  assert.ok(!('law' in row.input)&&!('goal_material' in row.input),'privileged state leaked');
  rows.push({action:row.action,current:row.input,last128:past.slice(-128).map(x=>({input:x.input,action:x.action})),goal,witness,lawEvent,world:c.world_index});
 });
 output.splits[split]={episodes:rows.length,history_parser_correct:modelHistoryParser,
   current_only:ceiling(rows,x=>x.current),last128:ceiling(rows,x=>({current:x.current,past:x.last128})),
   goal_only:ceiling(rows,x=>({world:x.world,goal:x.goal})),
   physical_probe_plus_goal:ceiling(rows,x=>({world:x.world,witness:x.witness,goal:x.goal})),
   remote_law_receipt_plus_goal:ceiling(rows,x=>({world:x.world,flip:x.lawEvent,goal:x.goal})),
   complete_observable_history_factors:ceiling(rows,x=>({world:x.world,initial_witness:x.witness,flip:x.lawEvent,goal:x.goal}))};
 assert.equal(modelHistoryParser,rows.length,'observable history parser failed');
 assert.equal(output.splits[split].current_only.accuracy,.5);
 assert.equal(output.splits[split].last128.accuracy,.5);
 assert.equal(output.splits[split].physical_probe_plus_goal.accuracy,.5);
 assert.equal(output.splits[split].remote_law_receipt_plus_goal.accuracy,.5);
 assert.equal(output.splits[split].complete_observable_history_factors.accuracy,1);
 }return output;}
if(require.main===module){assert.ok(process.argv[2],'usage node score_material_law_controls_v81.js CORPUS');console.log(JSON.stringify(score(path.resolve(process.argv[2])),null,2));}
module.exports={score};
