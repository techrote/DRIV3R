'use strict';
const assert=require('node:assert/strict');
const {layout,grid,legal,shortest,assertTrace,dynamicPrelude,makeRow,templates,produce}=require('./dynamic_challenge_v10.js');
const w=layout(11,18,37);
assert.deepEqual(w.hub,[20,19]);
assert.deepEqual(legal(w,w.hub),['MOVE_NORTH','MOVE_EAST','MOVE_WEST']);
for(const arm of ['WEST','EAST']){
 const d=dynamicPrelude(w,arm);assert(d.path.length>50);
 assert.equal(assertTrace(w,d.path,d.failedAt+1,d.gate),d.path.length-1);
 assert.deepEqual(d.path.at(-1),w.hub);
 assert.deepEqual(shortest(w,w.hub,w.merge,new Set([d.gate.join(',')]))[0],w.hub);
}
for(const task of ['shutter_recovery','goal_retarget','motif_echo']){
 const a=makeRow(w,task,'WEST','TRAIN'),b=makeRow(w,task,'EAST','TRAIN');
 assert.notEqual(a.target,b.target);
 assert.deepEqual({...a.model_input,context:undefined},{...b.model_input,context:undefined});
 assert.equal(a.provenance.host_applied_moves,a.evidence.prelude_positions.length-1);
 assert(a.legal_actions.includes(a.target));assert(b.legal_actions.includes(b.target));
 assert(a.hard_negatives.legal_but_wrong.every(x=>a.legal_actions.includes(x)&&x!==a.target));
 assert.equal(a.evidence.protected_entries,0);
 if(task==='shutter_recovery'){
  assert.deepEqual(a.controls.last12_only,b.controls.last12_only);
  assert.deepEqual(a.model_input.last12_positions,b.model_input.last12_positions);
  assert.equal(a.evidence.oracle_route.at(-1).join(','),w.merge.join(','));
 }
 if(task==='goal_retarget')assert.notDeepEqual(a.model_input.context.current_goal,b.model_input.context.current_goal);
 if(task==='motif_echo')assert.equal(a.model_input.context.remote_paint_receipts.length,24);
}
const x=produce(2),y=produce(2);
assert.deepEqual(x,y,'byte-stable deterministic generation');
assert.deepEqual(Object.fromEntries(Object.entries(x.shards).map(([p,rows])=>[p,rows.length])),{TRAIN:216,DEV:72,HELDOUT:72});
const splitOfShape=new Map(),groups=new Map(),views=[];
for(const [split,rows] of Object.entries(x.shards)){
 for(const row of rows){
  if(splitOfShape.has(row.topology_group))assert.equal(splitOfShape.get(row.topology_group),split);
  else splitOfShape.set(row.topology_group,split);
  if(groups.has(row.group_id))assert.equal(groups.get(row.group_id),split);
  else groups.set(row.group_id,split);
  assert(row.legal_actions.includes(row.target));
  assert(!row.hard_negatives.legal_but_wrong.includes(row.target));
  assert.equal(row.native_receipt,false);
  assert.equal(row.evidence.protected_entries,0);
  if(row.task==='shutter_recovery'){
   const p=row.evidence.prelude_positions;
   assert.deepEqual(row.model_input.last12_positions,p.slice(-12));
   assert.equal(row.evidence.shutter_event.kind,'remote_shutter_closed');
  }
 }
}
for(const rows of Object.values(x.shards))for(const [id,pair] of Object.entries(rows.reduce((acc,row)=>{(acc[row.pair_id]??=[]).push(row);return acc;},{}))){
 assert.equal(pair.length,2,id);assert.notEqual(pair[0].target,pair[1].target);
 if(pair[0].task==='shutter_recovery')assert.deepEqual(pair[0].controls.last12_only,pair[1].controls.last12_only);
}
const shapes=templates();assert.equal(shapes.TRAIN.length,18);assert.equal(shapes.DEV.length,6);assert.equal(shapes.HELDOUT.length,6);
console.log('PASS v10: genuine legal synthetic preludes, protected safety, paired alias, counterfactual targets, split integrity, deterministic output, hard negatives');
