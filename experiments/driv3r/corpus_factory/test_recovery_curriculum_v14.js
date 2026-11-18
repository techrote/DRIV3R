"use strict";
const assert=require("node:assert/strict"),crypto=require("node:crypto"),fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const v13=require("./relay_memory_v13.js");
const {generateDataset,createHost,observe,apply,writeDataset}=require("./recovery_curriculum_v14.js");
const a=generateDataset(2),b=generateDataset(2);
assert.equal(crypto.createHash("sha256").update(JSON.stringify(a)).digest("hex"),
 crypto.createHash("sha256").update(JSON.stringify(b)).digest("hex"),"deterministic in-process");
assert.deepEqual(Object.fromEntries(Object.entries(a.summary.counts).map(([s,v])=>[s,[v.episodes,v.world_groups,v.topologies]])),
 {TRAIN:[384,32,16],DEV:[96,8,4],HELDOUT:[144,12,6]});
const worldSplit=new Map(),topologySplit=new Map(),modes=new Map(),pairModes=new Map();
let recoveryMoves=0,acceptedPrefix=0,rejectedPrefix=0,rows=0,strict=0,replayed=0,goalRetargeted=0;
for(const split of ["TRAIN","DEV","HELDOUT"]){
 for(const ep of a.episodes[split]){
  assert.equal(ep.native_receipt,false);assert.equal(ep.split,split);
  assert.equal(worldSplit.get(ep.group_id)??split,split);worldSplit.set(ep.group_id,split);
  assert.equal(topologySplit.get(ep.topology_group)??split,split);topologySplit.set(ep.topology_group,split);
  const modeKey=ep.group_id+"/"+ep.intervention;
  modes.set(ep.intervention,(modes.get(ep.intervention)||0)+1);
  pairModes.set(ep.pair_id+"/"+ep.intervention,(pairModes.get(ep.pair_id+"/"+ep.intervention)||0)+1);
  const host=createHost(ep),first=observe(host);assert.deepEqual(first,ep.first_model_input);
  assert.equal(ep.parent.privileged_host.closed_gate_side===host.closed,ep.intervention!=="shutter_reversal");
  if(ep.intervention==="goal_retarget"){
   assert.notDeepEqual(first.goal,ep.parent.initial_model_input.goal);goalRetargeted++;
   assert(first.history.events.some(e=>e.kind==="objective_retarget_receipt"));
  }
  if(ep.intervention==="shutter_reversal")assert(first.history.events.some(e=>e.kind==="remote_shutter_change_receipt"));
  if(ep.intervention==="closed_gate_probe"){
   assert.equal(host.illegal_attempts,1);
   assert(first.history.events.some(e=>e.kind==="host_rejected_move"));
   assert(!first.legal_actions.includes("MOVE_SOUTH"));
  }
  assert(first.history.executed_positions.length>200);
  assert.equal(apply(host,"MOVE_DIAGONAL").ok,false);
  assert.equal(host.applied,ep.prefix_moves); // refused move did not advance
  for(const action of ep.expert_recovery.actions){assert.equal(apply(host,action).ok,true);recoveryMoves++;}
  assert(host.done);assert.deepEqual(host.position,first.goal);replayed++;
  assert.equal(host.illegal_attempts,ep.rejected_prefix_actions+1);
  acceptedPrefix+=ep.prefix_moves;rejectedPrefix+=ep.rejected_prefix_actions;
 }
 for(const row of a.decisions[split]){
  assert.equal(row.native_receipt,false);assert.equal(row.split,split);
  assert(row.model_input.legal_actions.includes(row.target));rows++;
  const collapsed=v13.shortHistory(row.model_input,64);
  assert.deepEqual(row.controls.last64_only,collapsed);
  assert(!("privileged_host" in row.model_input));
  for(const action of row.hard_negatives.strictly_longer_legal){
   assert(row.model_input.legal_actions.includes(action));assert.notEqual(action,row.target);strict++;
  }
  for(const action of row.hard_negatives.illegal_or_blocked)assert(!row.model_input.legal_actions.includes(action));
  assert(!row.annotation.equally_optimal_other_actions.includes(row.target));
 }
}
for(const count of pairModes.values())assert.equal(count,2,"all paired counterfactuals grouped and replicated");
assert.deepEqual(Object.fromEntries(modes),{closed_gate_probe:208,shutter_reversal:208,goal_retarget:208});
assert.equal(recoveryMoves,a.summary.oracle_recovery_moves);assert.equal(acceptedPrefix,a.summary.accepted_intervention_moves);
assert.equal(rejectedPrefix,a.summary.rejected_prefix_moves);assert.equal(strict,a.summary.strict_hard_negative_actions);
assert.equal(rows,Object.values(a.summary.counts).reduce((n,c)=>n+c.decisions,0));
assert.equal(goalRetargeted,208);
const folder=fs.mkdtempSync(path.join(os.tmpdir(),"driv3r-v14-test-"));
const output=path.join(folder,"corpus");
try{
 const manifest=writeDataset(output,1);for(const [name,record] of Object.entries(manifest.shards)){
  const bytes=fs.readFileSync(path.join(output,name));
  assert.equal(bytes.length,record.bytes);
  assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"),record.sha256);
  assert.equal(bytes.toString().trimEnd().split("\n").length,record.rows);
 }
 assert.throws(()=>writeDataset(output,1),/overwrite/);
} finally {fs.rmSync(folder,{force:true,recursive:true});}
console.log("PASS v14: split topology/world isolation, 3 action-intervention modes, strict replay, counterfactual retention, legal/protected safety, hashes and no overwrite",JSON.stringify({episodes:replayed,rows,acceptedPrefix,recoveryMoves,rejectedPrefix,strict,goalRetargeted}));
