"use strict";
const assert=require("node:assert/strict");
const crypto=require("node:crypto");
const {generateDataset,createHost,observe,apply,shortHistory,worldFor,route,tile}=require("./relay_memory_v13.js");
const a=generateDataset(4),b=generateDataset(4);
assert.equal(crypto.createHash("sha256").update(JSON.stringify(a)).digest("hex"),
 crypto.createHash("sha256").update(JSON.stringify(b)).digest("hex"),"repeatability");
assert.deepEqual(Object.fromEntries(Object.entries(a.summary.counts).map(([k,v])=>[k,[v.episodes,v.pairs,v.world_groups,v.decisions]])),
 {TRAIN:[256,128,64,2304],DEV:[64,32,16,576],HELDOUT:[96,48,24,864]});
const topologies=new Map(),worlds=new Map(),pairs=new Map();
let moves=0,illegal=0,targets=0,strictNegs=0,closedDenied=0;
for(const split of ["TRAIN","DEV","HELDOUT"]){
 for(const ep of a.episodes[split]){
  assert.equal(ep.native_receipt,false);
  assert.equal(topologies.get(ep.topology_group)??split,split);topologies.set(ep.topology_group,split);
  assert.equal(worlds.get(ep.group_id)??split,split);worlds.set(ep.group_id,split);
  const host=createHost(ep),initial=observe(host),historyLength=initial.history.executed_positions.length;
  assert.deepEqual(initial,ep.initial_model_input);
  assert(historyLength>200);
  assert(initial.history.events.filter(e=>e.step<historyLength-64).length>=2);
  assert.deepEqual(initial.local_material_7x7,Array.from({length:7},(_,y)=>initial.local_material_7x7[y]));
  assert.equal(apply(host,"MOVE_DIAGONAL").ok,false);illegal++;
  assert.equal(host.applied,0);
  const short=shortHistory(initial);
  const last=shortHistory(initial,1);
  const p=pairs.get(ep.pair_id),first=ep.expert.actions[0];
  if(!p)pairs.set(ep.pair_id,{short,last,first});
  else{
   assert.deepEqual(short,p.short,"last64 must alias");
   assert.deepEqual(last,p.last,"last1 must alias");
   assert.notEqual(first,p.first,"paired actions must oppose");
  }
  assert(initial.legal_actions.includes(first));
  const w=worldFor(ep.privileged_host.grammar,ep.privileged_host.span,ep.privileged_host.height,ep.privileged_host.seed);
  const blocked=[w.cx+(ep.privileged_host.closed_gate_side==="WEST"?-w.span:w.span),w.by-3];
  assert.equal(tile(w,blocked,ep.privileged_host.closed_gate_side),"X");
  assert(!route(w,initial.position,ep.privileged_host.closed_gate_side).some(p=>p.join(",")===blocked.join(",")));
  closedDenied++;
  for(const action of ep.expert.actions){assert.equal(apply(host,action).ok,true);moves++;}
  assert(host.done);assert.equal(host.applied,ep.expert.optimal_moves);
  assert.equal(host.illegal_attempts,1);
  assert.deepEqual(host.position,ep.initial_model_input.goal);
  assert.equal(apply(host,"MOVE_NORTH").reason,"already_finished");
 }
 for(const row of a.decisions[split]){
  assert(row.model_input.legal_actions.includes(row.target));targets++;
  for(const h of row.hard_negatives.provably_longer_legal){
   assert(row.model_input.legal_actions.includes(h));assert.notEqual(row.target,h);strictNegs++;
  }
  assert.equal(row.native_receipt,false);
 }
}
assert.equal(pairs.size,208);assert.equal(moves,a.summary.applied_oracle_moves);
assert(strictNegs>1000);
console.log("PASS v13: partitions/counterfactual last64, both relays, BFS replays, protected gates, illegal moves, strict hard negatives; "+JSON.stringify({episodes:illegal,pairs:pairs.size,moves,decisions:targets,strictNegs,closedDenied}));
