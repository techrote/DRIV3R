"use strict";
const assert=require("node:assert/strict");
const {generateRollouts,createHost,observe,apply,short32}=require("./replay_rollouts_v12.js");
const a=generateRollouts(2),b=generateRollouts(2);
assert.deepEqual(a,b,"determinism");
assert.deepEqual(Object.fromEntries(Object.entries(a.summary.counts).map(([s,c])=>[s,[c.episodes,c.pairs,c.decisions]])),
 {TRAIN:[128,64,896],DEV:[48,24,336],HELDOUT:[48,24,336]});
let moves=0,invalid=0;const seen=new Map(),starts=new Map();
for(const split of ["TRAIN","DEV","HELDOUT"]){
 for(const ep of a.episodes[split]){
  assert.equal(ep.native_receipt,false);
  assert.equal(seen.get(ep.topology_group)??split,split,"topology leakage");
  seen.set(ep.topology_group,split);
  const host=createHost(ep);
  assert.deepEqual(observe(host),ep.initial_model_input,"initial observation mismatch");
  assert.equal(apply(host,"MOVE_DIAGONAL").ok,false,"illegal accepted");
  assert.equal(host.applied,0);
  invalid++;
  const pair=starts.get(ep.pair_id),initial=short32(observe(host)),action=ep.expert.actions[0];
  if(pair){assert.deepEqual(initial,pair.initial);assert.notEqual(action,pair.action);}
  else starts.set(ep.pair_id,{initial,action});
  for(const action of ep.expert.actions){assert.equal(apply(host,action).ok,true);moves++;}
  assert.equal(host.done,true,"goal not reached");
  assert.equal(host.illegal_attempts,1);
  assert.equal(host.spent,ep.expert.total_traversal_cost);
 }
 for(const r of a.decisions[split]){
  assert(r.model_input.legal_actions.includes(r.target));
  assert(r.hard_negatives.legal_but_suboptimal.every(v=>v!==r.target&&r.model_input.legal_actions.includes(v)));
  assert.equal(r.native_receipt,false);
 }
}
assert.equal(moves,a.summary.applied_oracle_moves);
assert.equal(starts.size,112);
console.log("PASS v12: deterministic replay, grouped split, paired last32 controls, model/host isolation, illegal rejection, oracle cost and arrival; moves="+moves+" invalid="+invalid);
