"use strict";
const assert=require("node:assert/strict");
const {generateDataset}=require("./stateful_challenge_v11.js");
const a=generateDataset(2),b=generateDataset(2);
assert.deepEqual(a,b);
assert.deepEqual(Object.fromEntries(Object.entries(a.shards).map(([k,v])=>[k,v.length])),
  {TRAIN:192,DEV:72,HELDOUT:72});
const groups=new Map(),topologies=new Map(),signatures=new Map();
for(const [split,rows] of Object.entries(a.shards)){
  const pairs=new Map();
  for(const r of rows){
    assert.equal(r.native_receipt,false);
    assert.equal(r.evidence.protected_entries,0);
    assert(r.model_input.legal_actions.includes(r.target));
    assert(r.hard_negatives.legal_but_wrong.every(x=>x!==r.target&&r.model_input.legal_actions.includes(x)));
    assert(!r.hard_negatives.legal_but_wrong.includes(r.target));
    if(groups.has(r.group_id))assert.equal(groups.get(r.group_id),split);else groups.set(r.group_id,split);
    if(topologies.has(r.topology_group))assert.equal(topologies.get(r.topology_group),split);
    else topologies.set(r.topology_group,split);
    if(!pairs.has(r.pair_id))pairs.set(r.pair_id,[]);
    pairs.get(r.pair_id).push(r);
  }
  for(const [pair,rows] of pairs){
    assert.equal(rows.length,2,pair);
    assert.notEqual(rows[0].target,rows[1].target);
    assert.deepEqual(rows[0].controls.last32_only,rows[1].controls.last32_only);
    assert.deepEqual({...rows[0].model_input,history:null},{...rows[1].model_input,history:null});
    assert(rows[0].provenance.host_applied_movement_steps>40);
    if(rows[0].task==="periodic_motif"){
      assert.equal(rows[0].model_input.history.paint_receipts.length,73);
      assert.equal(rows[0].controls.last32_only.history.paint_receipts.length,32);
    }
  }
  signatures.set(split,[...pairs.keys()].length);
}
assert.deepEqual(Object.fromEntries(signatures),{TRAIN:96,DEV:36,HELDOUT:36});
assert.equal(a.summary.host_applied_movement_steps>10000,true);
for(const rows of Object.values(a.shards))for(const r of rows){
 assert.equal(r.model_input.local_material_7x7[3][2],"s");
 assert.equal(r.model_input.local_material_7x7[3][4],"m");
 if(r.task==="delayed_material_law"){
  const e=r.model_input.history.history_events[0];
  assert.deepEqual(e.traversal_costs,e.law==="DRY"?{SAND:1,METAL:7}:{SAND:7,METAL:1});
 }
}
console.log("PASS v11: paired counterfactuals, last32 alias, route legality, hashes pending CLI generation, split grammar isolation");
