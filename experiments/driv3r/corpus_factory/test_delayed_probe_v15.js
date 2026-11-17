"use strict";
const assert=require("node:assert/strict"),crypto=require("node:crypto"),fs=require("node:fs"),os=require("node:os"),path=require("node:path");
const v13=require("./relay_memory_v13.js");
const {generateDataset,createHost,observe,apply,eventControl,writeDataset}=require("./delayed_probe_v15.js");
const hash=x=>crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
const dataset=generateDataset(2),splits=["TRAIN","DEV","HELDOUT"];
assert.deepEqual(Object.fromEntries(splits.map(s=>[s,[dataset.summary.counts[s].episodes,dataset.summary.counts[s].decisions,
 dataset.summary.counts[s].quartets,dataset.summary.counts[s].world_groups]])),
 {TRAIN:[256,2304,64,32],DEV:[64,576,16,8],HELDOUT:[96,864,24,12]});
const worlds=new Map(),topologies=new Map(),quartets=new Map();let reached=0,executed=0,refused=0,rows=0,strict=0;
for(const split of splits){
 for(const ep of dataset.episodes[split]){
  assert.equal(ep.native_receipt,false);assert.equal(ep.split,split);
  assert.equal(worlds.get(ep.group_id)??split,split);worlds.set(ep.group_id,split);
  assert.equal(topologies.get(ep.topology_group)??split,split);topologies.set(ep.topology_group,split);
  if(!quartets.has(ep.quartet_id))quartets.set(ep.quartet_id,[]);quartets.get(ep.quartet_id).push(ep);
  const first=ep.first_model_input;
  assert.deepEqual(first.legal_actions,["MOVE_NORTH","MOVE_EAST","MOVE_WEST"]);
  assert(first.history.executed_positions.length>300);
  assert.deepEqual(v13.shortHistory(first,64).history.events,[],"causal receipt leaked into last64");
  assert.equal(first.history.events.filter(e=>e.kind==="remote_probe_receipt").length,1);
  assert.equal(first.history.events.filter(e=>e.kind==="remote_inversion_receipt").length,1);
  assert(!first.history.events.some(e=>e.kind.startsWith("remote_relay_")),"old synthetic relay leaked");
  const host=createHost(ep);assert.deepEqual(observe(host),first);
  assert.equal(host.illegal_attempts,ep.prefix_receipt.refused_moves);
  for(const action of ep.expert.actions){assert.equal(apply(host,action).ok,true);executed++;}
  assert.equal(host.done,true);assert.deepEqual(host.position,first.goal);
  reached++;refused+=ep.prefix_receipt.refused_moves;
 }
 for(const row of dataset.decisions[split]){
  assert.equal(row.native_receipt,false);
  assert(row.model_input.legal_actions.includes(row.target));
  assert.deepEqual(row.controls.last64_only,v13.shortHistory(row.model_input,64));
  assert.deepEqual(row.controls.event_history,eventControl(row.model_input,["remote_probe_receipt","remote_inversion_receipt"]));
  assert.deepEqual(row.controls.probe_only,eventControl(row.model_input,["remote_probe_receipt"]));
  assert.deepEqual(row.controls.inversion_only,eventControl(row.model_input,["remote_inversion_receipt"]));
  for(const action of row.hard_negatives.strictly_longer_legal){assert(row.model_input.legal_actions.includes(action));strict++;}
  for(const action of row.hard_negatives.blocked_or_illegal)assert(!row.model_input.legal_actions.includes(action));
  assert(!("privileged_host" in row.model_input));rows++;
 }
}
assert.equal(quartets.size,104);
for(const eps of quartets.values()){
 assert.equal(eps.length,4);
 assert.equal(new Set(eps.map(e=>hash(v13.shortHistory(e.first_model_input,64)))).size,1);
 assert.equal(new Set(eps.map(e=>hash(eventControl(e.first_model_input,["remote_probe_receipt","remote_inversion_receipt"])))).size,4);
 assert.equal(eps.filter(e=>e.expert.actions[0]==="MOVE_EAST").length,2);
 for(const kinds of [["remote_probe_receipt"],["remote_inversion_receipt"]]){
  const classes=new Map();for(const e of eps){const k=hash(eventControl(e.first_model_input,kinds));
   if(!classes.has(k))classes.set(k,[]);classes.get(k).push(e.expert.actions[0]);}
  assert.equal(classes.size,2);
  for(const labels of classes.values())assert.deepEqual(labels.sort(),["MOVE_EAST","MOVE_WEST"]);
 }
}
assert.equal(executed,dataset.summary.expert_moves);
assert.equal(refused,dataset.summary.host_refused_probe_moves);
assert.equal(strict,dataset.summary.strict_hard_negatives);
assert.equal(rows,Object.values(dataset.summary.counts).reduce((sum,x)=>sum+x.decisions,0));
assert.equal(reached,416);
const root=fs.mkdtempSync(path.join(os.tmpdir(),"driv3r-v15-test-")),dest=path.join(root,"data");
try{
 const m=writeDataset(dest,1);
 for(const [file,record] of Object.entries(m.shards)){
  const bytes=fs.readFileSync(path.join(dest,file));
  assert.equal(bytes.byteLength,record.bytes);
  assert.equal(crypto.createHash("sha256").update(bytes).digest("hex"),record.sha256);
  assert.equal(bytes.toString().trimEnd().split("\n").length,record.rows);
 }
 assert.throws(()=>writeDataset(dest,1),/overwrite/);
} finally {fs.rmSync(root,{force:true,recursive:true});}
console.log("PASS v15: 104 four-way counterfactuals, 64-step alias, single-receipt ablations, split isolation, safe closed-loop oracle replay, exact manifest and overwrite protection",JSON.stringify({episodes:reached,decisions:rows,executed,refused,strict}));
