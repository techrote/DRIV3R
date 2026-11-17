"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),os=require("node:os"),path=require("node:path"),crypto=require("node:crypto");
const v=require("./ordered_relay_v16.js"),v13=require("./relay_memory_v13.js");
const splits=["TRAIN","DEV","HELDOUT"],sha=x=>crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
const P="b49afca90c307d8126c8f4c1becba667b7ced6f1";
let n=0;function test(name,fn){fn();n++;console.log("PASS",name);}
test("56 possible permutations, 36 split-disjoint orders, balanced opposed labels",()=>{
 assert.equal(v.ORDERS.length,56);
 assert.equal(new Set(v.ORDERS.map(x=>x.join("/"))).size,56);
 assert.equal(new Set(Object.values(v.ORDERS_BY_SPLIT).flat().map(x=>x.join("/"))).size,36);
 for(const seq of Object.values(v.ORDERS_BY_SPLIT)){
  assert.equal(seq.length,12);
  assert.deepEqual(seq.map(v.closedFor).sort().reduce((a,b)=>(a[b]=(a[b]||0)+1,a),{}),{EAST:6,WEST:6});
 }
 assert.equal(v.applyGateCommand("WEST","TOGGLE"),"EAST");
 assert.notEqual(v.closedFor(["SET_EAST","SET_WEST","TOGGLE","TOGGLE"]),
  v.closedFor(["SET_WEST","SET_EAST","TOGGLE","TOGGLE"]));
});
const run=v.generateDataset(1);
test("TRAIN/DEV/HELDOUT world and topology groups disjoint",()=>{
 const allE=Object.values(run.episodes).flat(),ids=new Set(),worlds=new Map(),tops=new Map();
 assert.equal(allE.length,312);
 for(const ep of allE){assert(!ids.has(ep.episode_id));ids.add(ep.episode_id);
  assert.equal(ep.provenance.parent_source_git_blob,P);
  if(worlds.has(ep.group_id))assert.equal(worlds.get(ep.group_id),ep.split);else worlds.set(ep.group_id,ep.split);
  if(tops.has(ep.topology_group))assert.equal(tops.get(ep.topology_group),ep.split);else tops.set(ep.topology_group,ep.split);
  assert.equal(ep.native_receipt,false);
 }
 assert.deepEqual(splits.map(x=>run.summary.counts[x].episodes),[192,48,72]);
 assert.deepEqual(splits.map(x=>run.summary.counts[x].decisions),[1728,432,648]);
});
test("each family shares last64 / event-bag and opposes first actions",()=>{
 const families=new Map();for(const ep of Object.values(run.episodes).flat()){
  const a=families.get(ep.family_id)||[];a.push(ep);families.set(ep.family_id,a);
 }
 const proof=v.checkFamilies(families);assert.equal(proof.families,26);
 assert.deepEqual(Object.fromEntries(Object.entries(proof.controls).map(([split,c])=>
  [split,Object.fromEntries(Object.entries(c).map(([k,val])=>[k,val.correct]))])),{
   TRAIN:{bag:6,first:8,last:6,last2:7},DEV:{bag:6,first:10,last:6,last2:7},
   HELDOUT:{bag:6,first:9,last:8,last2:9}});
 const [family]=families.values(),recent=new Set(family.map(ep=>sha(v13.shortHistory(ep.initial_model_input,64))));
 assert.equal(recent.size,1);assert.equal(new Set(family.map(ep=>sha(v.eventProjection(ep.initial_model_input,"bag")))).size,1);
});
test("replay host applies legal actions, rejects blocked shutter and reaches goals",()=>{
 for(const split of splits){
  for(const ep of run.episodes[split].slice(0,4)){
   const host=v.createHost(ep),sample=v.replayAndLabel(ep);
   assert.equal(sample.rows.length,9);
   for(const action of ep.expert.actions){assert(v.observe(host).legal_actions.includes(action));assert(v.apply(host,action).ok);}
   assert.equal(host.done,true);assert.equal(host.illegal_attempts,0);
   const blocked=v13.createHost(structuredClone(ep.parent));
   blocked.closed="WEST";blocked.position=[blocked.w.cx-blocked.w.span,blocked.w.by-4];
   assert.equal(v.apply(blocked,"MOVE_SOUTH").ok,false);
  }
 }
});
test("hard negatives strictly increase oracle shortest distance",()=>{
 for(const row of Object.values(run.decisions).flat()){
  assert(row.model_input.legal_actions.includes(row.target));
  assert(!row.hard_negatives.strictly_longer_legal.includes(row.target));
  assert(row.hard_negatives.strictly_longer_legal.every(a=>row.model_input.legal_actions.includes(a)));
  assert(row.hard_negatives.blocked_or_illegal.every(a=>!row.model_input.legal_actions.includes(a)));
  assert.equal(row.native_receipt,false);
 }
 assert(run.summary.strict_hard_negatives>0);
});
test("fixed seed produces identical complete data",()=>{
 const again=v.generateDataset(1);assert.equal(sha(run),sha(again));
});
test("on-disk shards are SHA-256 verified, refuse overwrite",()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),"driv3r-v16-test-")),dest=path.join(root,"new");
 try{
  const m=v.writeDataset(dest,1);assert.equal(m.noncommutative_families,26);
  for(const [name,meta] of Object.entries(m.shards)){
   const buf=fs.readFileSync(path.join(dest,name));
   assert.equal(buf.byteLength,meta.bytes);
   assert.equal(crypto.createHash("sha256").update(buf).digest("hex"),meta.sha256);
   assert.equal(buf.toString().trimEnd().split("\n").length,meta.rows);
  }
  assert.throws(()=>v.writeDataset(dest,1),/refuse overwrite/);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
console.log(`${n}/7 regression sections PASS`);
