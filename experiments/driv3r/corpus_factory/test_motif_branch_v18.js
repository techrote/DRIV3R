"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),os=require("node:os"),path=require("node:path"),crypto=require("node:crypto");
const v=require("./motif_branch_v18.js"),v17=require("./aliased_order_v17.js");
const hash=x=>crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
let tests=0;function test(name,fn){fn();tests++;console.log("PASS",name);}
const run=v.generateDataset(1),all=Object.values(run.episodes).flat();
test("TRAIN/DEV/HELDOUT disjoint worlds and topology grammar",()=>{
 const groups=new Map(),tops=new Map(),ids=new Set();
 assert.equal(all.length,60);
 for(const ep of all){assert(!ids.has(ep.episode_id));ids.add(ep.episode_id);
  for(const [index,key] of [[groups,ep.group_id],[tops,ep.topology_group]]){
   if(index.has(key))assert.equal(index.get(key),ep.split);else index.set(key,ep.split);
  }
  assert.equal(ep.native_receipt,false);assert.equal(ep.provenance.synthetic_only,true);
 }
 assert.deepEqual(Object.fromEntries(Object.entries(run.summary.counts).map(([s,x])=>[s,x.episodes])),{TRAIN:36,DEV:12,HELDOUT:12});
});
test("all controlled families are full-order discriminative and last64/event-bag ambiguous",()=>{
 const fam=new Map();for(const ep of all){const x=fam.get(ep.family_id)||[];x.push(ep);fam.set(ep.family_id,x);}
 assert.equal(fam.size,5);
 for(const eps of fam.values()){
  assert.equal(eps.length,12);
  assert.equal(new Set(eps.map(x=>hash(v.recent(x.initial_model_input,64)))).size,1);
  assert.equal(new Set(eps.map(x=>hash(v.eventBag(x.initial_model_input)))).size,1);
  assert.equal(new Set(eps.map(x=>hash(x.initial_model_input))).size,12);
  assert.deepEqual(eps.map(v.oracleFirst).sort().reduce((a,k)=>(a[k]=(a[k]||0)+1,a),{}),{MOVE_NORTH:6,MOVE_SOUTH:6});
  assert.equal(new Set(eps.map(x=>hash(x.initial_model_input.local_material_7x7))).size,1);
 }
 for(const split of ["TRAIN","DEV","HELDOUT"])
  for(const mode of ["bag","first2","last2"])assert.equal(run.summary.first_action_control_ceiling_by_split[split][mode].correct,run.summary.counts[split].episodes/2);
});
test("per-episode oracle fully paints only correct lane, navigates bypass and reaches exit",()=>{
 for(const ep of all){const h=v.createHost(ep),actions=v.oracle(h);
  assert(h.done);assert.equal(h.invalid,0);assert.equal(actions[0],v.oracleFirst(ep));
  assert.equal(Object.keys(h.paint).length,5);assert.equal(h.receipts.length,5);
  assert.equal(Object.keys(ep.privileged_host.target).length,5);
  for(const [k,ink] of Object.entries(ep.privileged_host.target))assert.equal(h.paint[k],ink);
  for(const k of h.w.protected)assert.equal(h.paint[k],undefined);
  assert.equal(h.position.join(),h.w.exit.join());
 }
});
test("no protected traversal or paint is legal, rejected actions cannot advance",()=>{
 const ep=all[0],h=v.createHost(ep),w=h.w;
 const protect=[w.cx+4,w.jy-2],neighbor=[w.cx+3,w.jy-2],steps=v.shortest(w,h.position,neighbor);
 assert(steps);for(let i=1;i<steps.length;i++){
  const d=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]].find(x=>steps[i-1][0]+x[1]===steps[i][0]&&steps[i-1][1]+x[2]===steps[i][1]);assert(v.apply(h,d[0]).ok);
 }
 assert.equal(h.position.join(),neighbor.join());assert.equal(protect.join(),[w.cx+4,w.jy-2].join());
 const before=h.steps;assert.equal(v.apply(h,"MOVE_EAST").ok,false);assert.equal(h.steps,before);
 assert.equal(v.apply(h,"PAINT_STONE").ok,true);assert.equal(h.paint[neighbor.join()],"STONE");
 assert.equal(v.apply(h,"ERASE").ok,true);assert.equal(h.paint[neighbor.join()],undefined);
 assert.equal(v.apply(h,"PAINT_SAND").ok,true);assert.equal(h.done,false);
});
test("false completion is rejected; opposite material recipe is not visible",()=>{
 const ep=all[0],h=v.createHost(ep);
 const wrong=all.find(x=>x.family_id===ep.family_id&&v.oracleFirst(x)!==v.oracleFirst(ep));assert(wrong);
 assert.equal(hash(v.recent(ep.initial_model_input)),hash(v.recent(wrong.initial_model_input)));
 assert.equal(hash(v.eventBag(ep.initial_model_input)),hash(v.eventBag(wrong.initial_model_input)));
 const goalPath=v.shortest(h.w,h.position,h.w.exit);
 for(let i=1;i<goalPath.length;i++){const a=goalPath[i-1],b=goalPath[i],d=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]].find(x=>a[0]+x[1]===b[0]&&a[1]+x[2]===b[1]);assert(v.apply(h,d[0]).ok);}
 assert.equal(h.done,false);assert.equal(h.paint.length,undefined);
});
test("all sampled decisions are host-legal and SLM input is isolated",()=>{
 for(const row of Object.values(run.decisions).flat()){
  assert(row.model_input.legal_actions.includes(row.target));assert.equal(row.native_receipt,false);
  assert(row.hard_negatives.other_legal.every(a=>row.model_input.legal_actions.includes(a)&&a!==row.target));
  assert(row.hard_negatives.illegal.every(a=>!row.model_input.legal_actions.includes(a)));
  assert(!("privileged_host" in row.model_input));assert(!("target" in row.model_input));
  assert(!("split" in row.model_input));assert(!("lane" in row.model_input));
 }
 assert.equal(Object.values(run.decisions).flat().filter(r=>r.phase==="paint").length,all.length*5);
});
test("deterministic regeneration, SHA-256 disk manifests and overwrite refusal",()=>{
 assert.equal(hash(run),hash(v.generateDataset(1)));
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),"driv3r-v18-")),target=path.join(dir,"new");
 try{const m=v.writeDataset(target,1);for(const [file,meta] of Object.entries(m.shards)){
  const b=fs.readFileSync(path.join(target,file));assert.equal(b.byteLength,meta.bytes);
  assert.equal(crypto.createHash("sha256").update(b).digest("hex"),meta.sha256);
  assert.equal(b.toString().trimEnd().split("\n").length,meta.rows);
 }assert.throws(()=>v.writeDataset(target,1),/refuse overwrite/);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
console.log(`${tests}/7 regression sections PASS`);
