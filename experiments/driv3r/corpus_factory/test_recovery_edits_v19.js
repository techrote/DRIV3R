"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const v=require('./recovery_edits_v19.js'),v18=require('./motif_branch_v18.js'),v17=require('./aliased_order_v17.js');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
let n=0;function test(name,fn){fn();console.log('PASS',name);n++;}
const cache=new Map();function examples(split,seed,mode){const group=[];for(let i=0;i<12;i++)group.push(v.makeRecord(split,seed,i,mode));return group;}
const cases=[['TRAIN',0,'spot'],['TRAIN',1,'all'],['TRAIN',2,'blocked'],['DEV',0,'alternating'],['HELDOUT',0,'cross'],['HELDOUT',1,'exit']];
test('all modes replay, no teacher illegal moves, protected paint cannot remain',()=>{
 for(const [split,seed,mode] of cases)for(const record of examples(split,seed,mode)){
  const h=v.createHost(record),baseInvalid=h.invalid,labels=[];
  const r=v.recovery(h,row=>{labels.push(row.target);assert(row.model_input.legal_actions.includes(row.target));
   assert(!Object.hasOwn(row.model_input,'privileged_host'));assert(!Object.hasOwn(row.model_input,'target'));});
  assert(h.done);assert.equal(h.invalid,baseInvalid);assert.equal(labels.length,r.actions);
  assert.equal(Object.keys(h.paint).length,5);
  for(const k of h.w.protected)assert.equal(h.paint[k],undefined);
 }
});
test('12-way first-action counterfactual families alias recent-64 and unordered event inventory',()=>{
 for(const [split,seed,mode] of cases){const group=examples(split,seed,mode);
  assert.equal(new Set(group.map(v.recentKey)).size,1);assert.equal(new Set(group.map(v.bagKey)).size,1);
  assert.equal(new Set(group.map(x=>sha(JSON.stringify(x.checkpoint_model_input)))).size,12);
  const actions=group.map(r=>v.recovery(v.createHost(r)).first_action);
  assert.equal(actions.filter(a=>a==='MOVE_NORTH').length,6);assert.equal(actions.filter(a=>a==='MOVE_SOUTH').length,6);
 }
});
test('intervention receipts and required cleanup are auditable',()=>{
 for(const [split,seed,mode] of cases){const record=examples(split,seed,mode)[0],h=v.createHost(record),recs=h.receipts;
  if(mode==='blocked'){
   assert.equal(record.scripted_intervention_counts.rejected,2);assert.equal(h.invalid,2);
   assert.equal(recs.filter(r=>r.kind==='synthetic_refusal'&&r.accepted===false).length,2);
  }else assert.equal(h.invalid,0);
  const before=Object.keys(h.paint).length,r=v.recovery(h);
  if(mode==='spot')assert.equal(before,2);
  if(mode==='all')assert.equal(before,10);
  if(mode==='alternating')assert.equal(before,6);
  if(mode==='cross')assert.equal(before,4);
  if(mode==='exit')assert.equal(before,0);
  if(['spot','all','alternating','cross'].includes(mode))assert(r.erase_actions>0);
  assert.equal(h.invalid,mode==='blocked'?2:0);
 }
});
test('nontrivial recovery differs from original clean v18 oracle and retains wrong-paint cleanup',()=>{
 const r=v.makeRecord('TRAIN',0,0,'all'),h=v.createHost(r),metrics=v.recovery(h);
 assert.equal(metrics.erase_actions,5);assert.equal(metrics.paint_actions,5);
 assert.equal(metrics.initial_rejections,0);
 const a=v18.oracle(v18.createHost(r.base_episode));assert(metrics.actions>a.length);
});
test('split-specific challenge modes are explicit and heldout includes unseen contamination',()=>{
 assert.deepEqual(v.MODES.TRAIN,['spot','all','blocked']);assert.deepEqual(v.MODES.DEV,['alternating','blocked']);
 assert.deepEqual(v.MODES.HELDOUT,['cross','exit','blocked']);
 assert(!v.MODES.TRAIN.includes('cross'));assert(!v.MODES.DEV.includes('cross'));
});
test('scripted interventions never disclose privileged lane or oracle target in model-visible fields',()=>{
 const a=examples('TRAIN',0,'spot');for(const rec of a){const x=rec.checkpoint_model_input;
  assert(!Object.hasOwn(x,'lane'));assert(!Object.hasOwn(x,'privileged_host'));assert(!Object.hasOwn(x,'target'));
  assert(!Object.hasOwn(x,'intervention_mode'));assert(!Object.hasOwn(x,'split'));
  assert(x.legal_actions.length>0);assert(x.history.events.length>0);
 }
});
test('source pinned to v17 ordered histories and v18 replay semantics',()=>{
 assert.equal(v17.ORDERS_BY_SPLIT.TRAIN.length,12);
 assert.equal(v17.ORDERS_BY_SPLIT.DEV.length,12);
 assert.equal(v17.ORDERS_BY_SPLIT.HELDOUT.length,12);
 const rec=v.makeRecord('HELDOUT',1,3,'blocked');const h=v.createHost(rec);
 assert.equal(sha(JSON.stringify(v18.observe(h))),sha(JSON.stringify(rec.checkpoint_model_input)));
});
test('disk shards, complete SHA-256, deterministic regeneration, overwrite rejection',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v19-'));
 try{
  const a=path.join(dir,'first'),b=path.join(dir,'second'),m=v.recordStream(a,1),m2=v.recordStream(b,1);
  assert.deepEqual(m,m2);
  assert.deepEqual(Object.fromEntries(Object.entries(m.split_summary).map(([k,x])=>[k,x.episodes])),{TRAIN:108,DEV:24,HELDOUT:36});
  assert.equal(m.families,14);
  for(const [file,meta] of Object.entries(m.shards)){const x=fs.readFileSync(path.join(a,file)),y=fs.readFileSync(path.join(b,file));
   assert(x.equals(y));assert.equal(x.length,meta.bytes);assert.equal(sha(x),meta.sha256);
   assert.equal(x.toString().trimEnd().split('\n').length,meta.rows);
  }
  assert.throws(()=>v.recordStream(a,1),/refuse overwrite/);
 }finally{fs.rmSync(dir,{recursive:true,force:true});}
});
console.log(`${n}/8 sections PASS`);
