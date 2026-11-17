'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const v=require('./topology_relay_v20.js');const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
let passes=0;function test(name,fn){fn();passes++;console.log('PASS',name);}
test('world split diversity, non-overlapping length/detour bands and unique geometry fingerprints',()=>{
 const seen=new Set();for(const [split,cfg] of Object.entries(v.SPLITS))for(let i=0;i<cfg.groups;i++){
  const w=v.makeWorld(split,i);assert(!seen.has(w.fingerprint));seen.add(w.fingerprint);
  for(const n of Object.values(w.lengths))assert(n>=cfg.min&&n<=cfg.max);
  assert.equal(w.patches.N.length,5);assert.equal(w.patches.S.length,5);assert(w.gates.length>=2);
 }assert.equal(seen.size,36);
});
test('all 8 counterfactual siblings require three old receipts: recent64 and every 1-/2-bit ablation balanced',()=>{
 for(const [split,cfg] of Object.entries(v.SPLITS))for(let i=0;i<cfg.groups;i++){
  const records=v.verifyFamily(split,i);
  assert.equal(new Set(records.map(r=>v.projection(r))).size,1);
  for(const mask of [1,2,4,3,5,6]){const buckets={};for(const r of records){const b=r.private_target.bits.filter((_,j)=>mask&(1<<j)).join('');(buckets[b]??=[]).push(r.private_target.active);}
   for(const b of Object.values(buckets))assert.equal(b.filter(x=>x==='N').length,b.filter(x=>x==='S').length);
  }
 }
});
test('closed-loop oracle must obey observation legal actions and complete all sampled worlds',()=>{
 for(const split of Object.keys(v.SPLITS))for(let group=0;group<Math.min(4,v.SPLITS[split].groups);group++)for(let variant=0;variant<8;variant++){
  const rec=v.makeRecord(split,group,variant),host=v.createHost(rec);let steps=0;
  while(!host.done){assert(++steps<400);const input=v.observe(host),action=v.oracleAction(host);assert(input.legal_actions.includes(action));assert(v.apply(host,action).accepted);}
  assert(v.correctTerminal(host));assert(host.shutters_closed);assert.equal(Object.keys(host.paint).length,5);
  for(const [k,t] of Object.entries(host.world.tiles))if(t.type==='protected')assert.equal(host.paint[k],undefined);
 }
});
test('protected movement is rejected with actual synthetic host receipt and no actor movement',()=>{
 const rec=v.makeRecord('TRAIN',0,0),host=v.createHost(rec);
 assert.equal(host.pos,'0,0');assert.equal(host.invalid,1);
 const x=host.events.at(-1);assert.deepEqual([x.kind,x.accepted,x.reason],['action_receipt',false,'protected']);
 assert(!v.observe(host).legal_actions.includes('MOVE_EAST'));assert(!v.observe(host).legal_actions.includes('MOVE_WEST'));
});
test('dual shutter closures change legal movement but preserve alternative reachable routes',()=>{
 const host=v.createHost(v.makeRecord('HELDOUT',0,7));
 for(let i=0;i<8;i++){let action=v.oracleAction(host);assert(v.apply(host,action).accepted);}
 assert(host.shutters_closed);assert.equal(host.events.filter(x=>x.channel==='DUAL_SHUTTERS').length,1);
 assert(v.replay(v.makeRecord('HELDOUT',0,7)).terminal_correct);
});
test('model-visible record excludes target, world layouts, split and variant',()=>{
 const r=v.makeRecord('TRAIN',1,6),x=v.observe(v.createHost(r));
 for(const field of ['active','private_target','target','variant','world','split','geometry_sha256'])assert(!Object.hasOwn(x,field));
 assert(x.history.events.length>=83);assert.equal(x.view.length,25);
});
test('CLI shards, hashes, reproducibility, overwrite rejection and counts',()=>{
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v20-test-'));try{
  const a=path.join(tmp,'a'),b=path.join(tmp,'b'),one=v.recordStream(a),two=v.recordStream(b);
  assert.deepEqual(one,two);assert.equal(one.world_group_count,36);assert.equal(one.episodes,288);
  for(const [name,m] of Object.entries(one.shards)){assert.equal(m.sha256,two.shards[name].sha256);assert.equal(m.bytes,fs.statSync(path.join(a,name)).size);assert.equal(m.bytes,fs.statSync(path.join(b,name)).size);assert(m.rows>0);}
  assert.throws(()=>v.recordStream(a),/refuse overwrite/);
  assert(one.decisions>10000);
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
});
console.log(`${passes}/7 sections PASS`);
