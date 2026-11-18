'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const m=require('./delayed_obstruction_v40.js');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const assert=(x,s)=>{if(!x)throw Error(s);};
function audit(dirA,dirB){
 const a=JSON.parse(fs.readFileSync(path.join(dirA,'manifest.json'),'utf8'));
 const b=JSON.parse(fs.readFileSync(path.join(dirB,'manifest.json'),'utf8'));
 assert(JSON.stringify(a)===JSON.stringify(b),'manifest nondeterministic');
 assert(a.schema==='driv3r/corpus_factory/delayed-obstruction-v40/manifest','schema');
 let rows=0,positives=0,negatives=0,contexts=0,labels=new Map(),geos=new Map(),zero=0;
 for(const [name,meta] of Object.entries(a.shards)){
  const A=fs.readFileSync(path.join(dirA,name)),B=fs.readFileSync(path.join(dirB,name));
  assert(A.equals(B)&&sha(A)===meta.sha256&&A.length===meta.bytes,'shard mismatch '+name);
  const data=A.toString().trim().split('\n').map(JSON.parse);
  assert(data.length===meta.rows,'shard rows '+name);
  const split=name.split('_')[0];
  if(name.endsWith('_challenge.jsonl'))for(const c of data){
   contexts++;
   assert(c.synthetic_not_native&&c.steps_clear>0&&c.steps_blocked>0&&c.obstruction_event_age_actions===160,'challenge invalid');
   if(c.positive){positives++;assert(c.clear_regret>=2&&c.blocked_regret>=2,'no strict regret');
    assert(c.clear_action==='MOVE_E'&&c.blocked_action==='MOVE_S','label contrast absent');}
   else {negatives++;assert(c.clear_action===c.blocked_action,'control changed');}
  }
  if(name.endsWith('_policy.jsonl'))for(const row of data){
   rows++;const x=row.input;
   assert(JSON.stringify(Object.keys(row).sort())===JSON.stringify(['action','input','weight']),'teacher field leaked');
   assert(x.legal.includes(row.action),'illegal teacher label');
   assert(x.v==='synthetic-v40-delayed-obstruction','bad view version');
   const events=x.durable.filter(e=>e[1]==='REMOTE_OBSTRUCTION');
   assert(events.length===1&&[0,1].includes(events[0][2])&&events[0][3].length===2,'missing historical obstruction');
   assert(!Object.hasOwn(x,'obstruction'),'current-state obstruction leakage');
   assert([0,1].includes(row.weight),'bad weight');
   if(!row.weight)zero++;
   const key=sha(JSON.stringify(x)),old=labels.get(key);assert(!old||old.split===split,'cross-split input overlap');
   assert(!old||old.action===row.action,'input action conflict');labels.set(key,{split,action:row.action});
   const geo=sha(JSON.stringify([x.cells,x.edges]));const owner=geos.get(geo);
   assert(!owner||owner===split,'cross split topology overlap');geos.set(geo,split);
  }
 }
 assert(rows===(a.quick?4704:11744),'unexpected total policy decisions');
 assert(positives===(a.quick?16:32)&&negatives===(a.quick?16:32)&&contexts===(a.quick?32:64),'unexpected challenge balance');
 assert(zero===0,'unexpected duplicate weight');
 for(const split of ['TRAIN','DEV','HELDOUT']){
  assert(a.splits[split].positive_pairs>0,'no contrasts in '+split);
  const w=m.makeWorld(split,0,0),p=m.pick(w,split,0,0);
  assert(p.found.length===1,'incorrect positive family');
  const f=p.found[0],abl=x=>sha(JSON.stringify(m.forgetObstruction(x)));
  assert(f.a.eventAge===160&&f.b.eventAge===160,'historical transition not aged');
  assert(sha(JSON.stringify(f.a.h.history.events.slice(-64)))===sha(JSON.stringify(f.b.h.history.events.slice(-64))),'recent 64 diverged');
  assert(sha(JSON.stringify(f.a.h.history.positions.slice(-64)))===sha(JSON.stringify(f.b.h.history.positions.slice(-64))),'recent positions diverged');
  assert(sha(JSON.stringify(f.a.input))!==sha(JSON.stringify(f.b.input)),'full memory fails distinction');
  assert(abl(f.a.input)===abl(f.b.input),'ablation not ambiguous');
  assert(f.a.input.legal.includes(f.b.rank[0].action)&&f.b.input.legal.includes(f.a.rank[0].action),'trivial legality contrast');
  assert(f.contrast.clear_regret>=2&&f.contrast.blocked_regret>=2,'insufficient regret');
 }
 return {rows,positives,negatives,contexts,unique_inputs:labels.size,split_topologies:geos.size,manifest_sha256:sha(fs.readFileSync(path.join(dirA,'manifest.json')))};
}
if(require.main===module){const [a,b]=process.argv.slice(2);assert(a&&b,'usage: node test_delayed_obstruction_v40.js DIR1 DIR2');console.log(JSON.stringify(audit(a,b),null,2));}
module.exports={audit};
