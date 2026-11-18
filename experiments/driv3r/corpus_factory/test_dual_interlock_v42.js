'use strict';
// Independent deterministic corpus audit. No policy trained here.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const m=require('./dual_interlock_v42.js'),v41=require('./delayed_obstruction_v41.js');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const assert=(ok,msg)=>{if(!ok)throw Error(msg);};
const read=(p)=>JSON.parse(fs.readFileSync(p));
function audit(dir1,dir2,{quick=false}={}){
 const A=read(path.join(dir1,'manifest.json')),B=read(path.join(dir2,'manifest.json'));
 assert(JSON.stringify(A)===JSON.stringify(B),'manifest not byte-deterministic');
 assert(A.schema==='driv3r/corpus_factory/dual-interlock-v42/manifest'&&A.quick===quick,'wrong corpus schema');
 let records=0,pairs=0,positive=0,negative=0,success=0,zero=0,unique=0;
 const splitOwners=new Map(),geoOwners=new Map(),actions=new Map();
 for(const [name,meta] of Object.entries(A.shards)){
   const data=fs.readFileSync(path.join(dir1,name)),data2=fs.readFileSync(path.join(dir2,name));
   assert(data.equals(data2)&&sha(data)===meta.sha256&&data.length===meta.bytes,'different or corrupt shard '+name);
   const rows=data.toString().trim().split('\n').map(JSON.parse),split=name.split('_')[0];
   assert(rows.length===meta.rows,'wrong shard row count '+name);
   if(name.endsWith('_policy.jsonl'))for(const row of rows){
     records++;assert(JSON.stringify(Object.keys(row).sort())===JSON.stringify(['action','input','weight']),'trainer input leak');
     const x=row.input;
     assert(x.v==='synthetic-v42-dual-interlock'&&x.legal.includes(row.action),'non-legal action label');
     assert(!Object.hasOwn(x,'finalA')&&!Object.hasOwn(x,'finalB')&&!Object.hasOwn(x,'best'),'oracle hidden metadata leaked');
     assert(x.durable.filter(e=>e[1]==='REMOTE_INTERLOCK_A').length>=2,'no A event');
     assert(x.durable.filter(e=>e[1]==='REMOTE_INTERLOCK_B').length>=2,'no B event');
     assert(x.durable.filter(e=>e[1]==='REMOTE_SENSOR').length>=1,'no diagnostic');
     assert([0,1].includes(row.weight),'invalid weight'); if(row.weight===0)zero++;
     const key=sha(Buffer.from(JSON.stringify(x))),prior=actions.get(key);
     assert(!prior||prior===row.action,'teacher label contradiction');actions.set(key,row.action);
     const owner=splitOwners.get(key);assert(!owner||owner===split,'cross-split model input');splitOwners.set(key,split);
     const geometry=sha(Buffer.from(JSON.stringify([x.cells,x.edges]))),before=geoOwners.get(geometry);
     assert(!before||before===split,'split geometry overlap');geoOwners.set(geometry,split);
   }
   if(name.endsWith('_challenge.jsonl'))for(const c of rows){
     pairs++;success+=2;
     assert(c.synthetic_not_native&&c.ageA>=80&&c.ageB>=112,'bad event ageing');
     assert(c.full_a_sha256!==c.full_b_sha256&&c.ablate_B_alias_sha256,'no distinguishable full memory');
     if(c.positive){positive++;
       assert(c.finalA===1&&c.first_b0==='MOVE_E'&&c.first_b1==='MOVE_S','bad contrast');
       assert(c.regret_b0>=2&&c.regret_b1>=2,'strict first-action cost not proven');
       assert(c.b_changed_physical_b0>=1&&c.b_changed_physical_b1>=1,'B never changed physical graph');
     }else{negative++;
       assert(c.finalA===0&&c.first_b0===c.first_b1,'negative-control action changed');
       assert(c.b_changed_physical_b0>=1&&c.b_changed_physical_b1>=1,'negative control did not exercise B under A=1');
     }
   }
 }
 for(const split of ['TRAIN','DEV','HELDOUT']){
   const w=v41.makeWorld(split,0,0);
   for(const v of [0,1,2,3]){
     const p=m.pairAt(w,split,0,0,1,v,1);
     assert(p.contrast,'failure to recreate positive pair');
     const x=p.a.input,y=p.b.input;
     assert(sha(Buffer.from(JSON.stringify(m.withoutB(x))))===sha(Buffer.from(JSON.stringify(m.withoutB(y)))),'B-only ablation mismatch');
     const a=x.durable.filter(e=>e[1]==='REMOTE_INTERLOCK_A'),b=y.durable.filter(e=>e[1]==='REMOTE_INTERLOCK_A');
     assert(JSON.stringify(a)===JSON.stringify(b),'last A differs');
     assert(a.at(-1)[2]===1,'newest actuator not A=1');
     const latestA=a.at(-1)[0],latestB=x.durable.filter(e=>e[1]==='REMOTE_INTERLOCK_B').at(-1)[0];
     assert(latestA>latestB,'B was newer than A');
     assert(sha(Buffer.from(JSON.stringify(x.recent)))===sha(Buffer.from(JSON.stringify(y.recent))),'recently differing');
     assert(sha(Buffer.from(JSON.stringify(m.withoutInterlocks(x))))===sha(Buffer.from(JSON.stringify(m.withoutInterlocks(y)))),'too much ablation alias mismatch');
     const neg=m.pairAt(w,split,0,0,1,v,0);
     assert(!neg.contrast&&neg.a.rank[0].action===neg.b.rank[0].action,'A=0 control false positive');
   }
 }
 unique=actions.size;
 return {decisions:records,positive_evaluator_pairs:positive,negative_evaluator_pairs:negative,all_evaluator_pairs:pairs,
  evaluator_terminal_successes:success,unique_inputs:unique,unique_geometries:geoOwners.size,zero_weight:zero,
  manifest_sha256:sha(fs.readFileSync(path.join(dir1,'manifest.json')))};
}
if(require.main===module){const [a,b,flag]=process.argv.slice(2);assert(a&&b,'usage: node test_dual_interlock_v42.js DIR1 DIR2 [--quick]');console.log(JSON.stringify(audit(a,b,{quick:flag==='--quick'}),null,2));}
module.exports={audit};
