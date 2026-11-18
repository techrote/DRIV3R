'use strict';
// Independent corpus-format and causal-memory audit, not an SLM evaluation.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const m=require('./delayed_obstruction_v41.js');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const fail=(x,s)=>{if(!x)throw Error(s);};
function audit(a,b){
 const A=JSON.parse(fs.readFileSync(path.join(a,'manifest.json')));
 const B=JSON.parse(fs.readFileSync(path.join(b,'manifest.json')));
 fail(JSON.stringify(A)===JSON.stringify(B),'nondeterministic manifest');
 fail(A.schema==='driv3r/corpus_factory/delayed-obstruction-v41/manifest','unexpected schema');
 let rows=0,positives=0,negative=0,distinct=0,seen=new Map(),geo=new Map(),zero=0;
 for(const [f,meta] of Object.entries(A.shards)){
   const bytes=fs.readFileSync(path.join(a,f)),second=fs.readFileSync(path.join(b,f));
   fail(bytes.equals(second)&&sha(bytes)===meta.sha256&&bytes.length===meta.bytes,'shard bytes '+f);
   const records=bytes.toString().trim().split('\n').map(JSON.parse);
   fail(records.length===meta.rows,'shard count '+f);
   const split=f.split('_')[0];
   if(f.endsWith('_challenge.jsonl'))for(const c of records){
     distinct++;
     fail(c.synthetic_not_native&&c.obstruction_event_age_actions>=72&&c.junction_event_count>=2&&c.sensor_event_count>=1,'invalid challenge');
     fail(c.pattern>=0&&c.pattern<=3,'invalid schedule');
     if(c.positive){positives++;
       fail(c.clear_action==='MOVE_E'&&c.blocked_action==='MOVE_S','not an optimal first-action contrast');
       fail(c.clear_regret>=2&&c.blocked_regret>=2,'wrong-first regret too low');
     }else{negative++;fail(c.clear_action===c.blocked_action,'negative control changed');}
   }
   if(f.endsWith('_policy.jsonl'))for(const r of records){
     rows++;fail(JSON.stringify(Object.keys(r).sort())===JSON.stringify(['action','input','weight']),'teacher leakage');
     fail(r.input.legal.includes(r.action),'teacher illegal action');
     fail(r.input.v==='synthetic-v41-delayed-obstruction'&&!Object.hasOwn(r.input,'obstruction'),'observation current-state leakage');
     const real=r.input.durable.filter(e=>e[1]==='REMOTE_OBSTRUCTION'),sensor=r.input.durable.filter(e=>e[1]==='REMOTE_SENSOR');
     fail(real.length>=2&&sensor.length>=1,'missing mixed historical events');
     fail(real[real.length-1][0]<r.input.durable.length+1000,'bad event indices');
     fail([0,1].includes(r.weight),'bad duplicate weight'); if(!r.weight)zero++;
     const k=sha(JSON.stringify(r.input)),prior=seen.get(k);
     fail(!prior||prior.split===split,'cross-split policy input');
     fail(!prior||prior.action===r.action,'conflicting policy labels');seen.set(k,{split,action:r.action});
     const topology=sha(JSON.stringify([r.input.cells,r.input.edges]));
     fail(!geo.has(topology)||geo.get(topology)===split,'cross-split geometry');geo.set(topology,split);
   }
 }
 for(const split of ['TRAIN','DEV','HELDOUT']){
   const w=m.makeWorld(split,0,0),p=m.pick(w,split,0,0);
   fail(p.found.length===4,'missing four patterns');
   for(const pair of p.found){
     const x=pair.a,y=pair.b;
     fail(x.phase==='CLEAR'&&y.phase==='BLOCKED','phase mismatch');
     fail(x.eventAge>=72&&y.eventAge>=72,'event not old enough');
     fail(JSON.stringify(x.h.history.events.slice(-64))===JSON.stringify(y.h.history.events.slice(-64)),'recent64 events differ');
     fail(JSON.stringify(x.h.history.positions.slice(-64))===JSON.stringify(y.h.history.positions.slice(-64)),'recent64 positions differ');
     fail(sha(JSON.stringify(x.input))!==sha(JSON.stringify(y.input)),'full input alias');
     fail(sha(JSON.stringify(m.forgetObstruction(x.input)))===sha(JSON.stringify(m.forgetObstruction(y.input))),'obstruction-only ablation fails');
     const remoteA=x.input.durable.filter(e=>e[1]==='REMOTE_OBSTRUCTION'),remoteB=y.input.durable.filter(e=>e[1]==='REMOTE_OBSTRUCTION');
     fail(remoteA.length>=2&&remoteB.length===remoteA.length,'multi-event causal transcript lost');
     fail(remoteA.at(-1)[2]===0&&remoteB.at(-1)[2]===1,'final event does not match world');
     fail(x.rank[0].action==='MOVE_E'&&y.rank[0].action==='MOVE_S','replay oracle disagrees');
   }
 }
 return {rows,positive_challenge_pairs:positives,negative_controls:negative,challenge_pairs:distinct,unique_inputs:seen.size,unique_geometries:geo.size,zero_weight:zero,manifest_sha256:sha(fs.readFileSync(path.join(a,'manifest.json')))};
}
if(require.main===module){const [a,b]=process.argv.slice(2);fail(a&&b,'usage: node test_delayed_obstruction_v41.js DIR1 DIR2');console.log(JSON.stringify(audit(a,b),null,2));}
module.exports={audit};
