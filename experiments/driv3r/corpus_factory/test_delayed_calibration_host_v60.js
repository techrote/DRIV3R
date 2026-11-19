#!/usr/bin/env node
'use strict';
/* Independent v60 consumer replay: never imports v60 generator. The source
   graph host is synthetic, not CyberSand native. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v58=require('./offpolicy_repair_v58.js'),v46=require('./mission_memory_v46.js');
const H=b=>crypto.createHash('sha256').update(b).digest('hex'),E=JSON.stringify;
const ok=(v,s)=>{if(!v)throw Error('AUDIT '+s)};
const cache=new Map();
const getWorld=(s,i)=>{const key=s+':'+i;if(!cache.has(key))cache.set(key,v46.candidate(s,700+i).w);return cache.get(key);};
const readRows=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
function replay(s,cert,ep){
 const arm=v53.ARMS.find(a=>a.id===cert.arm);ok(arm,'arm missing');
 const w=getWorld(s,cert.world_index);
 ok(w.seed===cert.source_seed&&w.geometry_sha256===cert.geometry_sha256,'world tamper');
 const pars=v53.armParams(arm),host=v53.prep(w,arm);let prev=[],lastBProbe=null,atCritical=null;
 let nforced=0,marked=0,criticalSeen=0;
 for(let t=0;t<ep.steps.length;t++){
  const st=ep.steps[t];const n=v52.notice(host,pars);v53.injectB(host,arm,n,t);
  const obs=v53.observe(host),stream=v58.streamify(obs,prev);
  ok(E(stream)===E(st.input),'observation changed '+s+'/'+cert.episode+'/'+t);
  ok(obs.legal.includes(st.action),'illegal action');
  ok(!('durable' in st.input)&&Array.isArray(st.input.events_delta),'durable event leak');
  if(st.weight===0 && t>=cert.critical_step-cert.delay_actions && t<cert.critical_step)nforced++;
  if(t===cert.critical_step){
   criticalSeen++;ok(st.weight===1&&st.focus_loss_weight===8,'critical label masked');
   ok(obs.durable.some(r=>r[1]==='PAINT_RESULT'&&r[4]==='B'),'no prior physical B result');
   ok(!stream.events_delta.some(r=>r[1]==='PAINT_RESULT'&&r[4]==='B'),'B receipt resent');
   ok(st.action===cert.final_action,'final action mismatch');
   atCritical={input:stream,action:st.action,receipt:obs.durable.filter(r=>r[1]==='PAINT_RESULT'&&r[4]==='B').at(-1)};
   ok(t-lastBProbe>=cert.delay_actions,'too short since physical B probe');
   const opposite=st.action==='PAINT_A'?'PAINT_B':'PAINT_A';
   const negative=structuredClone(host),r=v52.command(negative,opposite,pars);
   ok(r.material==='A'&&!negative.done&&negative.stage===1,'wrong-material causal fork unexpectedly progressed');
  }
  prev=obs.durable;
  const before=host.pos,stage=host.stage;
  const result=v52.command(host,st.action,pars);
  if(stage===1 && before!==host.w.goal2 && st.action.startsWith('PAINT_'))lastBProbe=t+1;
  if(t===cert.critical_step)ok(result.material==='B'&&host.stage===2,'correct physical outcome not stage 2');
  marked++;
 }
 ok(nforced===cert.delay_actions,'forced detour count mismatch');
 ok(host.done&&host.success&&!host.protected_damage,'no host terminal success');
 ok(criticalSeen===1&&marked===ep.steps.length,'critical not exactly once');
 return atCritical;
}
function audit(dir){const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 ok(m.schema==='driv3r/corpus_factory/delayed-material-memory-v60/1','schema');
 const owners=new Map();let total=0,pairs=0,episodes=0;
 for(const s of ['TRAIN','DEV','HELDOUT']){
  const dat=s+(s==='TRAIN'?'_delayed_memory_stream.jsonl':'_EVALUATOR_ONLY_delayed_memory_stream.jsonl'),meta=s+'_delayed_memory_certificates_EVALUATOR_ONLY.jsonl';
  const data=readRows(path.join(dir,dat)),cert=readRows(path.join(dir,meta));
  ok(data.length===cert.length && data.length===m.splits[s].episodes,'split row lengths');
  for(let k=0;k<data.length;k+=2){
   const a=cert[k],b=cert[k+1];ok(a.pair_id===b.pair_id&&a.side===0&&b.side===1,'pair topology');
   for(const c of [a,b]){const old=owners.get(c.geometry_sha256);ok(!old||old===s,'world leakage');owners.set(c.geometry_sha256,s);}
   const x=replay(s,a,data[k]),y=replay(s,b,data[k+1]);
   ok(E(x.input)===E(y.input),'instant input differs across causal arms');
   ok(x.action!==y.action,'no label contrast');
   ok(H(E(x.input))===a.paired_current_input_sha256&&H(E(y.input))===b.paired_current_input_sha256,'snapshot SHA not authenticated');
   ok(x.receipt[3]!==y.receipt[3],'B physical result not contrasting');
   pairs++;
  }
  for(const e of data)total+=e.steps.length;
  episodes+=data.length;
 }
 ok(total===m.total_decisions&&pairs===m.total_material_pairs,'manifest totals');
 console.log(JSON.stringify({pass:true,synthetic_host_only:true,episodes,pairs,decisions:total,worlds:owners.size,final_memoryless_paired_ceiling:0.5,full_host_replay:true}));
}
const d=process.argv[2];ok(d,'usage node test_delayed_calibration_host_v60.js DIRECTORY');audit(d);
