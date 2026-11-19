#!/usr/bin/env node
'use strict';
/** Corpus Factory v64 — delayed obsolete remote-gate reports with explicit
 * measurement timestamps, counterfactual first-route decisions and full
 * synthetic-host replay. Not native CyberSand. DATA ONLY, never policy training.
 * No new action or privileged model-input token is introduced. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const partial=require('./frontier_memory_v45.js');
const stream=require('./offpolicy_repair_v58.js');
const v63=require('./gate_revision_v63.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(ok,msg)=>{if(!ok)throw Error('V64 '+msg)};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const parents=['unseen_topology_v44.js','frontier_memory_v45.js','offpolicy_repair_v58.js','gate_revision_v63.js'];
const SPLITS={TRAIN:8,DEV:4,HELDOUT:4},DELAY=v63.DELAY;
function lateStep(split){const d=DELAY[split];return split==='TRAIN'?d*3/4:split==='DEV'?Math.floor(d*5/6):Math.floor(d*7/8);}
function eventStateByMeasurementTime(durable){
 const gate=durable.filter(x=>x[1]==='EDGE_GATE');
 check(gate.length>=2,'no gate measurement');
 const max=gate.reduce((a,b)=>b[0]>a[0]?b:a);
 check(!gate.some(x=>x[0]===max[0]&&x[4]!==max[4]),'conflicting measurements with same timestamp');
 return !!max[4];
}
function modelVisibleTimestampTeacher(raw){
 const visible={...raw,durable:raw.durable.slice()};
 const gates=visible.durable.filter(x=>x[1]==='EDGE_GATE').sort((a,b)=>a[0]-b[0]);
 const rest=visible.durable.filter(x=>x[1]!=='EDGE_GATE');
 visible.durable=rest.concat(gates); // only gate receipts are ordered by measurement time
 return partial.teacher(visible);
}
function sourceEpisode(w,kind,oldClosed,newClosed,delay){return v63.episode(w,kind,oldClosed,newClosed,delay);}
function replay(w,kind,oldClosed,newClosed,order,split,original){
 const d=DELAY[split],h=partial.prelude(w,kind,oldClosed),staleAt=order==='stale-last'?lateStep(split):Math.floor(d/4);
 check(h.pos===w.s&&h.stage===0&&!h.done,'bad prelude');
 const previous=h.events.filter(x=>x.kind==='EDGE_GATE');
 check(previous.length>=2 && previous.at(-1).value===Number(oldClosed),'old receipt absent');
 const obsolete={...previous.at(-1)}; // measurement timestamp retains its original prelude value
 const rows=[];let prior=[],critical=null,staleInputStep=-1,firstFreshStep=-1,trainCritical=0;
 for(let t=0;t<original.steps.length;t++){
  if(t===d/2){h.closed=newClosed;h.events.push({kind:'EDGE_GATE',accepted:true,edge:h.gate,value:Number(newClosed),at:h.accepted});firstFreshStep=t;}
  if(t===staleAt){h.events.push({...obsolete});staleInputStep=t;}
  const raw=partial.observe(h),inp=stream.streamify(raw,prior),reference=original.steps[t];
  check(raw.legal.includes(reference.action),'illegal original action at '+t);
  check(!Object.hasOwn(inp,'durable')&&!Object.hasOwn(inp,'closed')&&!Object.hasOwn(inp,'gate'),'state leakage');
  if(t===staleAt){const gate=inp.events_delta.filter(x=>x[1]==='EDGE_GATE');check(gate.length===1&&gate[0][0]===obsolete.at&&gate[0][4]===obsolete.value,'stale delivered improperly');}
  if(t===d/2){const gate=inp.events_delta.filter(x=>x[1]==='EDGE_GATE');check(gate.length===1&&gate[0][4]===Number(newClosed),'fresh measurement missing');}
  if(t===d){
   check(inp.events_delta.length===0&&h.pos===w.s,'critical event leakage or position');
   check(modelVisibleTimestampTeacher(raw)===reference.action,'timestamp-visible teacher disagrees with physical oracle');
   check(eventStateByMeasurementTime(raw.durable)===newClosed,'measurement time fails to select live state');
   const shortest=host.ranks(h.g,h.pos,w.goal,newClosed?h.gate:-1)[0]?.action;
   check(shortest===reference.action,'physical optimal route contradicted');
   trainCritical=reference.weight;
   critical={step:t,input_sha256:H(inp),action:reference.action,
    latest_arrival_state:!!raw.durable.filter(x=>x[1]==='EDGE_GATE').at(-1)[4],
    measurement_time_state:eventStateByMeasurementTime(raw.durable)};
  }
  rows.push({input:inp,action:reference.action,weight:reference.weight,focus_loss_weight:reference.focus_loss_weight});
  prior=raw.durable;
  check(host.apply(h,reference.action).accepted,'host rejected '+t);
 }
 check(h.done&&h.success&&h.stage===2&&critical&&trainCritical===1,'failed complete objectives');
 return {steps:rows,critical,staleStep:staleInputStep,freshStep:firstFreshStep,obsoleteAt:obsolete.at,delay:d,world:w,kind,order};
}
function write(root,name,rows,files){const data=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(root,name),data,{flag:'wx'});files[name]={sha256:H(data),bytes:data.length,rows:rows.length};}
function generate(root,{quick=false}={}){
 check(!fs.existsSync(root),'destination exists');fs.mkdirSync(root,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/delayed-bulletin-v64/1',date:'2026-11-19',quick,
  synthetic_not_native_CyberSand:true,generator_sha256:H(fs.readFileSync(__filename)),
  source_module_sha256:Object.fromEntries(parents.map(n=>[n,H(fs.readFileSync(path.join(__dirname,n)))])),
  scenario:'Counterfactual remote pivotal-vs-irrelevant graph obstruction; old authentic timestamped survey receipt is re-delivered before or after a newer physical gate-state change. False most-recent-arrival inference competes with correct highest-measurement-time inference after long masked branch detours.',
  input_contract:'Only steps[].input and preceding visible actions reach model; no durable replay, hidden gate state, evaluator certificate or split-arm names. EDGE_GATE tuple[0] is measurement timestamp, not receipt order. Every events_delta is delivered exactly once.',
  controls:'Pivotal memoryless-first and old-first 50% critical accuracy; most-recent-arrival 75% over both delivery orders, fails every stale-last true-revision case; highest-measurement-time 100% symbolic oracle. These are not neural-model measurements.',
  labels:'Detour host-executed but zero-weight intervention. Remaining oracle route is host-valid, observable-survey and measurement-time derivable. Critical decision weight 1 / focus 8.',
  integrity:'Split-distinct source geometries, 16-arm paired worlds. Evaluator certificates separated. No TRAIN examples from DEV/HELDOUT. TRAIN noncritical weights inherit v63 duplicate-mask; critical opposites remain.',
  limitations:['Synthetic graph host not native CyberSand; no target SLM optimizer updates, autonomous neural HELDOUT, CPU neural RSS/latency or immutable Actions artifacts.',
   'Timestamp-aware symbolic event parser solves this benchmark; tests evidence admissible data causality, not learned planning.',
   'Historical source survey and transient detour commands are scripted; 4 HELDOUT worlds are a focused diagnostic, not broad generalization.'],
  files:{},splits:{}};
 const owners=new Map();let total=0,episodes=0,positives=0,controls=0,staleFailures=0;
 for(const [split,n] of Object.entries(SPLITS)){
  let count=quick?Math.min(1,n):n,shards=[],certs=[],seenHistory=new Set(),seenStateAction=new Set(),decisions=0,criticalMemoryless=0,criticalArrival=0,criticalTimestamp=0,
   criticalPos=0,maskedDetours=0,trainDuplicateMasked=0,staleLastRevisions=0;
  for(let i=0;i<count;i++){
   const {w,sourceIndex,trial}=v63.qualifiedWorld(split,i);
   const owner=owners.get(w.geometry_sha256);check(!owner||owner===split,'cross-split geometry leakage');owners.set(w.geometry_sha256,split);
   const arms=[];
   for(const kind of ['positive','control'])for(const oldClosed of [false,true])for(const newClosed of [false,true]){
    const base=sourceEpisode(w,kind,oldClosed,newClosed,DELAY[split]);
    for(const order of ['fresh-last','stale-last']){
     const r=replay(w,kind,oldClosed,newClosed,order,split,base);
     arms.push({kind,oldClosed,newClosed,order,r});
    }
   }
   check(arms.length===16,'arm count');
   const baseline=arms[0].r.steps[DELAY[split]].input;
   check(arms.every(x=>same(x.r.steps[DELAY[split]].input,baseline)),'critical inputs differ');
   const pivot=arms.filter(x=>x.kind==='positive'),control=arms.filter(x=>x.kind==='control');
   check(new Set(pivot.filter(x=>x.newClosed===false).map(x=>x.r.critical.action)).size===1,'open route differs');
   check(new Set(pivot.filter(x=>x.newClosed===true).map(x=>x.r.critical.action)).size===1,'closed route differs');
   check(pivot[0].r.critical.action!==pivot[2].r.critical.action,'no causal pivotal route switch');
   check(control.every(x=>x.r.critical.action===control[0].r.critical.action),'irrelevant gate changed route');
   for(const arm of arms){
    const {r}=arm,idx=shards.length;
    check(r.critical.measurement_time_state===arm.newClosed,'physically selected gate inconsistent');
    // Retain intentionally contradictory critical labels but remove noncritical
    // repeated TRAIN supervision, including duplicate prefixes in the two
    // temporal delivery orders. Never manipulate DEV/HELDOUT loss masks.
    if(split==='TRAIN'){
     let history='episode';
     for(let j=0;j<r.steps.length;j++){
      const row=r.steps[j];history=H(history+JSON.stringify([row.input,row.action]));
      if(row.weight&&j!==r.critical.step){
       const stateAction=H([row.input,row.action]);
       if(seenHistory.has(history)||seenStateAction.has(stateAction)){
        row.weight=0;row.focus_loss_weight=0;trainDuplicateMasked++;
       } else{seenHistory.add(history);seenStateAction.add(stateAction);}
      }
     }
    }
    shards.push({steps:r.steps});decisions+=r.steps.length;episodes++;
    maskedDetours+=DELAY[split];
    if(arm.kind==='positive'){
     criticalPos++;positives++;criticalMemoryless+=Number(r.critical.action===pivot[0].r.critical.action);
     const lastState=r.critical.latest_arrival_state;
     criticalArrival+=Number(lastState===arm.newClosed);
     criticalTimestamp+=Number(r.critical.measurement_time_state===arm.newClosed);
     if(arm.order==='stale-last'&&arm.oldClosed!==arm.newClosed){staleLastRevisions++;staleFailures+=Number(lastState!==arm.newClosed);}
    }else{controls++;}
    certs.push({split,group:split+':'+i,episode:idx,geometry_sha256:w.geometry_sha256,seed:w.seed,
     source_world_index:sourceIndex,source_trial:trial,kind:arm.kind,old_closed:arm.oldClosed,
     revised_closed:arm.newClosed,delivery:arm.order,stale_receipt_measurement_at:r.obsoleteAt,
     revision_step:r.freshStep,stale_delivery_step:r.staleStep,critical_step:r.critical.step,
     critical_input_sha256:r.critical.input_sha256,critical_action:r.critical.action,
     last_arrival_state:r.critical.latest_arrival_state,highest_timestamp_state:r.critical.measurement_time_state,
     host_terminal_success:true,loss_masked_detour_steps:DELAY[split]});
   }
  }
  const file=split+(split==='TRAIN'?'_delayed_bulletin_stream.jsonl':'_EVALUATOR_ONLY_delayed_bulletin_stream.jsonl');
  write(root,file,shards,manifest.files);write(root,split+'_delayed_bulletin_certificates_EVALUATOR_ONLY.jsonl',certs,manifest.files);
  manifest.splits[split]={worlds:count,groups:count,episodes:shards.length,successful_host_episodes:shards.length,
   decisions,positive_pivotal_episodes:criticalPos,irrelevant_gate_controls:shards.length-criticalPos,
   masked_detour_decisions:maskedDetours,additional_duplicate_train_noncritical_masks:trainDuplicateMasked,
   long_delay_actions:DELAY[split],late_stale_delivery_step:lateStep(split),
   current_only_accuracy:criticalMemoryless/criticalPos,latest_arrival_accuracy:criticalArrival/criticalPos,
   highest_measurement_time_accuracy:criticalTimestamp/criticalPos,
   stale_last_actual_revisions:staleLastRevisions};total+=decisions;
 }
 manifest.total_decisions=total;manifest.total_episodes=episodes;manifest.distinct_world_geometries=owners.size;
 manifest.positive_pivotal_episodes=positives;manifest.irrelevant_gate_control_episodes=controls;
 manifest.stale_last_revision_failures_for_latest_arrival=staleFailures;
 fs.writeFileSync(path.join(root,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){check(process.argv[2],'usage: node delayed_bulletin_v64.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
module.exports={generate};
