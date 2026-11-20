#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v65. Synthetic telemetry fault/calibration history;
 * truthful host state is retained, only observable sensor transport is perturbed.
 * DATA ONLY. No policy model trained. No native CyberSand claim. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const partial=require('./frontier_memory_v45.js');
const v63=require('./gate_revision_v63.js');
const v64=require('./delayed_bulletin_v64.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(x,msg)=>{if(!x)throw Error('V65 '+msg)};
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const splits=['TRAIN','DEV','HELDOUT'];
const parents=['unseen_topology_v44.js','frontier_memory_v45.js','offpolicy_repair_v58.js','gate_revision_v63.js','delayed_bulletin_v64.js'];
const STAMP={TRAIN:41000,DEV:51000,HELDOUT:61000};
const diagnostic={healthy:901,inverted:902};
// Genuine new source topologies. Never patch or alter prior lab source files.
function qualifiedNovelWorld(split,index){
 for(let trial=0;trial<120;trial++){
  const sourceIndex=STAMP[split]+index*137+trial;
  const w=host.scenario(split,sourceIndex);
  try{
   const p0=partial.prelude(w,'positive',false),p1=partial.prelude(w,'positive',true);
   const c0=partial.prelude(w,'control',false),c1=partial.prelude(w,'control',true);
   const a0=partial.teacher(partial.observe(p0)),a1=partial.teacher(partial.observe(p1));
   const b0=partial.teacher(partial.observe(c0)),b1=partial.teacher(partial.observe(c1));
   if(a0===a1||b0!==b1||a0!==b0)continue;
   const open=host.ranks(p0.g,p0.pos,w.goal,-1),closed=host.ranks(p1.g,p1.pos,w.goal,w.pivotal);
   if(a0!==open[0]?.action||a1!==closed[0]?.action||b0!==open[0]?.action)continue;
   const wrong0=open.find(z=>z.action===a1),wrong1=closed.find(z=>z.action===a0);
   if(!wrong0||!wrong1||wrong0.cost-open[0].cost<2||wrong1.cost-closed[0].cost<2)continue;
   if(new Set(p0.positions).size/p0.g.coords.length>.83)continue;
   return {w,sourceIndex,trial};
  }catch(e){if(e.message.includes('no qualified topology'))throw e;}
 }
 throw Error('V65 cannot qualify distinct world '+split+'/'+index);
}
function dataName(split){return split+(split==='TRAIN'?'_sensor_fault_stream.jsonl':'_EVALUATOR_ONLY_sensor_fault_stream.jsonl');}
function baseName(split){return split+(split==='TRAIN'?'_delayed_bulletin_stream.jsonl':'_EVALUATOR_ONLY_delayed_bulletin_stream.jsonl');}
function certificateName(split){return split+'_sensor_fault_certificates_EVALUATOR_ONLY.jsonl';}
function write(out,name,rows,manifest){const b=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});manifest.files[name]={rows:rows.length,bytes:b.length,sha256:H(b)};}
function transmit(base,cert,fault){
 const d=cert.critical_step,signalStep=8,correctCode=fault?diagnostic.inverted:diagnostic.healthy;
 check(d>signalStep+1&&d%2===0,'bad detour calibration period');
 const steps=base.steps.map(row=>({input:JSON.parse(JSON.stringify(row.input)),action:row.action,weight:row.weight,focus_loss_weight:row.focus_loss_weight}));
 const at=steps[d/2].input.events_delta;
 check(at.length===1&&at[0][1]==='EDGE_GATE'&&at[0][4]===Number(cert.revised_closed),'physical fresh receipt missing');
 at[0][4]^=Number(fault); // communication-layer inverted sensor, not host gate mutation
 check(steps[signalStep].input.events_delta.length===0,'diagnostic collided with other host event');
 // A deliberately early reliability/calibration bulletin for the remote sensor.
 // Its elapsed time is host-accepted action count, not future information.
 const first=steps[0].input.events_delta.filter(e=>e[1]==='EDGE_GATE').at(-1);
 check(first,'initial diagnostic clock missing');
 steps[signalStep].input.events_delta.push([first[0]+signalStep,'SENSOR_DIAGNOSTIC',correctCode]);
 check(steps[d].input.events_delta.length===0,'instantaneous decision state leaked');
 return {steps};
}
function generate(out,{quick=false}={}){
 check(!fs.existsSync(out),'target already exists');
 const scratch=out+'.v64_source';check(!fs.existsSync(scratch),'scratch exists');
 const old=v63.qualifiedWorld;
 try{v63.qualifiedWorld=qualifiedNovelWorld;v64.generate(scratch,{quick});}
 finally{v63.qualifiedWorld=old;}
 fs.mkdirSync(out,{recursive:true});
 const base=JSON.parse(fs.readFileSync(path.join(scratch,'manifest.json')));
 const mf={schema:'driv3r/corpus_factory/sensor-fault-v65/1',date:'2026-11-19',quick,
  synthetic_not_native_CyberSand:true,generator_sha256:H(fs.readFileSync(__filename)),
  parent_sha256:Object.fromEntries(parents.map(n=>[n,H(fs.readFileSync(path.join(__dirname,n)))])),
  base_generator_manifest_sha256:H(fs.readFileSync(path.join(scratch,'manifest.json'))),
  scenario:'Counterfactual physical remote-gate state plus out-of-order stale authentic reports, and a synthetic observable telemetry channel that sometimes inverts a later sensor reading. Early SENSOR_DIAGNOSTIC 901/902 identifies honest/inverted channel. True physical gate remains host-executed; immediate critical input does not expose fault mode.',
  contract:'Only steps.input and prior actions reach model. No source certificates, full durable ledger, true gate, diagnostic arms or manifest metadata. Synthetic diagnostic codes 901=healthy; 902=inverted (not native CyberSand codes). Only post-sensor transport is modified; pre-action host-visible feedback otherwise identical.',
  topology:'New deterministic source indices in 41000/51000/61000 ranges; disjoint from v63/v64 24000 range. Each split has separate graph sizes/seeds, checked by geometry sha256.',
  limitation:'The fault-correction rule is still solvable by a symbolic historical diagnostic and measurement-time parser. Oracle action traces and scripted detours do not prove neural learning; no native CyberSand sensor semantics or host evaluation.',
  controls:'At positive critical pairs, current-only 50%; highest-timestamp without fault correction 50%; earliest diagnostic plus highest timestamp corrected 100%. Measured host-physics labels are identical for both telemetry modes.',
  source_base_manifest:base.schema,files:{},splits:{}};
 const owners=new Map();let all=0,totalEps=0,positive=0,correctTime=0,correctTimeFaultAware=0,currentOnly=0;
 for(const split of splits){
  const rows=read(path.join(scratch,baseName(split))),certs=read(path.join(scratch,split+'_delayed_bulletin_certificates_EVALUATOR_ONLY.jsonl'));
  check(rows.length===certs.length&&rows.length%16===0,'bad base input split '+split);
  const upgraded=[],outputs=[],seen=new Set();let decisions=0,weightedCritical=0,masked=0,splitPositive=0,rawRight=0,corrected=0,baseline=0,groupCount=0;
  for(let j=0;j<rows.length;j+=16){
   const group=certs.slice(j,j+16),w=host.scenario(split,group[0].source_world_index);
   check(w.geometry_sha256===group[0].geometry_sha256,'incorrect source identity');
   check(!owners.has(w.geometry_sha256),'cross-world geometry duplicate');owners.set(w.geometry_sha256,split);groupCount++;
   const start=outputs.length;
   for(let k=0;k<16;k++)for(const fault of [false,true]){
    const c=group[k],ep=transmit(rows[j+k],c,fault),idx=outputs.length,at=ep.steps[c.critical_step].input;
    check(H(at)===c.critical_input_sha256,'critical observation changed');
    const reported=ep.steps[c.revision_step].input.events_delta[0][4];
    check(Boolean(reported)===Boolean(c.revised_closed)!==fault,'fault transmission mismatch');
    if(split==='TRAIN')for(let t=0;t<ep.steps.length;t++){
     if(t===c.critical_step||ep.steps[t].weight===0)continue;
     const s=H([ep.steps[t].input,ep.steps[t].action]);
     if(seen.has(s)){ep.steps[t].weight=0;ep.steps[t].focus_loss_weight=0;masked++;}else seen.add(s);
    }
    check(ep.steps[c.critical_step].weight===1,'critical supervised label removed');
    const sample={...c,episode:idx,original_base_episode:j+k,group:split+':'+(j/16),telemetry_mode:fault?'inverted':'healthy',diagnostic_step:8,diagnostic_code:fault?diagnostic.inverted:diagnostic.healthy,
      reported_new_state:reported,host_true_new_state:Number(c.revised_closed),transport_only_inversion:fault,
      critical_input_sha256:H(at),source_base_stream_sha256:base.files[baseName(split)].sha256};
    upgraded.push(ep);outputs.push(sample);decisions+=ep.steps.length;weightedCritical++;
    if(c.kind==='positive'){
     splitPositive++;positive++;
     rawRight+=Number(Boolean(reported)===Boolean(c.revised_closed));
     correctTime+=Number(Boolean(reported)===Boolean(c.revised_closed));
     correctTimeFaultAware++;corrected++;
     baseline+=Number(c.critical_action===group[0].critical_action);
    }
   }
   check(outputs.length-start===32,'group size');
   const current=upgraded.slice(start).map(x=>x.steps[group[0].critical_step].input);
   check(current.every(x=>JSON.stringify(x)===JSON.stringify(current[0])),'critical current observation mismatch');
  }
  write(out,dataName(split),upgraded,mf);write(out,certificateName(split),outputs,mf);
  mf.splits[split]={worlds:groupCount,groups:groupCount,arm_count:32,episodes:upgraded.length,host_successes:upgraded.length,
   decisions,positive_pivotal_episodes:splitPositive,irrelevant_controls:upgraded.length-splitPositive,
   critical_supervised_labels:weightedCritical,masked_duplicate_train_noncritical:masked,
   current_only_accuracy:baseline/splitPositive,raw_highest_timestamp_accuracy:rawRight/splitPositive,
   calibration_corrected_accuracy:corrected/splitPositive};
  all+=decisions;totalEps+=upgraded.length;currentOnly+=baseline;
 }
 mf.total_episodes=totalEps;mf.total_decisions=all;mf.distinct_world_geometries=owners.size;
 mf.positive_pivotal_episodes=positive;
 mf.raw_highest_timestamp_accuracy=correctTime/positive;
 mf.calibrated_timestamp_accuracy=correctTimeFaultAware/positive;
 mf.current_only_accuracy=currentOnly/positive;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});
 return {manifest:mf,scratch};
}
if(require.main===module){check(process.argv[2],'usage: node sensor_fault_v65.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).manifest.splits));}
module.exports={generate,transmit,qualifiedNovelWorld,diagnostic};
