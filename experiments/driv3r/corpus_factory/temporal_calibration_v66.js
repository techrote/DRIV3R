#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v66: when did the sensor reconfigure relative to the
 * physical measurement? Synthetic observation transport only; DATA, not policy. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,s)=>{if(!x)throw Error('V66 '+s)};
const splits=['TRAIN','DEV','HELDOUT'];
const name=(s,k)=>s+(k==='stream'?(s==='TRAIN'?'_temporal_calibration_stream.jsonl':'_EVALUATOR_ONLY_temporal_calibration_stream.jsonl'):'_temporal_calibration_certificates_EVALUATOR_ONLY.jsonl');
const parent=(s,k)=>s+(k==='stream'?(s==='TRAIN'?'_sensor_fault_stream.jsonl':'_EVALUATOR_ONLY_sensor_fault_stream.jsonl'):'_sensor_fault_certificates_EVALUATOR_ONLY.jsonl');
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
const shaFile=p=>H(fs.readFileSync(p));
function transform(source,cert,phase){
 ok(phase==='before'||phase==='after','phase');
 const d=cert.critical_step,r=cert.revision_step,step=phase==='before'?r-Math.max(4,Math.floor(r/3)):r+Math.max(4,Math.floor(r/3));
 ok(step>cert.diagnostic_step+1&&step<d-8,'calibration order');
 const e={steps:source.steps.map(x=>({input:JSON.parse(JSON.stringify(x.input)),action:x.action,weight:x.weight,focus_loss_weight:x.focus_loss_weight}))};
 const evt=e.steps[r].input.events_delta;ok(evt.length===1&&evt[0][1]==='EDGE_GATE','original revised observation');
 const newestCode=cert.diagnostic_code===901?902:901;
 const originalReport=evt[0][4],reportTime=evt[0][0];
 if(phase==='before')evt[0][4]^=1;
 ok(e.steps[step].input.events_delta.length===0,'calibration collision');
 const stamp=phase==='before'?reportTime-1:reportTime+1;
 e.steps[step].input.events_delta.push([stamp,'SENSOR_DIAGNOSTIC',newestCode]);
 ok(e.steps[d].input.events_delta.length===0,'critical input leaked');
 ok(e.steps[d].weight===1,'critical label masked');
 return {episode:e,calibration_step:step,calibration_stamp:stamp,revision_report_time:reportTime,
  revision_reported_state:evt[0][4],original_reported_state:originalReport,initial_code:cert.diagnostic_code,latest_code:newestCode};
}
function generate(out,input){
 ok(!fs.existsSync(out),'destination exists');
 const pm=JSON.parse(fs.readFileSync(path.join(input,'manifest.json'),'utf8'));
 ok(pm.schema==='driv3r/corpus_factory/sensor-fault-v65/1'&&pm.synthetic_not_native_CyberSand,'wrong parent');
 for(const [f,m] of Object.entries(pm.files))ok(shaFile(path.join(input,f))===m.sha256,'parent hash '+f);
 fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/temporal-calibration-v66/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,
  generator_sha256:shaFile(__filename),parent_v65_manifest_sha256:shaFile(path.join(input,'manifest.json')),
  parent_v65_generator_sha256:pm.generator_sha256,parent_dataset_file_sha256:Object.fromEntries(Object.entries(pm.files).map(([n,m])=>[n,m.sha256])),
  physical_host:'Unmodified v65 source host decisions; only telemetered revised EDGE_GATE state and post-diagnostic event are transformed. No additional host action or success measurement.',
  purpose:'Calibrated sensor changes before or after a physical measurement. The earliest diagnostic and the latest diagnostic EACH give only 50% pivotal accuracy; choose the diagnostic effective at measurement time.',
  consumer_contract:'TRAIN observation/action input only. DEV/HELDOUT streams and ALL certificates evaluator-only; no group/phase/true-state metadata in model inputs. Code 901/902 is synthetic, NOT native CyberSand ABI.',
  topology:'Exact v65 geometry reused with matching split assignments; not independent cross-version world generalization. No TRAIN/DEV/HELDOUT world overlap.',
  limitations:'Synthetic host; scripted detours excluded from loss; no model trained. Measurement-time rule remains solvable symbolically. Full native CyberSand interpretation unsupported.',
  files:{},splits:{}};
 let total=0,positive=0,oldHit=0,lastHit=0,chronHit=0,decisions=0;
 const owners=new Map();
 for(const split of splits){
  const rows=read(path.join(input,parent(split,'stream'))),certs=read(path.join(input,parent(split,'cert')));
  ok(rows.length===certs.length&&rows.length%32===0,'base cardinality '+split);
  const generated=[],proof=[];let splitPos=0,splitOld=0,splitLast=0,splitChron=0,splitSteps=0;
  for(let i=0;i<rows.length;i++){
   const source=rows[i],c=certs[i];
   ok(c.split===split&&c.episode===i&&c.host_terminal_success,'parent certificate mismatch');
   if(!owners.has(c.geometry_sha256))owners.set(c.geometry_sha256,split);
   ok(owners.get(c.geometry_sha256)===split,'split geometry leakage');
   for(const phase of ['before','after']){
    const z=transform(source,c,phase),idx=generated.length,t=c.critical_step;
    generated.push(z.episode);splitSteps+=z.episode.steps.length;
    const effective=phase==='before'?z.latest_code:z.initial_code;
    const physical=Number(Boolean(z.revision_reported_state)^(effective===902));
    ok(physical===Number(c.revised_closed),'physical model-report inconsistency');
    ok(H(z.episode.steps[t].input)===c.critical_input_sha256,'current observation changed');
    proof.push({split,episode:idx,parent_episode:i,geometry_sha256:c.geometry_sha256,source_world_index:c.source_world_index,
     host_terminal_success:c.host_terminal_success,kind:c.kind,critical_step:t,critical_action:c.critical_action,
     critical_input_sha256:c.critical_input_sha256,physical_revised_closed:Number(c.revised_closed),
     original_diagnostic_code:z.initial_code,late_diagnostic_code:z.latest_code,late_diagnostic_phase:phase,
     late_diagnostic_step:z.calibration_step,late_diagnostic_measurement_timestamp:z.calibration_stamp,
     revised_gate_measurement_timestamp:z.revision_report_time,revised_gate_reported_state:z.revision_reported_state,
     last_diagnostic_at_decision_is_stale:phase==='after',parent_v65_manifest_sha256:manifest.parent_v65_manifest_sha256});
    if(c.kind==='positive'){
     splitPos++;positive++;
     const correct=Number(c.revised_closed);
     splitOld+=Number(Number(Boolean(z.revision_reported_state)^(z.initial_code===902))===correct);
     splitLast+=Number(Number(Boolean(z.revision_reported_state)^(z.latest_code===902))===correct);
     splitChron+=Number(physical===correct);
    }
   }
  }
  const fn=name(split,'stream'),cn=name(split,'certificate');
  for(const [n,content] of [[fn,generated],[cn,proof]]){
   const buf=Buffer.from(content.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,n),buf,{flag:'wx'});
   manifest.files[n]={rows:content.length,bytes:buf.length,sha256:H(buf)};
  }
  manifest.splits[split]={worlds:new Set(certs.map(c=>c.geometry_sha256)).size,episodes:generated.length,
   host_certified_successful_episodes:generated.length,decisions:splitSteps,positive_pivotal_cases:splitPos,
   first_diagnostic_accuracy:splitOld/splitPos,latest_diagnostic_accuracy:splitLast/splitPos,
   measurement_time_calibrated_accuracy:splitChron/splitPos};
  total+=generated.length;decisions+=splitSteps;oldHit+=splitOld;lastHit+=splitLast;chronHit+=splitChron;
 }
 ok(oldHit*2===positive&&lastHit*2===positive&&chronHit===positive,'controls unbalanced');
 manifest.total_episodes=total;manifest.total_decisions=decisions;manifest.distinct_world_geometries=owners.size;
 manifest.positive_pivotal_cases=positive;manifest.first_diagnostic_accuracy=oldHit/positive;
 manifest.latest_diagnostic_accuracy=lastHit/positive;manifest.measurement_time_calibrated_accuracy=chronHit/positive;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
if(require.main===module){ok(process.argv[2]&&process.argv[3],'usage: node temporal_calibration_v66.js NEW_DIR V65_DIR');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),path.resolve(process.argv[3])).splits));}
module.exports={generate,transform,name,parent};
