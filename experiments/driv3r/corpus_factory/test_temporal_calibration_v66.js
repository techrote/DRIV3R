#!/usr/bin/env node
'use strict';
/* Independent v66 model-input transport auditor. Does not import generator.
 * Parent source host physics validated separately by v65/v64 host auditors. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,s)=>{if(!x)throw Error('V66 AUDIT '+s)};
const load=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const rows=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const name=(s,c)=>s+(c?(s==='TRAIN'?'_sensor_fault_stream.jsonl':'_EVALUATOR_ONLY_sensor_fault_stream.jsonl'):'_sensor_fault_certificates_EVALUATOR_ONLY.jsonl');
const target=(s,c)=>s+(c?(s==='TRAIN'?'_temporal_calibration_stream.jsonl':'_EVALUATOR_ONLY_temporal_calibration_stream.jsonl'):'_temporal_calibration_certificates_EVALUATOR_ONLY.jsonl');
const same=(x,y)=>JSON.stringify(x)===JSON.stringify(y);
function audit(out,parent){
 const m=load(path.join(out,'manifest.json')),p=load(path.join(parent,'manifest.json'));
 ok(m.schema==='driv3r/corpus_factory/temporal-calibration-v66/1'&&m.synthetic_not_native_CyberSand,'schema');
 ok(p.schema==='driv3r/corpus_factory/sensor-fault-v65/1'&&m.parent_v65_manifest_sha256===H(fs.readFileSync(path.join(parent,'manifest.json'))),'parent manifest');
 ok(m.generator_sha256===H(fs.readFileSync(path.join(__dirname,'temporal_calibration_v66.js'))),'generator source mismatch');
 for(const [fn,meta] of Object.entries(p.files))ok(H(fs.readFileSync(path.join(parent,fn)))===meta.sha256,'parent content hash '+fn);
 for(const [fn,meta] of Object.entries(m.files)){
  const b=fs.readFileSync(path.join(out,fn));ok(H(b)===meta.sha256&&b.length===meta.bytes&&rows(path.join(out,fn)).length===meta.rows,'output content hash '+fn);
 }
 const owners=new Map();let episodes=0,decisions=0,positive=0,firstHit=0,lastHit=0,chronHit=0,minGap=Infinity;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const base=rows(path.join(parent,name(split,true))),bc=rows(path.join(parent,name(split,false)));
  const policy=rows(path.join(out,target(split,true))),cert=rows(path.join(out,target(split,false)));
  ok(policy.length===base.length*2&&cert.length===policy.length,'counts');
  let hits=[0,0,0],pos=0,count=0;const groups=new Map();
  for(let i=0;i<policy.length;i++){
   const j=Math.floor(i/2),c=bc[j],q=cert[i],orig=base[j],e=policy[i],phase=i%2?'after':'before';
   ok(q.split===split&&q.episode===i&&q.parent_episode===j&&q.late_diagnostic_phase===phase&&c.episode===j,'certificate binding');
   ok(q.geometry_sha256===c.geometry_sha256&&q.host_terminal_success===c.host_terminal_success&&c.host_terminal_success,'host certificate');
   if(!owners.has(c.geometry_sha256))owners.set(c.geometry_sha256,split);
   ok(owners.get(c.geometry_sha256)===split,'cross-split geometry');
   const d=c.critical_step,r=c.revision_step,delta=Math.max(4,Math.floor(r/3)),step=phase==='before'?r-delta:r+delta;
   const report=orig.steps[r].input.events_delta[0],newReport=Number(report[4])^(phase==='before'?1:0);
   ok(report[1]==='EDGE_GATE'&&newReport===q.revised_gate_reported_state,'report identity');
   ok(q.critical_step===d&&q.critical_action===c.critical_action&&q.physical_revised_closed===Number(c.revised_closed),'critical certificate');
   ok(q.original_diagnostic_code===c.diagnostic_code&&q.late_diagnostic_code===(c.diagnostic_code===901?902:901),'diagnostic value');
   ok(q.late_diagnostic_step===step&&q.late_diagnostic_measurement_timestamp===report[0]+(phase==='before'?-1:1),'diagnostic chronology');
   ok(step>c.diagnostic_step+1&&step<d-8,'critical suffix gap');minGap=Math.min(minGap,d-step);
   ok(e.steps.length===orig.steps.length,'steps changed');
   for(let t=0;t<e.steps.length;t++){
    const expected=JSON.parse(JSON.stringify(orig.steps[t].input));
    if(t===r){ok(expected.events_delta.length===1,'report missing');expected.events_delta[0][4]=newReport;}
    if(t===step){ok(expected.events_delta.length===0,'diagnostic collision');expected.events_delta.push([report[0]+(phase==='before'?-1:1),'SENSOR_DIAGNOSTIC',q.late_diagnostic_code]);}
    const a=e.steps[t],b=orig.steps[t];
    ok(same(a.input,expected),'input discrepancy '+split+'/'+i+'/'+t);
    ok(a.action===b.action&&a.weight===b.weight&&a.focus_loss_weight===b.focus_loss_weight,'physical action or mask changed');
    count++;
   }
   ok(H(e.steps[d].input)===c.critical_input_sha256&&q.critical_input_sha256===c.critical_input_sha256,'critical current leaked');
   const bucket=groups.get(c.geometry_sha256)||{inputs:new Set(),actions:new Set(),episodes:0,positive:0};
   bucket.inputs.add(JSON.stringify(e.steps[d].input));bucket.actions.add(c.critical_action);bucket.episodes++;bucket.positive+=Number(c.kind==='positive');groups.set(c.geometry_sha256,bucket);
   episodes++;
   if(c.kind==='positive'){
    positive++;pos++;
    const truth=Number(c.revised_closed),reported=newReport;
    const first=Number((reported^(c.diagnostic_code===902))===truth);
    const latest=Number((reported^(q.late_diagnostic_code===902))===truth);
    const timed=Number((reported^((phase==='before'?q.late_diagnostic_code:c.diagnostic_code)===902))===truth);
    firstHit+=first;lastHit+=latest;chronHit+=timed;hits[0]+=first;hits[1]+=latest;hits[2]+=timed;
   }
  }
  ok(count===m.splits[split].decisions&&policy.length===m.splits[split].episodes,'split totals');
  for(const g of groups.values())ok(g.inputs.size===1&&g.actions.size===2&&g.episodes===64&&g.positive===32,'group counterfactual contrast or instant-observation leak');
  ok(hits[0]/pos===.5&&hits[1]/pos===.5&&hits[2]===pos,'split baseline contrast');
  decisions+=count;
 }
 ok(episodes===m.total_episodes&&decisions===m.total_decisions&&owners.size===m.distinct_world_geometries,'global totals');
 ok(firstHit*2===positive&&lastHit*2===positive&&chronHit===positive,'global baselines');
 const result={pass:true,synthetic_not_native_CyberSand:true,episodes,decisions,worlds:owners.size,
   positive_pivotal_cases:positive,first_diagnostic_accuracy:firstHit/positive,
   latest_diagnostic_accuracy:lastHit/positive,measurement_time_calibrated_accuracy:chronHit/positive,
   minimum_observation_gap_to_critical:minGap,source_host_replay:'requires separate v65/v64 source-host audit'};
 console.log(JSON.stringify(result));return result;
}
if(require.main===module){ok(process.argv[2]&&process.argv[3],'usage: node test_temporal_calibration_v66.js V66_DIR V65_DIR');audit(path.resolve(process.argv[2]),path.resolve(process.argv[3]));}
module.exports={audit};
