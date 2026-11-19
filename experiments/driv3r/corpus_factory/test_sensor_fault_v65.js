#!/usr/bin/env node
'use strict';
/** Independent v65 transport audit: source host is replayed by v64 auditor,
 * then verify calibration-corrupt observations without importing v65 generator.
 * Model sees input-only streams; evaluator certificates remain sealed. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const parentAudit=require('./test_delayed_bulletin_host_v64.js');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,s)=>{if(!x)throw Error('V65 AUDIT '+s)};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const load=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const rows=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
const SPLITS=['TRAIN','DEV','HELDOUT'];
function file(split,kind){return split+(kind==='policy'?(split==='TRAIN'?'_sensor_fault_stream.jsonl':'_EVALUATOR_ONLY_sensor_fault_stream.jsonl'):'_sensor_fault_certificates_EVALUATOR_ONLY.jsonl');}
function baseFile(split){return split+(split==='TRAIN'?'_delayed_bulletin_stream.jsonl':'_EVALUATOR_ONLY_delayed_bulletin_stream.jsonl');}
function audit(out,baseDir,priorDir){
 const m=load(path.join(out,'manifest.json')),bm=load(path.join(baseDir,'manifest.json'));
 ok(m.schema==='driv3r/corpus_factory/sensor-fault-v65/1'&&m.synthetic_not_native_CyberSand,'schema or host claim');
 ok(m.base_generator_manifest_sha256===H(fs.readFileSync(path.join(baseDir,'manifest.json'))),'source base manifest mutated');
 const sourceProof=parentAudit.audit(baseDir);
 for(const [name,meta] of Object.entries(m.files)){
  const b=fs.readFileSync(path.join(out,name));ok(H(b)===meta.sha256&&b.length===meta.bytes&&rows(path.join(out,name)).length===meta.rows,'bad file '+name);
 }
 const prior=new Set();if(priorDir)for(const split of SPLITS){
  const p=path.join(priorDir,split+'_delayed_bulletin_certificates_EVALUATOR_ONLY.jsonl');
  for(const c of rows(p))prior.add(c.geometry_sha256);
 }
 const worlds=new Set();let episodes=0,decisions=0,positive=0,rawCorrect=0,corrected=0,currentCorrect=0,wrongTurnMin=Infinity;
 const checkCodes={901:0,902:1};
 for(const split of SPLITS){
  const s=m.splits[split],policy=rows(path.join(out,file(split,'policy'))),cert=rows(path.join(out,file(split,'cert')));
  const original=rows(path.join(baseDir,baseFile(split))),originCert=rows(path.join(baseDir,split+'_delayed_bulletin_certificates_EVALUATOR_ONLY.jsonl'));
  ok(policy.length===original.length*2&&cert.length===policy.length&&policy.length%32===0,'arm or episode cardinality');
  let splitDec=0,splitPos=0,splitRaw=0,splitCorrect=0,splitCurrent=0;
  for(let base=0;base<policy.length;base+=32){
   const source=originCert[base/2],world=host.scenario(split,source.source_world_index),sha=source.geometry_sha256;
   ok(world.geometry_sha256===sha&&source.source_world_index>=(split==='TRAIN'?41000:split==='DEV'?51000:61000),'wrong source geometry range');
   ok(!worlds.has(sha),'duplicate geometry across worlds or splits');worlds.add(sha);
   ok(!prior.has(sha),'leakage against prior v64 topology');
   const critical=policy[base].steps[cert[base].critical_step].input;
   for(let k=0;k<32;k++){
    const e=policy[base+k],c=cert[base+k],originalIndex=base/2+Math.floor(k/2),b=original[originalIndex],bc=originCert[originalIndex];
    const fault=k%2,code=fault?902:901,d=c.critical_step;
    ok(e.steps.length===b.steps.length&&d===bc.critical_step&&c.episode===base+k&&c.original_base_episode===originalIndex,'episode mapping');
    ok(c.geometry_sha256===sha&&c.telemetry_mode===(fault?'inverted':'healthy')&&c.host_true_new_state===Number(c.revised_closed),'mode or truth');
    ok(c.diagnostic_code===code&&c.diagnostic_step===8&&c.source_base_stream_sha256===bm.files[baseFile(split)].sha256,'source binding or diagnostic certificate');
    ok(eq(e.steps[d].input,critical)&&H(critical)===c.critical_input_sha256,'current observation mismatch');
    let diagnosticCount=0,editedCount=0;
    for(let t=0;t<e.steps.length;t++){
     const row=e.steps[t],truth=b.steps[t],expected=JSON.parse(JSON.stringify(truth.input));
     if(t===8){const initial=b.steps[0].input.events_delta.filter(x=>x[1]==='EDGE_GATE').at(-1);
      ok(expected.events_delta.length===0,'diagnostic merged other packet');
      expected.events_delta.push([initial[0]+8,'SENSOR_DIAGNOSTIC',code]);diagnosticCount++;
     }
     if(t===d/2){ok(expected.events_delta.length===1&&expected.events_delta[0][1]==='EDGE_GATE','missing new report');expected.events_delta[0][4]^=fault;editedCount++;}
     ok(eq(row.input,expected),'model-visible telemetry not as certified split '+split+' episode '+(base+k)+' t '+t);
     ok(row.action===truth.action,'physical oracle trajectory changed');
     if(split!=='TRAIN'||row.weight===1)ok(row.weight===truth.weight&&row.focus_loss_weight===truth.focus_loss_weight,'illegal mask elevation');
     if(t===d){ok(row.weight===1&&row.focus_loss_weight===8,'critical label masked');}
     if(t<d)ok(row.weight===0&&row.focus_loss_weight===0,'synthetic detour trained');
     for(const x of row.input.events_delta||[])ok(x[1]!=='SENSOR_DIAGNOSTIC'||t===8||x[2]!==code,'extra diagnostic leak');
    }
    ok(diagnosticCount===1&&editedCount===1&&c.host_terminal_success,'incorrect diagnostic/physics');
    splitDec+=e.steps.length;episodes++;
    if(c.kind==='positive'){
     positive++;splitPos++;const report=e.steps[d/2].input.events_delta[0][4];
     rawCorrect+=Number(Boolean(report)===c.revised_closed);splitRaw+=Number(Boolean(report)===c.revised_closed);
     const inferred=Boolean(report)^Boolean(code===902);
     corrected+=Number(Number(inferred)===Number(c.revised_closed));splitCorrect+=Number(Number(inferred)===Number(c.revised_closed));
     const currentRight=c.critical_action===originCert[base/2].critical_action;
     currentCorrect+=Number(currentRight);splitCurrent+=Number(currentRight);
    }
   }
   const q=cert.slice(base,base+32);const p=q.filter(x=>x.kind==='positive');
   ok(p.length===16&&p.some(x=>!x.revised_closed)&&p.some(x=>x.revised_closed),'missing two real physical branches');
   ok(new Set(p.map(x=>x.critical_action)).size===2,'counterfactual labels collapsed');
   const open=host.ranks(host.hydrate(world),world.s,world.goal,-1),closed=host.ranks(host.hydrate(world),world.s,world.goal,world.pivotal);
   const a=p.find(x=>!x.revised_closed).critical_action,b=p.find(x=>x.revised_closed).critical_action;
   const regret0=open.find(x=>x.action===b)?.cost-open[0].cost,regret1=closed.find(x=>x.action===a)?.cost-closed[0].cost;
   ok(regret0>=2&&regret1>=2,'no meaningful wrong route penalty');wrongTurnMin=Math.min(wrongTurnMin,regret0,regret1);
  }
  ok(splitDec===s.decisions&&policy.length===s.episodes&&splitPos===s.positive_pivotal_episodes,'split metrics');
  ok(splitRaw/splitPos===s.raw_highest_timestamp_accuracy&&splitCorrect/splitPos===s.calibration_corrected_accuracy&&splitCurrent/splitPos===s.current_only_accuracy,'split baselines');
  decisions+=splitDec;
 }
 ok(episodes===m.total_episodes&&decisions===m.total_decisions&&worlds.size===m.distinct_world_geometries,'aggregate totals');
 ok(rawCorrect===positive/2&&corrected===positive&&currentCorrect===positive/2,'controls must have 50%,100%,50%');
 const verdict={pass:true,synthetic_not_native_CyberSand:true,source_host_independent_audit:sourceProof.pass,
  source_host_episodes:sourceProof.episodes,episodes,decisions,positive_pivotal_cases:positive,distinct_new_topologies:worlds.size,
  prior_topology_collision:0,minimum_wrong_route_regret_moves:wrongTurnMin,
  current_only_accuracy:currentCorrect/positive,raw_highest_timestamp_accuracy:rawCorrect/positive,diagnostic_calibrated_accuracy:corrected/positive};
 console.log(JSON.stringify(verdict));return verdict;
}
if(require.main===module){ok(process.argv[2]&&process.argv[3],'usage: node test_sensor_fault_v65.js V65_DIR V64_SOURCE_DIR [PRIOR_V64_DIR]');audit(...process.argv.slice(2).map(p=>path.resolve(p)));}
module.exports={audit};
