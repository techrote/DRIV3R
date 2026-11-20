#!/usr/bin/env node
'use strict';
/* Independent synthetic host replay for Corpus Factory v62. Does NOT import
 * relevance_gate_v62.js or rely on certificates to approve action legality. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v58=require('./offpolicy_repair_v58.js'),v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const ok=(x,msg)=>{if(!x)throw Error('V62 AUDIT '+msg)};
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
const cache=new Map();
function world(s,ix){const k=s+':'+ix;if(!cache.has(k))cache.set(k,v46.candidate(s,700+ix).w);return cache.get(k);}
function replay(split,cert,episode){
 const w=world(split,cert.world_index),arm=v53.ARMS[cert.old_swapped?2:1],pars=v53.armParams(arm),h=v53.prep(w,arm);
 ok(w.seed===cert.seed&&w.geometry_sha256===cert.geometry_sha256,'world provenance disagreement');
 let prior=[],firstErased=false,movesSinceFirst=0,secondAlert=false,lateInjected=false,critical=null,blind=null;
 const probes=[];let terminalGoal=null,maskedDetour=0,materialWarnings=0;
 for(let t=0;t<episode.steps.length;t++){
  const row=episode.steps[t],n=v52.notice(h,pars);v53.injectB(h,arm,n,t);
  if(firstErased&&movesSinceFirst>=1&&!secondAlert&&h.stage===1&&h.pos!==h.siteB){
   ok(h.painted[h.siteB]===undefined,'old physical probe not erased');
   h.laws.B=cert.mid_swapped?'SWAPPED':'NORMAL';
   h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
   secondAlert=true;
  }
  if(t===cert.critical_step-cert.forced_detour_actions){
   ok(probes.length===2&&h.pos===w.goal2&&h.stage===1&&!lateInjected,'late event location/age/probe count');
   const cond=cert.condition;
   if(cond==='irrelevant_A')h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'A',site:h.g.coords[w.s]});
   if(cond==='realert_B_stable'||cond==='realert_B_flip'){
    if(cond==='realert_B_flip')h.laws.B=h.laws.B==='NORMAL'?'SWAPPED':'NORMAL';
    h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
    materialWarnings++;
   }
   lateInjected=true;
  }
  const obs=v53.observe(h),input=v58.streamify(obs,prior);
  ok(eq(input,row.input),'host observation mismatch '+split+' ep '+cert.episode+' action '+t);
  ok(obs.legal.includes(row.action),'illegal command '+row.action);
  ok(!('durable' in input)&&!('laws' in input)&&!('mid_swapped' in input)&&!('condition' in input),'hidden experiment information exposed');
  if(t>=cert.critical_step-cert.forced_detour_actions&&t<cert.critical_step){
   ok(row.weight===0&&row.focus_loss_weight===0&&row.action.startsWith('MOVE_'),'detour not zero-weighted legal movement');maskedDetour++;
  }
  if(t===cert.critical_step){
   ok(h.pos===w.goal2&&h.stage===1&&probes.length===2,'critical checkpoint misplaced');
   ok(row.weight===1&&row.focus_loss_weight===8,'critical supervision absent');
   ok(!input.events_delta.some(e=>e[1]==='SURFACE_ALERT'||e[1]==='PAINT_RESULT'),'old alert or receipt leaked at checkpoint');
   ok(H(input)===cert.critical_input_sha256,'critical input hash');
   ok(row.action===cert.critical_action,'critical action certificate');
   ok((row.action.startsWith('MOVE_'))===(cert.condition==='realert_B_stable'||cert.condition==='realert_B_flip'),'alert vs non-alert action mismatch');
   // Independent physically executed causal fork: ignoring late B warning and
   // painting the last measured law at the protected B goal is only reliable
   // when the law has not changed. No label in the training stream uses this fork.
   const staleLawPaint=cert.mid_swapped?'PAINT_A':'PAINT_B';
   const alternate=structuredClone(h);
   ok(obs.legal.includes(staleLawPaint),'blind last-receipt paint illegal');
   const receipt=v52.command(alternate,staleLawPaint,pars);
   blind={material:receipt.material,stage:alternate.stage,progressed:alternate.stage===2,success:!!alternate.success};
   if(cert.condition==='realert_B_flip')ok(!blind.progressed&&receipt.material==='A','blind stale paint should fail after drift');
   else ok(blind.progressed&&receipt.material==='B','blind paint should physically succeed without B drift');
   critical={input,action:row.action};
  }
  prior=obs.durable;const pre=h.stage,at=h.pos,result=v52.command(h,row.action,pars);
  if(pre===1&&at===h.siteB&&row.action==='PAINT_A'){
   ok(result.material==='A'||result.material==='B','physical probe did not return valid material');
   probes.push({index:t,material:result.material});
  }
  if(pre===1&&at===h.siteB&&row.action==='ERASE'&&probes.length===1)firstErased=true;
  if(firstErased&&!secondAlert&&row.action.startsWith('MOVE_'))movesSinceFirst++;
  if(pre===1&&at===w.goal2&&row.action.startsWith('PAINT_')){
   ok(result.material==='B'&&h.stage===2,'protected goal not physically completed');terminalGoal={action:row.action,index:t};
  }
 }
 ok(h.done&&h.success&&!h.protected_damage,'host did not terminate successfully');
 ok(secondAlert&&lateInjected&&critical&&blind&&terminalGoal,'missing completed challenge');
 ok(maskedDetour===cert.forced_detour_actions,'forced detour count mismatch');
 ok(materialWarnings===Number(cert.condition.startsWith('realert_B')),'late warnings mismatch');
 ok(probes.length===(materialWarnings?3:2),'physical re-probe count not controlled');
 ok(eq(probes.map(p=>p.material),cert.probe_materials)&&eq(probes.map(p=>p.index),cert.probe_steps),'probe outcomes not matching host');
 ok(terminalGoal.action===cert.final_action,'terminal paint mismatch');
 return {critical,blind,probes:probes.map(p=>p.material),terminal:terminalGoal.action};
}
function audit(dir){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 ok(mf.schema==='driv3r/corpus_factory/relevance-gated-memory-v62/1','schema');
 for(const [f,m] of Object.entries(mf.files)){
  const b=fs.readFileSync(path.join(dir,f));ok(H(b)===m.sha256&&b.length===m.bytes&&read(path.join(dir,f)).length===m.rows,'shard integrity '+f);
 }
 const owners=new Map();let sets=0,episodes=0,decisions=0,lateB=0,irrelevantA=0,blindError=0,oldProbeWrong=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const name=split+(split==='TRAIN'?'_relevance_stream.jsonl':'_EVALUATOR_ONLY_relevance_stream.jsonl');
  const data=read(path.join(dir,name)),certs=read(path.join(dir,split+'_relevance_certificates_EVALUATOR_ONLY.jsonl'));
  ok(data.length===certs.length&&data.length===mf.splits[split].episodes&&data.length%16===0,'episode count mismatch');
  let splitDecisions=0;
  for(let i=0;i<data.length;i+=16){
   const sixteen=certs.slice(i,i+16),answers=[];
   for(let j=0;j<16;j++){
    const c=sixteen[j],expectedOld=j>=8,expectedMid=(j%8)>=4;
    const expectedCondition=['none','irrelevant_A','realert_B_stable','realert_B_flip'][j%4];
    ok(c.episode===i+j&&c.split===split&&c.group_id===sixteen[0].group_id,'group sequence');
    ok(c.old_swapped===expectedOld&&c.mid_swapped===expectedMid&&c.condition===expectedCondition,'not fully balanced 16-arm design');
    const owner=owners.get(c.geometry_sha256);ok(!owner||owner===split,'world split leaked');owners.set(c.geometry_sha256,split);
    const replayed=replay(split,c,data[i+j]);answers.push(replayed);splitDecisions+=data[i+j].steps.length;
    if(c.condition.startsWith('realert_B')){lateB++;if(!replayed.blind.progressed)blindError++;}
    if(c.condition==='irrelevant_A')irrelevantA++;
    if(replayed.probes[0]!==replayed.probes[1])oldProbeWrong++;
   }
   ok(answers.every(z=>eq(z.critical.input,answers[0].critical.input)),'current observation differs across 16-arm matched group');
   const decisionsFor=(cond)=>answers.filter((_,j)=>sixteen[j].condition===cond).map(z=>z.critical.action);
   ok(decisionsFor('none').every((a,j)=>a===decisionsFor('irrelevant_A')[j]),'unrelated surface alert changed policy action');
   ok(decisionsFor('realert_B_stable').every((a,j)=>a===decisionsFor('realert_B_flip')[j]&&a.startsWith('MOVE_')),'late B warning choices not matched despite hidden drift');
   const common=answers.map(z=>z.critical.action);
   ok(common.filter(a=>a.startsWith('MOVE_')).length===8&&common.filter(a=>a==='PAINT_A').length===4&&common.filter(a=>a==='PAINT_B').length===4,'not balanced memory contrast');
   // Hidden post-warning B law cannot be known without the third safe probe.
   for(let j=0;j<16;j+=4){
    ok(answers[j+2].blind.progressed&&!answers[j+3].blind.progressed,'stable/flip blind causal pair not distinguished');
    ok(answers[j+2].terminal!==answers[j+3].terminal,'third physical outcome did not change final command');
   }
   sets++;
  }
  ok(splitDecisions===mf.splits[split].decisions,'split decisions mismatch');episodes+=data.length;decisions+=splitDecisions;
 }
 ok(decisions===mf.total_decisions&&sets===mf.total_sets&&owners.size===mf.distinct_world_geometries,'global counts');
 ok(lateB===episodes/2&&irrelevantA===episodes/4&&blindError===episodes/4&&oldProbeWrong===episodes/2,'negative/positive balance mismatch');
 const result={pass:true,synthetic_host_only:true,independent_full_host_replay:true,
  sets,episodes,decisions,world_geometries:owners.size,post_B_alert_episodes:lateB,irrelevant_A_alert_episodes:irrelevantA,
  blind_stale_paint_physical_failures:blindError,misleading_older_B_probe_episodes:oldProbeWrong,
  critical_actions:{reprobe:episodes/2,paint_A:episodes/4,paint_B:episodes/4},
  memoryless_majority_critical_ceiling:0.5,oracle_accuracy_not_neural:1.0};
 console.log(JSON.stringify(result));return result;
}
if(require.main===module){ok(process.argv[2],'usage: node test_relevance_gate_host_v62.js DATA_DIR');audit(process.argv[2]);}
module.exports={audit};
