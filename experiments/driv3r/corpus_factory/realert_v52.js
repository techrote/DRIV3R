'use strict';
/* DRIV3R Corpus Factory v52 — temporal invalidation of safe physical evidence.
 * DATA ONLY. Synthetic host events are NOT native CyberSand observations.
 * Each second warning occurs DURING the return from a host-executed safe probe.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v51=require('./active_probe_v51.js');
const v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,msg)=>{if(!x)throw Error('V52 '+msg)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:32,DEV:12,HELDOUT:12};
const ARMS=[
 {id:'00_no_warning',A:'NORMAL',B:'NORMAL'},
 {id:'01_A_once',A:'NORMAL',B:'NORMAL',alertA:true},
 {id:'02_A_twice_same',A:'NORMAL',B:'NORMAL',alertA:true,again:'A'},
 {id:'03_A_twice_changed',A:'NORMAL',B:'NORMAL',alertA:true,again:'A',againFlip:true},
 {id:'04_A_twice_reversed_prior_same',A:'SWAPPED',B:'NORMAL',alertA:true,again:'A'},
 {id:'05_A_twice_reversed_prior_changed',A:'SWAPPED',B:'NORMAL',alertA:true,again:'A',againFlip:true},
 {id:'06_B_once',A:'NORMAL',B:'NORMAL',alertB:true},
 {id:'07_B_twice_same',A:'NORMAL',B:'NORMAL',alertB:true,again:'B'},
 {id:'08_B_twice_changed',A:'NORMAL',B:'NORMAL',alertB:true,again:'B',againFlip:true},
 {id:'09_A_first_drift_then_reverted',A:'NORMAL',B:'NORMAL',alertA:true,flipA:true,again:'A',againFlip:true},
 {id:'10_A_first_drift_then_stable',A:'NORMAL',B:'NORMAL',alertA:true,flipA:true,again:'A'},
 {id:'11_B_first_drift_then_reverted',A:'NORMAL',B:'NORMAL',alertB:true,flipB:true,again:'B',againFlip:true}
];
function prep(w,arm){const h=v51.prep(w,arm);h.v52={reAlerted:false,completedFirstProbe:false,movesSinceProbe:0,secondNoticeAt:null};return h;}
function goal(h,s){return s==='A'?h.w.goal:h.w.goal2;}
function safeSite(h,s){return s==='A'?h.w.s:h.siteB;}
function surface(h){return h.stage===0?'A':h.stage===1?'B':null;}
function notice(h,arm){
 // First alert is the v51 host event: physically non-diagnostic, delivered only at the objective.
 const first=v51.alert(h,arm);
 // After first safe probe+erase, alert again while the agent is travelling back.
 const s=arm.again;
 if(s&&!h.v52.reAlerted&&h.v52.completedFirstProbe&&surface(h)===s&&
    h.v52.movesSinceProbe>=(arm.delay||1)&&h.pos!==safeSite(h,s)){
   if(arm.againFlip)h.laws[s]=h.laws[s]==='NORMAL'?'SWAPPED':'NORMAL';
   ok(h.painted[safeSite(h,s)]===undefined,'second alert after erased calibration');
   h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:s,site:h.g.coords[safeSite(h,s)]});
   h.v52.reAlerted=true;h.v52.secondNoticeAt=h.accepted;
   return 'again_'+s;
 }
 return first;
}
function observe(h){const x=v51.observe(h);x.v='synthetic-v52-temporal-probe-invalidation';return x;}
function command(h,a,arm){const s=arm.again,prior=h.pos,stage=surface(h),wasPainted=h.painted[prior]!==undefined;
 const r=v51.command(h,a);
 if(s&&stage===s&&prior===safeSite(h,s)&&a==='ERASE'&&wasPainted&&!h.v52.completedFirstProbe){
  const alerts=h.events.filter(e=>e.kind==='SURFACE_ALERT'&&e.surface===s);
  const probes=h.events.filter(e=>e.kind==='PAINT_RESULT'&&e.surface===s&&e.at>alerts[0]?.at&&e.site.join(',')===h.g.coords[prior].join(','));
  if(probes.length){h.v52.completedFirstProbe=true;h.v52.movesSinceProbe=0;}
 }
 if(s&&stage===s&&h.v52.completedFirstProbe&&a.startsWith('MOVE_')&&!h.v52.reAlerted)h.v52.movesSinceProbe++;
 return r;
}
function baselineAfterFirst(i,s){const x=cp(i),alerts=x.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===s);
 if(alerts.length>1){const newest=alerts.at(-1);let removed=false;
  x.durable=x.durable.filter(e=>{if(!removed&&e===newest){removed=true;return false;}return true;});
  // Here array reference compares are to x.durable elements, so only the final notice is dropped.
 }
 return x;
}
function rollout(w,arm){const h=prep(w,arm),rows=[],probes=[],paints=[],signalActions=[];let notices=0,priorSecond=null;
 for(let step=0;step<340&&!h.done;step++){
  const n=notice(h,arm);if(n){notices++;if(n.startsWith('again'))priorSecond=cp(observe(h));}
  const i=observe(h),a=v51.teacher(i);ok(i.legal.includes(a),'teacher illegal '+a);
  rows.push({input:i,action:a,weight:1});const stage=surface(h),at=h.pos,r=command(h,a,arm);
  if(a.startsWith('PAINT_')){if(at===goal(h,stage)){paints.push({surface:stage,action:a,material:r.material,input:cp(i)});}
   else probes.push({surface:stage,action:a,material:r.material,input:cp(i)});}
  if(n&&n.startsWith('again'))signalActions.push({action:a,position:i.p,oldAction:v51.teacher(baselineAfterFirst(i,arm.again)),input:cp(i)});
 }
 ok(h.done&&h.success&&!h.protected_damage,'host goal did not succeed '+arm.id);
 const expected=Number(!!arm.alertA)+Number(!!arm.alertB)+Number(!!arm.again);
 ok(notices===expected,'missing alert '+arm.id+' '+notices+'/'+expected);
 ok(probes.length===expected,'host probe count '+arm.id+' '+probes.length+'/'+expected);
 ok(probes.every(p=>p.action==='PAINT_A'),'probe not PAINT_A');
 ok(paints.length===2&&paints[0].material==='A'&&paints[1].material==='B','protected goal paint wrong');
 ok(!Object.entries(h.painted).some(([key])=>+key!==h.w.goal&&+key!==h.w.goal2),'unremoved probe material');
 if(arm.again){ok(h.v52.reAlerted&&priorSecond,'second alert not delivered');
  ok(signalActions.length===1&&signalActions[0].action!==signalActions[0].oldAction,'second alert did not change correct movement');
  const second=probes.findLast(p=>p.surface===arm.again);
  ok(second&&second.input.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===arm.again).length===2,'no second physical probe');
 }
 return {rows,probes,paints,notices,signalActions,steps:rows.length,secondNoticeAt:h.v52.secondNoticeAt};
}
function maskAfterSecond(i,s){const x=cp(i),alerts=x.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===s);
 ok(alerts.length===2,'mask requires two notices');const t=alerts.at(-1)[0];
 x.durable=x.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[4]===s&&e[0]>t));return x;
}
function world(w){const traces=ARMS.map(a=>rollout(w,a));
 // Identical observable states before second physical measurement: the SECOND
 // PAINT_A command must produce different material without a hidden law leak.
 for(const [aa,bb,s] of [[2,3,'A'],[4,5,'A'],[7,8,'B'],[9,10,'A']]){
  const a=traces[aa],b=traces[bb],proA=a.probes.filter(p=>p.surface===s).at(-1),proB=b.probes.filter(p=>p.surface===s).at(-1);
  ok(H(proA.input)===H(proB.input)&&proA.action===proB.action,'pre-second-probe inputs differ '+aa+'/'+bb);
  ok(proA.material!==proB.material,'second physical measurement not causal '+aa+'/'+bb);
  const pa=a.paints.find(p=>p.surface===s),pb=b.paints.find(p=>p.surface===s);
  ok(pa.action!==pb.action,'final goal paint not affected '+aa+'/'+bb);
  ok(H(maskAfterSecond(pa.input,s))===H(maskAfterSecond(pb.input,s)),'post-second-receipt ablation mismatch '+aa+'/'+bb);
 }
 ok(traces[0].probes.length===0&&traces[1].probes.length===1,'no-warning and single-warning controls');
 ok(traces[2].probes.length===2&&traces[7].probes.length===2,'repeated warning must demand fresh evidence');
 ok(traces[9].paints[0].action===traces[0].paints[0].action,'double flip should return to original law');
 return {traces,meta:{synthetic_not_native:true,split:w.split,source_world_index:w.index,source_seed:w.seed,
  geometry_sha256:w.geometry_sha256,arms:ARMS.map(a=>a.id),episode_steps:traces.map(t=>t.steps),
  successful_synthetic_host_episodes:traces.length,active_probe_paints:traces.map(t=>t.probes.length),
  temporal_invalidation_counterfactuals:[[2,3],[4,5],[7,8],[9,10]],
  no_warning_negative:[0,1],reAlert_no_drift_negative:[1,2],B_single_repeated_control:[6,7],
  teacher_vs_stale_warning_changed_actions:traces.filter(t=>t.signalActions.length).length,
  second_alerts_on_return:true,second_paint_receipt_ablation_checked:true}};
}
function write(dir,f,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,f),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output already exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},allGeometries=new Map(),inputs=new Map();let total=0;
 for(const [split,n] of Object.entries(SPLITS)){
  const count=quick?Math.min(n,2):n,policy=[],challenge=[];let terminals=0,probeCount=0,zero=0,secondAlerts=0;
  for(let ix=0;ix<count;ix++){
   const w=v46.candidate(split,500+ix).w,collision=allGeometries.get(w.geometry_sha256);
   ok(!collision||collision===split,'cross-split geometry leakage');allGeometries.set(w.geometry_sha256,split);
   const z=world(w);challenge.push(z.meta);
   for(const trace of z.traces){terminals++;probeCount+=trace.probes.length;secondAlerts+=trace.signalActions.length;
    for(const r of trace.rows){const id=H(r.input),prev=inputs.get(id);
     ok(!prev||prev.split===split,'identical input across splits');ok(!prev||prev.action===r.action,'contradictory action');
     if(prev&&split==='TRAIN'){r.weight=0;zero++;}else inputs.set(id,{split,action:r.action});policy.push(r);
    }
   }
  }
  shards[split+'_policy.jsonl']=write(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=write(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,successful_host_episodes:terminals,policy_decisions:policy.length,
   accepted_safe_probe_paints:probeCount,repeated_alerts_while_returning:secondAlerts,
   per_world_causal_second_probe_pairs:4,train_exact_duplicates_zero_weight:zero};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/realert-v52/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'realert_v52.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contract:{model_visible:'Only policy.input is inference-visible; action and weight are training labels; NEVER include evaluator-only challenge in training.',
   actions:'Original legal host MOVE, PAINT_A, PAINT_B, ERASE, STOP; no invented actions.',
   intervention:'New observable warning invalidates previously acquired physical calibration, even if the previously observed cell had been safely erased and the model has already begun moving to the objective.',
   paired_heldout:'Four causally matched pairs per world differ only in hidden law after second alert, with identical inputs before physical re-probe, and different correct protected-goal paint actions afterward.',
   negatives:'Second alert with no drift still requires fresh probe; absence of warning must not induce one; stale-warning control would move toward the objective instead of the safe probe.',
   grouping:'World geometry/source seed retained per split; v52 uses v46 candidate indices 500+, distinct from v44-v51; consumers MUST group cross-version identities by geometry_sha256 and source_seed.',
   dedup:'Exact repeated TRAIN observations zero-weighted; all actions remain for host replay and counterfactual evaluation.'},
  splits,shards,unique_world_geometries:allGeometries.size,total_policy_decisions:total,
  limitations:['Synthetic graph host, NOT native CyberSand; SURFACE_ALERT and protected-target semantics hypothetical.',
   'Synthetic event after departure from the safe site is externally injected, not generated by CyberSand gameplay.',
   'A small symbolic per-surface timestamped physical-receipt controller can still solve this benchmark; measure neural HELDOUT instead of claiming learned capability.',
   'Host-executed prelude/survey remains scripted; no target SLM training, inference latency/RSS, native trial or immutable Actions artifact.']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage: node realert_v52.js NEW_OUTPUT_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,ARMS,prep,notice,observe,command,rollout,world,generate,maskAfterSecond,baselineAfterFirst,H};
