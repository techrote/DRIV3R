'use strict';
/* DRIV3R Corpus Factory v51: learned-action-ready active physical probes after
 * an observable but non-diagnostic surface alert. DATA, not policy training.
 * All actions synthetic host accepted; NOT native CyberSand. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v50=require('./online_recovery_v50.js'),v49=require('./contextual_probe_v49.js');
const v46=require('./mission_memory_v46.js'),v45=require('./frontier_memory_v45.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error(m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const ARMS=[
 {id:'00_no_alert',alertA:false,alertB:false,flipA:false,flipB:false,A:'NORMAL',B:'NORMAL'},
 {id:'01_A_warning_no_drift',alertA:true,alertB:false,flipA:false,flipB:false,A:'NORMAL',B:'NORMAL'},
 {id:'02_A_warning_drift',alertA:true,alertB:false,flipA:true,flipB:false,A:'NORMAL',B:'NORMAL'},
 {id:'03_B_warning_no_drift',alertA:false,alertB:true,flipA:false,flipB:false,A:'NORMAL',B:'NORMAL'},
 {id:'04_B_warning_drift',alertA:false,alertB:true,flipA:false,flipB:true,A:'NORMAL',B:'NORMAL'},
 {id:'05_AB_warning_no_drift',alertA:true,alertB:true,flipA:false,flipB:false,A:'NORMAL',B:'NORMAL'},
 {id:'06_AB_warning_drift',alertA:true,alertB:true,flipA:true,flipB:true,A:'NORMAL',B:'NORMAL'},
 {id:'07_A_reverse_no_drift',alertA:true,alertB:false,flipA:false,flipB:false,A:'SWAPPED',B:'NORMAL'},
 {id:'08_A_reverse_drift',alertA:true,alertB:false,flipA:true,flipB:false,A:'SWAPPED',B:'NORMAL'}
];
function prep(w,arm){return v50.prep(w,arm);}
function alert(h,arm){
 const surf=h.stage===0?'A':h.stage===1?'B':null;
 const target=surf==='A'?h.w.goal:surf==='B'?h.w.goal2:null;
 if(!surf||h.pos!==target||h.painted[h.pos]!==undefined||h.alerted?.[surf])return null;
 if(!((surf==='A')?arm.alertA:arm.alertB))return null;
 if(!h.alerted)h.alerted={};h.alerted[surf]=true;
 if((surf==='A')?arm.flipA:arm.flipB)h.laws[surf]=h.laws[surf]==='NORMAL'?'SWAPPED':'NORMAL';
 const site=surf==='A'?h.w.s:h.siteB;
 ok(h.surfaces[site]===surf&&h.painted[site]===undefined,'unsafe calibration site');
 h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:surf,site:h.g.coords[site]});
 return surf;
}
function observe(h){const z=v50.observe(h);z.v='synthetic-v51-active-safe-probe';
 z.durable=[...z.durable,...h.events.filter(e=>e.kind==='SURFACE_ALERT').map(e=>[e.at,'SURFACE_ALERT',e.surface,e.site])]
  .sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));return z;
}
function same(a,b){return JSON.stringify(a)===JSON.stringify(b);}
function target(i){const q=i.durable.filter(z=>z[1]==='MISSION_ASSIGN'||z[1]==='GOAL_REVISION');ok(q.length,'missing mission');return {coord:q.at(-1)[2],material:q.at(-1)[3]};}
function lastAlert(i,surf){return i.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===surf).at(-1);}
function postAlertProbe(i,notice){return i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===notice[2]&&e[0]>notice[0]&&e[2]==='PAINT_A'&&same(e[5],notice[3])).at(-1);}
function teacher(i){
 if(i.durable.some(e=>e[1]==='MISSION_COMPLETE'))return 'STOP';
 const active=target(i),notice=lastAlert(i,active.material);
 if(!notice)return v50.teacher(i);
 const site=notice[3],fresh=postAlertProbe(i,notice),materialAtSite=i.painted.some(([p])=>same(p,site));
 if(!fresh){
  if(!same(i.p,site))return v45.pathOnSurvey(i,site);
  return materialAtSite?'ERASE':'PAINT_A';
 }
 if(same(i.p,site)&&materialAtSite)return 'ERASE';
 if(!same(i.p,active.coord))return v45.pathOnSurvey(i,active.coord);
 return v50.teacher(i);
}
function command(h,action){
 const before=h.stage,at=h.pos,r=v50.command(h,action);
 if(action.startsWith('PAINT_')&&h.alerted){const surf=before===0?'A':before===1?'B':null;
  if(surf&&h.alerted[surf]&&at===(surf==='A'?h.w.goal:h.w.goal2)&&r.material!==surf){
   h.done=true;h.success=false;h.protected_damage=true;
  }
 }
 return r;
}
function ablatePhysical(i,surf,notice){const z=cp(i);z.durable=z.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[4]===surf&&e[0]>notice[0]));return z;}
function rollout(w,arm,model=teacher){const h=prep(w,arm),rows=[],probes=[],goalPaints=[];let alerts=0;
 for(let step=0;step<320&&!h.done;step++){
  if(alert(h,arm))alerts++;
  const i=observe(h),a=model(i);ok(i.legal.includes(a),'illegal model action '+a);
  rows.push({input:i,action:a,weight:1});const before=h.stage,at=h.pos,r=command(h,a);
  if(a.startsWith('PAINT_')){
   if((before===0&&at===h.w.goal)||(before===1&&at===h.w.goal2))goalPaints.push({surface:before===0?'A':'B',action:a,material:r.material,input:cp(i)});
   else probes.push({surface:v49.label(h),action:a,material:r.material,site:h.g.coords[at],input:cp(i)});
  }
 }
 ok(h.done&&h.success&&!h.protected_damage,'protected target failed '+arm.id);
 ok(alerts===Number(!!arm.alertA)+Number(!!arm.alertB),'missing alerts');
 ok(probes.length===alerts&&probes.every(z=>z.action==='PAINT_A'),'probe not selected by teacher');
 ok(goalPaints.length===2&&goalPaints[0].material==='A'&&goalPaints[1].material==='B','wrong physical objective paint');
 ok(!Object.entries(h.painted).some(([k])=>+k!==h.w.goal&&+k!==h.w.goal2),'probe left residual paint');
 return {rows,probes,goalPaints,alerts,steps:rows.length};
}
function world(w){const traces=ARMS.map(a=>rollout(w,a));
 // Alert/no-flip versus alert/flip: all observations and actions are identical
 // until AFTER physically issuing the safe test paint.
 for(const [k,l,surf] of [[1,2,'A'],[3,4,'B'],[5,6,'A'],[7,8,'A']]){
  const a=traces[k],b=traces[l];ok(a.probes.length&&b.probes.length,'missing active probe');
  const x=a.probes.find(z=>z.surface===surf),y=b.probes.find(z=>z.surface===surf);
  ok(x&&y&&H(x.input)===H(y.input)&&x.action===y.action,'pre-probe observation leaked hidden drift');
  ok(x.material!==y.material,'positive physical probe not causal');
  const gpA=a.goalPaints.find(z=>z.surface===surf),gpB=b.goalPaints.find(z=>z.surface===surf);
  ok(gpA.action!==gpB.action,'post-probe goal command did not change');
  const notice=lastAlert(gpA.input,surf);ok(notice,'no noticed surface');
  ok(H(ablatePhysical(gpA.input,surf,notice))===H(ablatePhysical(gpB.input,surf,notice)),'post-probe ablation not matched');
 }
 ok(traces[0].probes.length===0&&traces[1].goalPaints[0].action===traces[0].goalPaints[0].action,'false-warning changed expected material');
 ok(traces[3].goalPaints[0].action===traces[0].goalPaints[0].action,'future-surface warning altered A action');
 // Late B probe is physically triggered only after A completes, not as part of a scripted prelude.
 ok(traces[3].probes[0].surface==='B'&&traces[3].probes[0].input.durable.some(e=>e[1]==='GOAL_REVISION'),'B active probe missing second stage');
 const meta={synthetic_not_native:true,source_split:w.split,source_world_index:w.index,source_seed:w.seed,geometry_sha256:w.geometry_sha256,
  arms:ARMS.map(a=>a.id),decisions:traces.map(t=>t.steps),active_probe_paints:traces.map(t=>t.probes.length),
  protected_goal_correct_paints:traces.map(t=>t.goalPaints.length),host_successful_episodes:traces.length,
  alert_only_A_negative:[0,1],alert_only_B_negative:[0,3],A_safe_probe_positive:[1,2],B_safe_probe_positive:[3,4],
  AB_joint_probe_positive:[5,6],reverse_prior_A_positive:[7,8],
  probe_pre_action_identical_for_counterfactual:true,
  no_new_synthetic_action_tokens:true};
 return {traces,meta};
}
function write(dir,name,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,name),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},seenGeometry=new Map(),seenInput=new Map();let total=0;
 for(const [split,n] of Object.entries(SPLITS)){
  const policy=[],challenge=[],count=quick?Math.min(n,2):n;let probes=0,zero=0,terminal=0;
  for(let index=0;index<count;index++){
   const w=v46.candidate(split,300+index).w,x=world(w),prior=seenGeometry.get(w.geometry_sha256);
   ok(!prior||prior===split,'cross-split geometry overlap');seenGeometry.set(w.geometry_sha256,split);
   challenge.push(x.meta);
   for(const t of x.traces){terminal++;probes+=t.probes.length;
    for(const row of t.rows){const k=H(row.input),prev=seenInput.get(k);
     ok(!prev||prev.split===split,'cross-split input collision');ok(!prev||prev.action===row.action,'conflicting label');
     if(prev&&split==='TRAIN'){row.weight=0;zero++;}else seenInput.set(k,{split,action:row.action});policy.push(row);
    }
   }
  }
  shards[split+'_policy.jsonl']=write(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=write(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,successful_synthetic_episodes:terminal,policy_decisions:policy.length,
   active_host_probe_paints:probes,protected_correct_goal_paints:terminal*2,
   causal_A_probe_pairs:count,causal_B_probe_pairs:count,causal_joint_probe_pairs:count,reverse_prior_probe_pairs:count,
   warning_without_drift_negative_controls:count*2,train_exact_repeats_zero_weight:zero};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/active-probe-v51/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'active_probe_v51.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contract:{input:'Only policy.input is model-visible; action/weight training labels; never train on evaluator-only challenge rows.',
   action_space:'Same host legal movement, PAINT_A, PAINT_B, ERASE, STOP. No new policy action token.',
   grouping:'Retain train/dev/heldout source split and group by source_seed/geometry_sha256 across all corpus versions; v51 sources use v46 candidate indices >=300.',
   active_probe:'A non-diagnostic surface alert is visible at a target. Teacher returns through its observed graph to the safe site, issues actual PAINT_A, receives the host physical result, ERASEs, returns, and selects material-correct target paint.',
   protection:'Synthetic protected-goal test: wrong physical paint after a surface warning terminates the episode unsuccessfully; no failed examples disguised as successes.',
   negatives:'Warning with no law change must still cause safe validation but must not change the correct material command. No-alert baseline avoids unnecessary probe.',
   causal:'Matched no-drift/drift arms are input-identical before the safe probe; after it, removing the new physical outcome makes target inputs identical although legal correct commands differ.',
   leak:'Hidden mapping/flip/protection-damage and scenario labels appear only in evaluator metadata, never in input.'},
  splits,shards,total_policy_decisions:total,unique_graph_worlds:seenGeometry.size,
  limitations:['Synthetic graph/material adapter NOT native CyberSand; warnings and failure semantics hypothetical, require native ABI validation.',
  'Prelude survey and historical calibrations are scripted host-executed operations, not learned exploration.',
  'One per-surface safe test still suffices; labels are symbolic observable-state teacher not neural success.',
  'No LFM/Sprocket training, SLM p95 latency/RSS, native episodes, or immutable Actions shard artifacts.']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage: node active_probe_v51.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,ARMS,prep,alert,observe,target,lastAlert,postAlertProbe,teacher,command,ablatePhysical,rollout,world,generate,H};
