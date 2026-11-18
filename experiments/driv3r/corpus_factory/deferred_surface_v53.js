'use strict';
/* DRIV3R Corpus Factory v53 — deferred cross-surface warning obligations.
 * DATA/ORACLE ONLY, synthetic host != native CyberSand. No policy trained.
 * Emits an observable B-surface warning while mission A remains active.
 * The warning may be interleaved with A's second (recalibration) alert.
 * B's unseen physical law may independently change; only a safe physical
 * B probe/erase can resolve it when B eventually becomes the objective.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v52=require('./realert_v52.js'),v51=require('./active_probe_v51.js');
const v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,msg)=>{if(!x)throw Error('V53 '+msg)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:32,DEV:12,HELDOUT:12};
const base={A:'NORMAL',B:'NORMAL'};
const ARMS=[
 {id:'00_none'},
 {id:'01_B_early_no_drift',when:'initial'},
 {id:'02_B_early_drift',when:'initial',flipB:true},
 {id:'03_B_early_reverse_no_drift',when:'initial',B:'SWAPPED'},
 {id:'04_B_early_reverse_drift',when:'initial',B:'SWAPPED',flipB:true},
 {id:'05_A_one',alertA:true},
 {id:'06_A_one_B_early_no_drift',alertA:true,when:'initial'},
 {id:'07_A_one_B_early_drift',alertA:true,when:'initial',flipB:true},
 {id:'08_A_two_same',alertA:true,again:'A'},
 {id:'09_A_two_B_during_second_no_drift',alertA:true,again:'A',when:'second'},
 {id:'10_A_two_B_during_second_drift',alertA:true,again:'A',when:'second',flipB:true},
 {id:'11_A_two_changed',alertA:true,again:'A',againFlip:true},
 {id:'12_A_changed_B_during_second_no_drift',alertA:true,again:'A',againFlip:true,when:'second'},
 {id:'13_A_changed_B_during_second_drift',alertA:true,again:'A',againFlip:true,when:'second',flipB:true},
 {id:'14_A_changed_B_early_no_drift',alertA:true,again:'A',againFlip:true,when:'initial'},
 {id:'15_A_changed_B_early_drift',alertA:true,again:'A',againFlip:true,when:'initial',flipB:true},
 {id:'16_A_two_B_early_no_drift',alertA:true,again:'A',when:'initial'},
 {id:'17_A_two_B_early_drift',alertA:true,again:'A',when:'initial',flipB:true}
];
function armParams(arm){return {...base,...arm,alertB:false,flipA:false};}
function injectB(h,arm,signal,step){if(!arm.when||h.v53.futureB)return false;
 if(!((arm.when==='initial'&&step===0)||(arm.when==='second'&&signal==='again_A')))return false;
 ok(h.stage===0,'B deferred alert must occur during A');
 ok(h.painted[h.siteB]===undefined,'unsafe B surface warning');
 if(arm.flipB)h.laws.B=h.laws.B==='NORMAL'?'SWAPPED':'NORMAL';
 h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
 h.v53.futureB={at:h.accepted,stage:h.stage,phase:arm.when};return true;
}
function prep(w,arm){const h=v52.prep(w,armParams(arm));h.v53={futureB:null};return h;}
function observe(h){const i=v52.observe(h);i.v='synthetic-v53-deferred-cross-surface';return i;}
function surface(h){return h.stage===0?'A':h.stage===1?'B':null;}
function rollout(w,arm){const h=prep(w,arm),rows=[],probes=[],paints=[],secondActions=[];let aAlerts=0,bAlerts=0,bGoalAge=null;
 for(let step=0;step<360&&!h.done;step++){
  const notice=v52.notice(h,armParams(arm));if(notice)aAlerts++;
  if(injectB(h,arm,notice,step))bAlerts++;
  const i=observe(h),s=surface(h),a=v51.teacher(i);ok(i.legal.includes(a),'illegal action '+a);
  if(s==='B'&&h.pos===h.w.goal2&&bGoalAge===null&&h.v53.futureB)bGoalAge=h.accepted-h.v53.futureB.at;
  rows.push({input:i,action:a,weight:1});
  if(notice==='again_A')secondActions.push({input:cp(i),action:a});
  const at=h.pos,r=v52.command(h,a,armParams(arm));
  if(a.startsWith('PAINT_')){
   if(at===(s==='A'?h.w.goal:h.w.goal2))paints.push({surface:s,action:a,material:r.material,input:cp(i)});
   else probes.push({surface:s,action:a,material:r.material,input:cp(i)});
  }
 }
 ok(h.done&&h.success&&!h.protected_damage,'synthetic host failed '+arm.id);
 ok(aAlerts===Number(!!arm.alertA)+Number(!!arm.again),'A warning count '+arm.id);
 ok(bAlerts===Number(!!arm.when),'B warning count '+arm.id);
 ok(probes.length===aAlerts+bAlerts,'physical probes missing '+arm.id);
 ok(probes.every(x=>x.action==='PAINT_A'),'calibration must be PAINT_A');
 ok(paints.length===2&&paints[0].material==='A'&&paints[1].material==='B','protected paint mismatch');
 ok(!Object.entries(h.painted).some(([k])=>+k!==h.w.goal&&+k!==h.w.goal2),'unerased probe residue');
 if(arm.when){ok(h.v53.futureB&&bGoalAge!==null&&bGoalAge>0,'B warning not deferred to future mission');
  ok(probes.some(x=>x.surface==='B'),'no deferred B probe');}
 return {rows,probes,paints,secondActions,bGoalAge,aAlerts,bAlerts,steps:rows.length};
}
function ablateBPhysical(i){const z=cp(i);const warning=z.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]==='B').at(-1);ok(warning,'missing B warning');
 z.durable=z.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[4]==='B'&&e[0]>warning[0]));return z;}
function ablateBWarning(i){const z=cp(i);z.durable=z.durable.filter(e=>!(e[1]==='SURFACE_ALERT'&&e[2]==='B'));return z;}
function world(w){const traces=ARMS.map(a=>rollout(w,a));
 const pairs=[[1,2],[3,4],[6,7],[9,10],[12,13],[14,15],[16,17]];
 for(const [a,b] of pairs){const x=traces[a],y=traces[b],probeX=x.probes.find(z=>z.surface==='B'),probeY=y.probes.find(z=>z.surface==='B');
  ok(probeX&&probeY&&H(probeX.input)===H(probeY.input),'hidden B drift leaked preprobe '+a+'/'+b);
  ok(probeX.material!==probeY.material,'noncausal physical B measurement '+a+'/'+b);
  const lastX=x.paints.find(z=>z.surface==='B'),lastY=y.paints.find(z=>z.surface==='B');
  ok(lastX.action!==lastY.action,'same final B command despite different B law '+a+'/'+b);
  ok(H(ablateBPhysical(lastX.input))===H(ablateBPhysical(lastY.input)),'B receipt ablation fails '+a+'/'+b);
 }
 for(const [a,b] of [[0,1],[5,6],[8,9],[11,12],[11,14],[8,16]]){
  const x=traces[a],y=traces[b];ok(x.paints.find(z=>z.surface==='A').action===y.paints.find(z=>z.surface==='A').action,'B warning changed A paint '+a+'/'+b);
 }
 for(const [a,b] of [[8,9],[11,12]]){
  const x=traces[a].secondActions[0],y=traces[b].secondActions[0];
  ok(x&&y&&x.action===y.action,'B warning changed active A alert action');
  ok(H(x.input)===H(ablateBWarning(y.input)),'unexpected observation difference at coincident A/B alert');
 }
 ok(traces[0].probes.length===0&&traces[1].probes.length===1,'no-warning and future-only controls');
 const ages=traces.filter(z=>z.bGoalAge!==null).map(z=>z.bGoalAge);
 return {traces,meta:{synthetic_not_native:true,split:w.split,source_world_index:w.index,source_seed:w.seed,
  geometry_sha256:w.geometry_sha256,arms:ARMS.map(z=>z.id),episode_steps:traces.map(t=>t.steps),
  deferred_B_alert_age_at_goal_actions:traces.map(t=>t.bGoalAge),
  host_successful_episodes:traces.length,physical_probe_paints:traces.map(t=>t.probes.length),
  deferred_B_causal_physical_pairs:pairs,irrelevant_B_warning_A_negative:[[0,1],[5,6],[8,9],[11,12]],
  late_B_warning_interleaved_with_A_second:[[8,9],[11,12]],
  worst_deferred_age_actions:Math.max(...ages),earliest_deferred_age_actions:Math.min(...ages),
  preprobe_hidden_drift_equal_observation:true,postprobe_B_action_contrast:true}};
}
function write(dir,f,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,f),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output exists');fs.mkdirSync(dir,{recursive:true});
 const splits={},shards={},geometry=new Map(),inputs=new Map();let allRows=0;
 for(const [split,n] of Object.entries(SPLITS)){
  const count=quick?Math.min(n,2):n,policy=[],challenge=[];let terminals=0,probes=0,duplicates=0,maxAge=0;
  for(let ix=0;ix<count;ix++){
   const w=v46.candidate(split,700+ix).w,old=geometry.get(w.geometry_sha256);
   ok(!old||old===split,'cross-split geometry leakage');geometry.set(w.geometry_sha256,split);
   const r=world(w);challenge.push(r.meta);maxAge=Math.max(maxAge,r.meta.worst_deferred_age_actions);
   for(const trace of r.traces){terminals++;probes+=trace.probes.length;
    for(const row of trace.rows){const hash=H(row.input),prev=inputs.get(hash);
     ok(!prev||prev.split===split,'cross-split duplicate observations');
     ok(!prev||prev.action===row.action,'conflicting labels');
     if(prev&&split==='TRAIN'){row.weight=0;duplicates++;}else inputs.set(hash,{split,action:row.action});
     policy.push(row);
    }
   }
  }
  shards[split+'_policy.jsonl']=write(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=write(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,synthetic_host_successes:terminals,policy_decisions:policy.length,host_probe_paints:probes,
   deferred_B_physical_causal_pairs:7*count,irrelevant_B_warning_negative_pairs:4*count,
   coincident_A_B_warning_controls:2*count,max_deferred_warning_age_actions:maxAge,
   exact_repeats_zero_weight_train:duplicates};allRows+=policy.length;
 }
 const parents=['realert_v52.js','active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js',
  'empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
 const mf={schema:'driv3r/corpus_factory/deferred-cross-surface-v53/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'deferred_surface_v53.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:parents.map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contract:{model_visible:'Only policy.input. NEVER supply evaluator-only challenge metadata to the learner.',
   action_tokens:'Unmodified legal MOVE, PAINT_A, PAINT_B, ERASE, STOP.',
   intervention:'B alert while A active creates future B calibration obligation; A second alert and B warning may share action timestamp.',
   counterfactual:'Seven B law flip/no-flip pairs share identical preprobe input but different physical paint results and final B goal commands.',
   isolation:'Worlds use v46 candidate indices 700+ and split-locked topology hashes; group same seeds/geometries across all versions.',
   dedup:'Exact TRAIN repeats zero-weighted without deleting host-replay actions; DEV/HELDOUT unsampled.'},
  splits,shards,world_geometries:geometry.size,total_policy_decisions:allRows,
  limitations:['Synthetic graph host only, NOT native CyberSand; SURFACE_ALERT and protection semantics hypothetical.',
   'A symbolic per-surface event/receipt parser can still solve all held-out scenarios.',
   'Pre-mission physical survey and calibration scripted; no learned exploration demonstrated.',
   'No actual target SLM trained, no native integration, no neural latency/RSS or immutable Actions shard publication.']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage: node deferred_surface_v53.js NEW_OUTPUT_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,ARMS,armParams,injectB,prep,observe,rollout,ablateBPhysical,ablateBWarning,world,generate,H};