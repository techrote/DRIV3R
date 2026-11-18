'use strict';
/* DRIV3R Corpus Factory v50 — unsignalled contextual tool-law drift and
 * host-feedback recovery. DATA ONLY. Synthetic host != native CyberSand. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v49=require('./contextual_probe_v49.js'),v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error(m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const arms=[
 {id:'00_static_normal',A:'NORMAL',B:'NORMAL',flipA:false,flipB:false},
 {id:'01_A_drift',A:'NORMAL',B:'NORMAL',flipA:true,flipB:false},
 {id:'02_B_drift',A:'NORMAL',B:'NORMAL',flipA:false,flipB:true},
 {id:'03_AB_drift',A:'NORMAL',B:'NORMAL',flipA:true,flipB:true},
 {id:'04_static_A_swapped',A:'SWAPPED',B:'NORMAL',flipA:false,flipB:false},
 {id:'05_A_drift_reverse',A:'SWAPPED',B:'NORMAL',flipA:true,flipB:false},
 {id:'06_static_B_swapped',A:'NORMAL',B:'SWAPPED',flipA:false,flipB:false},
 {id:'07_B_drift_reverse',A:'NORMAL',B:'SWAPPED',flipA:false,flipB:true},
 {id:'08_mission_retarget',A:'NORMAL',B:'NORMAL',flipA:false,flipB:false,mission:true},
 {id:'09_gate_closed',A:'NORMAL',B:'NORMAL',flipA:false,flipB:false,closed:true},
 {id:'10_obsolete_A_probe',A:'NORMAL',B:'NORMAL',oldA:'SWAPPED',flipA:false,flipB:false},
 {id:'11_obsolete_B_probe',A:'NORMAL',B:'NORMAL',oldB:'SWAPPED',flipA:false,flipB:false}
];
function prep(w,arm){return v49.prepare(w,{id:arm.id,A:arm.A,B:arm.B,oldA:arm.oldA||'NORMAL',oldB:arm.oldB||'NORMAL',closed:!!arm.closed,mission:!!arm.mission,reverse:false});}
function observe(h){const x=v49.observe(h);x.v='synthetic-v50-online-contextual-recovery';return x;}
// A real issued-paint receipt is generated ONLY after a physical host action.
// No TOOL_LAW or drift flag is included in the model input.
function command(h,a){const r=v49.command(h,a);if(a.startsWith('PAINT_')){
 ok(r.material==='A'||r.material==='B','missing host physical paint outcome');
 h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:a,material:r.material,surface:v49.label(h),site:h.g.coords[h.pos]});
 }return r;}
function inferredLaw(i){const q=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===i.surface);
 ok(q.length>=2,'physical calibrations are absent');const e=q.at(-1);
 ok(['PAINT_A','PAINT_B'].includes(e[2])&&['A','B'].includes(e[3]),'invalid physical receipt');
 return e[2].slice(-1)===e[3]?'NORMAL':'SWAPPED';}
function teacher(i){const physical=v46.teacher(i);if(!physical.startsWith('PAINT_'))return physical;
 const law=inferredLaw(i);return law==='NORMAL'?physical:'PAINT_'+(physical.endsWith('A')?'B':'A');}
function maybeDrift(h,arm){let surf=null;
 if(h.stage===0&&h.pos===h.w.goal&&h.painted[h.pos]===undefined&&arm.flipA&&!h.didFlipA){surf='A';h.didFlipA=true;}
 else if(h.stage===1&&h.pos===h.w.goal2&&h.painted[h.pos]===undefined&&arm.flipB&&!h.didFlipB){surf='B';h.didFlipB=true;}
 if(surf){ok(h.painted[h.pos]===undefined,'drift must happen at empty objective');
 h.laws[surf]=h.laws[surf]==='NORMAL'?'SWAPPED':'NORMAL';
 }return surf;}
function rollout(h,arm){const rows=[],attempts=[],errors=[],erases=[],preexistingErases=[];const pending=new Set();let firstA=null,firstB=null,flips=[];
 for(let t=0;t<256&&!h.done;t++){
  const drift=maybeDrift(h,arm);if(drift)flips.push({surface:drift,at:h.accepted});
  const i=observe(h),a=teacher(i);ok(i.legal.includes(a),'oracle illegal '+a);
  if(a.startsWith('PAINT_')&&h.stage===0&&h.pos===h.w.goal&&!firstA)firstA={input:cp(i),action:a};
  if(a.startsWith('PAINT_')&&h.stage===1&&h.pos===h.w.goal2&&!firstB)firstB={input:cp(i),action:a};
  rows.push({input:i,action:a,weight:1});const beforeStage=h.stage,at=h.pos,r=command(h,a);
  if(a.startsWith('PAINT_')){const target=beforeStage===0?h.w.goal:h.w.goal2,expected=beforeStage===0?'A':'B';
   ok(at===target,'paint outside active goal');attempts.push({surface:i.surface,command:a,material:r.material,expected,correct:r.material===expected});
   if(r.material!==expected){errors.push({surface:i.surface,command:a,material:r.material});pending.add(i.surface);}
  }
  if(a==='ERASE'&&at===(beforeStage===0?h.w.goal:h.w.goal2)){if(pending.has(i.surface)){erases.push(i.surface);pending.delete(i.surface);}else preexistingErases.push(i.surface);}
 }
 ok(h.done&&h.success,'synthetic host did not reach goal');
 ok(flips.length===Number(!!arm.flipA)+Number(!!arm.flipB),'unsignalled drift did not happen');
 ok(errors.length===flips.length,'each drift must cause one recoverable mistaken action');
 ok(erases.length===errors.length&&!pending.size,'wrong material must be physically erased '+arm.id);
 ok(attempts.length===2+errors.length,'each mission target needs one successful painting');
 ok(attempts.filter(z=>z.correct).length===2,'both actual materials must be correct');
 for(const z of errors)ok(erases.includes(z.surface),'no recovery erase');
 return {rows,attempts,errors,erases,preexistingErases,flips,firstA,firstB};}
function ablateSurface(i,surface){const x=cp(i);x.durable=x.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[4]===surface));return x;}
function world(w){const traces=arms.map(a=>rollout(prep(w,a),a));
 const t=traces,p=t.map(x=>x.attempts);
 ok(t[0].firstA.action===t[1].firstA.action&&H(t[0].firstA.input)===H(t[1].firstA.input),'unannounced A drift leaked before first action');
 ok(t[0].firstB.action===t[2].firstB.action&&H(t[0].firstB.input)===H(t[2].firstB.input),'unannounced B drift leaked before first action');
 ok(t[0].firstA.action!==t[4].firstA.action&&t[1].firstA.action!==t[5].firstA.action,'A old calibration ignored');
 ok(t[0].firstB.action!==t[6].firstB.action&&t[2].firstB.action!==t[7].firstB.action,'B old calibration ignored');
 ok(t[0].rows[0].action!==t[8].rows[0].action&&t[0].rows[0].action!==t[9].rows[0].action,'retarget/gate contrasts lost');
 ok(t[0].firstA.action===t[10].firstA.action&&t[0].firstB.action===t[11].firstB.action,'obsolete physical receipts changed command');
 ok(p[0].every(a=>a.correct)&&p[4].every(a=>a.correct)&&p[6].every(a=>a.correct),'static controls should not require recovery');
 ok(t[1].errors.length===1&&t[2].errors.length===1&&t[3].errors.length===2&&t[5].errors.length===1&&t[7].errors.length===1,'unexpected feedback recovery count');
 // Under equal initial calibrations the latest physical outcome is the only
 // differentiator after the surprised paint; no simulator drift bit was issued.
 const example=t[1].rows.find(z=>z.action==='ERASE'&&z.input.surface==='A');
 ok(example&&example.input.painted.some(([pos,mat])=>mat==='B'&&JSON.stringify(pos)===JSON.stringify(example.input.p)),'physical wrong-material evidence missing');
 const post=t[1].rows.find((z,k)=>k>0&&z.action==='PAINT_B'&&z.input.surface==='A'&&z.input.painted.every(([pos])=>JSON.stringify(pos)!==JSON.stringify(z.input.p)));
 ok(post&&inferredLaw(post.input)==='SWAPPED','teacher did not recover mapping from actual action outcome');
 const meta={synthetic_not_native:true,source_world_index:w.index,source_seed:w.seed,source_split:w.split,geometry_sha256:w.geometry_sha256,
  arms:arms.map(a=>a.id),terminal_synthetic_episodes:t.length,decision_counts:t.map(x=>x.rows.length),
  drift_flags_EVALUATOR_ONLY:arms.map(a=>[!!a.flipA,!!a.flipB]),recovery_erases:t.map(x=>x.erases.length),
  unexpected_wrong_paint:t.map(x=>x.errors.length),correct_paints:t.map(x=>x.attempts.filter(a=>a.correct).length),
  initial_actions:t.map(x=>x.rows[0].action),first_paint_actions:t.map(x=>x.firstA.action),second_paint_actions:t.map(x=>x.firstB.action),
  causal_unsignalled_A_drift:[0,1],causal_unsignalled_B_drift:[0,2],double_drift:[0,3],
  opposite_prior_A_and_drift:[4,5],opposite_prior_B_and_drift:[6,7],mission_counterfactual:[0,8],gate_counterfactual:[0,9],
  obsolete_A_negative:[0,10],obsolete_B_negative:[0,11],
  pre_drift_A_equal_input_sha256:H(t[0].firstA.input),pre_drift_B_equal_input_sha256:H(t[0].firstB.input)};
 return {traces:t,meta};}
function save(dir,f,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,f),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output dir exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},geoms=new Map(),inputs=new Map(),nearNavigation=new Map();let total=0;
 for(const [split,n] of Object.entries(SPLITS)){
  const count=quick?Math.min(2,n):n,policy=[],challenge=[];let zero=0,nearZero=0,terminals=0,wrong=0,erase=0;
  for(let ix=0;ix<count;ix++){
   // Entirely different v46 candidate families from v46-v49; retain source split.
   const w=v46.candidate(split,200+ix).w,old=geoms.get(w.geometry_sha256);
   ok(!old||old===split,'cross-split geometry collision');geoms.set(w.geometry_sha256,split);
   const x=world(w);challenge.push(x.meta);
   for(const trace of x.traces){terminals++;wrong+=trace.errors.length;erase+=trace.erases.length;
    for(const row of trace.rows){const key=H(row.input),oldInput=inputs.get(key);
     ok(!oldInput||oldInput.split===split,'cross-split input leak');ok(!oldInput||oldInput.action===row.action,'contradictory label');
     if(oldInput&&split==='TRAIN'){row.weight=0;zero++;}else inputs.set(key,{split,action:row.action});
     if(split==='TRAIN'&&row.action.startsWith('MOVE_')){const sig=H(ablateSurface(ablateSurface(row.input,'A'),'B')),prev=nearNavigation.get(sig);
      ok(!prev||prev===row.action,'calibration-ablated movement contradiction');
      if(prev){if(row.weight!==0)nearZero++;row.weight=0;}else nearNavigation.set(sig,row.action);}
     policy.push(row);
    }
   }
  }
  shards[split+'_policy.jsonl']=save(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=save(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,synthetic_terminal_episodes:terminals,policy_decisions:policy.length,
    unsignalled_A_drift_pairs:count,unsignalled_B_drift_pairs:count,double_drift_pairs:count,
    backwards_prior_drift_pairs:2*count,stale_probe_controls:2*count,mission_pairs:count,gate_pairs:count,
    wrong_physical_paints:wrong,host_erases_after_wrong_paint:erase,
    train_exact_duplicate_zero_weight:zero,train_near_duplicate_movement_zero_weight:nearZero};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/online-recovery-v50/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'online_recovery_v50.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contract:{input:'ONLY policy.input is model-visible; action and weight are supervised labels. Never train on *_challenge_EVALUATOR_ONLY.jsonl.',
   world_group:'Group by (source_split,source_world_index,source_seed,geometry_sha256), and explicitly compare geometry hashes across all v44-v50 releases. v50 source-world indices start at 200 and are not used by v46-v49.',
   drift:'Physical host flips the contextual material mapping at first arrival at an objective without issuing a change event. Teacher cannot see flip; it initially issues the previously calibrated command, reads physical outcome, erases wrong material and corrects using that new observation.',
   leakage:'No hidden drift flags or host laws are in model input. Episode-labelled world / arm / expected material / error counts occur only in evaluator metadata.',
   dedup:'TRAIN exact repeated inputs and movement records that differ only by irrelevant A/B physical receipts are zero-weighted. DEV/HELDOUT retained unsampled.',
   labels:'Every trajectory action was legal and host accepted; PAINT_RESULT is generated only after actual synthetic accepted paint, not an oracle tool-law signal.'},
  splits,shards,total_policy_decisions:total,source_worlds:geoms.size,
  limitations:['Synthetic graph/material game-like host, not native CyberSand','Pre-mission calibration and early survey remain scripted host operations',
   'Binary contextual tool laws permit symbolic inference; model training and actual transfer remain unevaluated','Unannounced mid-objective drift is synthetic; native game semantics require separate validation',
   'A simple online receipt parser can still solve recovery; this is a discriminative DATA challenge, not learned behavior',
   'No LFM/Sprocket training or neural latency/RSS; dataset not in immutable Actions artifacts']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){const [out,flag]=process.argv.slice(2);ok(out,'usage node online_recovery_v50.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(out),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,arms,prep,observe,command,inferredLaw,teacher,maybeDrift,rollout,ablateSurface,world,generate,H};
