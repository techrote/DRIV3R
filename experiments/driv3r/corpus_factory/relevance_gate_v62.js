#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v62: post-calibration alert attribution and temporal
 * validity. Synthetic v53 graph host, NOT native CyberSand. DATA, NOT policy.
 * All probe/painter actions executed against the host. Long detours zero loss. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v51=require('./active_probe_v51.js'),v58=require('./offpolicy_repair_v58.js');
const v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error('V62 '+m)};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const COUNTS={TRAIN:8,DEV:4,HELDOUT:4};
const DELAYS={TRAIN:[64],DEV:[128],HELDOUT:[256]};
const CONDITIONS=['none','irrelevant_A','realert_B_stable','realert_B_flip'];
const parents=['deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','offpolicy_repair_v58.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
function run(w,oldSwapped,midSwapped,condition,delay,salt){
 ok(CONDITIONS.includes(condition),'invalid condition');
 const arm=v53.ARMS[oldSwapped?2:1],pars=v53.armParams(arm),h=v53.prep(w,arm);
 let t=0,prior=[],firstErased=false,movesSinceFirst=0,secondAlert=-1,lateInjected=false;
 const steps=[],probes=[];let checkpoint=null,final=null,excursions=0,forced=0,lateProbeIndex=null;
 function signals(){
  const n=v52.notice(h,pars);v53.injectB(h,arm,n,t);
  if(firstErased&&movesSinceFirst>=1&&secondAlert<0&&h.stage===1&&h.pos!==h.siteB){
   ok(h.painted[h.siteB]===undefined,'second alert with painted safe cell');
   h.laws.B=midSwapped?'SWAPPED':'NORMAL';
   h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
   secondAlert=t;
  }
 }
 function step(action,weight,focus=0){
  signals();const obs=v53.observe(h),input=v58.streamify(obs,prior);
  ok(obs.legal.includes(action),'illegal '+action+' at '+t);
  const idx=steps.length,pre=h.stage,at=h.pos;
  prior=obs.durable;steps.push({input,action,weight,focus_loss_weight:focus});
  const r=v52.command(h,action,pars);t++;
  if(pre===1&&at===h.siteB&&action==='PAINT_A'){
   ok(r.material==='A'||r.material==='B','invalid physical safe probe');
   probes.push({index:idx,at:t,material:r.material});
   if(probes.length===3)lateProbeIndex=idx;
  }
  if(pre===1&&at===h.siteB&&action==='ERASE'&&probes.length===1)firstErased=true;
  if(firstErased&&secondAlert<0&&action.startsWith('MOVE_'))movesSinceFirst++;
  if(pre===1&&at===h.w.goal2&&action.startsWith('PAINT_'))final={index:idx,action,material:r.material};
  ok(!h.done||h.success,'non-success host terminal');
 }
 function detour(){
  let remain=delay,segment=0;
  while(remain>0){
   const origin=h.pos,forward=[];let last=null;
   for(let k=0;k<Math.min(4,remain/2);k++){
    signals();const legal=v53.observe(h).legal.filter(x=>x.startsWith('MOVE_')&&x!==reverse[last]).sort();
    if(!legal.length)break;
    const a=legal[(w.seed+salt*17+segment*11+k*7)%legal.length];
    step(a,0);forward.push(a);last=a;
   }
   ok(forward.length>0,'detour blocked');
   if(forward.length>1)excursions++;
   for(let k=forward.length-1;k>=0;k--)step(reverse[forward[k]],0);
   ok(h.pos===origin,'detour did not retrace');
   remain-=forward.length*2;segment++;
   ok(segment<400,'unbounded detour');
  }
  ok(h.pos===h.w.goal2&&steps.slice(-delay).every(s=>s.weight===0),'invalid fixed detour');forced=delay;
 }
 while(!h.done&&t<1200){
  signals();const obs=v53.observe(h),teacher=v51.teacher(obs);
  ok(obs.legal.includes(teacher),'illegal oracle '+teacher);
  if(!lateInjected&&h.stage===1&&h.pos===h.w.goal2&&teacher.startsWith('PAINT_')&&probes.length===2){
   // A late ALERT is delivered exactly once at B's objective, then aged by a
   // host-executed detour. An A alert is a non-target distractor, not B invalidation.
   if(condition==='irrelevant_A')h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'A',site:h.g.coords[h.w.s]});
   if(condition.startsWith('realert_B')){
    if(condition==='realert_B_flip')h.laws.B=h.laws.B==='NORMAL'?'SWAPPED':'NORMAL';
    h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
   }
   lateInjected=true;detour();
   signals();const snap=v53.observe(h),modelInput=v58.streamify(snap,prior);
   ok(modelInput.events_delta.length===0,'historical alert leaked at checkpoint');
   const needed=v51.teacher(snap);ok(snap.legal.includes(needed),'critical action illegal');
   checkpoint={index:steps.length,input:modelInput,action:needed,age:delay,
    alert:condition,law_before:midSwapped?'SWAPPED':'NORMAL'};
   ok(needed.startsWith('MOVE_')===condition.startsWith('realert_B'),'invalid temporal alert decision');
   continue;
  }
  const critical=checkpoint&&steps.length===checkpoint.index;
  step(teacher,1,critical?8:teacher.startsWith('PAINT_')&&h.stage===1?4:1);
 }
 ok(h.done&&h.success&&!h.protected_damage,'host failed terminal');
 ok(checkpoint&&final&&lateInjected&&secondAlert>=0,'incomplete warning experiment');
 ok(probes.length===(condition.startsWith('realert_B')?3:2),'incorrect physical probe count');
 ok(probes[0].material===(oldSwapped?'B':'A')&&probes[1].material===(midSwapped?'B':'A'),'first/second probe wrong law');
 if(condition.startsWith('realert_B')){
  ok(lateProbeIndex!==null&&probes[2].index>checkpoint.index,'missing after-alert physical re-probe');
  ok(probes[2].material===(condition==='realert_B_flip'?(midSwapped?'A':'B'):(midSwapped?'B':'A')),'third probe did not track law');
 }
 ok(final.material==='B'&&final.action===((condition==='realert_B_flip'? !midSwapped:midSwapped)?'PAINT_A':'PAINT_B'),'goal material wrong');
 ok(steps[checkpoint.index].action===checkpoint.action&&steps[checkpoint.index].weight===1&&steps[checkpoint.index].focus_loss_weight===8,'critical supervision mismatch');
 ok(steps.filter(s=>s.weight===0).length===delay,'non-intervention movement incorrectly masked');
 return {steps,checkpoint,probes,final,forced,excursions};
}
function write(out,name,rows,files){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});files[name]={sha256:H(b),bytes:b.length,rows:rows.length};}
function generate(out,quick=false){
 ok(!fs.existsSync(out),'output exists');fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/relevance-gated-memory-v62/1',date:'2026-11-19',quick,
  synthetic_not_native_CyberSand:true,source_generator_sha256:H(fs.readFileSync(__filename)),
  parent_shas:Object.fromEntries(parents.map(f=>[f,H(fs.readFileSync(path.join(__dirname,f)))])),
  contract:{train:'TRAIN_relevance_stream.jsonl only; DEV/HELDOUT evaluator-only, never fit.',
   visible:'Only each steps[].input in sequence and previous agent actions; no hidden laws, variant identities or sidecars.',
   contrast:'Identical critical instantaneous host observation across OLD x MID material laws and post-calibration {none, irrelevant_A, B_no_drift, B_flip}.',
   outcome:'B warning after latest B probe requires another host-executed B probe even when no physical drift. Unrelated A warning does not invalidate B evidence.',
   actions:'Legal host MOVE, PAINT_A, PAINT_B, ERASE, STOP, no invented action tokens.',
   intervention:'Long excursions/retraces execute in synthetic host; zero supervised weight. Alerts synthetically injected, never claimed native.',
   scoring:'Critical action split between current-paint and move-to-probe; full host completion, per-16-arm all-correct, false alarm handling; ablate surface and relative timestamp.',
   leakage:'Group by original v46 geometry_sha256 and split, event deltas once, evaluator certificates only in sidecars.',
   negative:'Current-only majority ceiling 50% at critical action; latest-probe-only ignores late B invalidation and is wrong for 50%.',
   cross_release:'v61 world family indices 700+; do not mix v61 held-out with v62 training.'},
  files:{},splits:{},limitations:['Synthetic host, not native CyberSand. Late surface warning injected by experiment.',
   'Timestamped per-surface symbolic event/probe controller solves benchmark; not neural performance.',
   'Long detours are scripted zero-weight host interventions, not learned navigation.',
   'No SLM trained, no native trial, no CPU neural latency/RSS or immutable Actions data artifact.']};
 const geometry=new Map(),seenTrain=new Map();let total=0,sets=0;
 for(const [split,n] of Object.entries(COUNTS)){
  const streams=[],certs=[];let duplicated=0,forced=0;
  for(let ix=0;ix<(quick?Math.min(1,n):n);ix++){
   const w=v46.candidate(split,700+ix).w;
   ok(!geometry.has(w.geometry_sha256)||geometry.get(w.geometry_sha256)===split,'cross-split geometry collision');geometry.set(w.geometry_sha256,split);
   for(let di=0;di<DELAYS[split].length;di++){
    const delay=DELAYS[split][di],runs=[];
    for(const old of [false,true])for(const mid of [false,true])for(const condition of CONDITIONS){
     const r=run(w,old,mid,condition,delay,di);runs.push({old,mid,condition,r});
    }
    ok(runs.length===16&&runs.every(x=>eq(x.r.checkpoint.input,runs[0].r.checkpoint.input)),'16-arm critical inputs mismatch');
    const actions=runs.map(x=>x.r.checkpoint.action),counts=new Map();for(const a of actions)counts.set(a,(counts.get(a)||0)+1);
    ok(actions.filter(a=>a.startsWith('MOVE_')).length===8,'unbalanced reprobe action');
    ok(actions.filter(a=>a==='PAINT_A').length===4&&actions.filter(a=>a==='PAINT_B').length===4,'unbalanced material actions');
    ok(counts.size===3,'expected MOVE PAINT_A PAINT_B critical choices');
    for(const x of runs){
     const ep=streams.length,r=x.r;
     for(let j=0;j<r.steps.length;j++){
      const st=r.steps[j];if(st.weight===0)continue;
      const key=H(st.input),seen=seenTrain.get(key);
      ok(!seen||seen.split===split,'split leakage of model-visible observation');
      if(seen&&seen.action===st.action&&split==='TRAIN'&&j!==r.checkpoint.index){st.weight=0;st.focus_loss_weight=0;duplicated++;}
      else if(!seen)seenTrain.set(key,{split,action:st.action});
     }
     ok(r.steps[r.checkpoint.index].weight===1,'critical action was deduplicated');
     streams.push({steps:r.steps});forced+=r.forced;
     certs.push({episode:ep,group_id:split+':'+ix+':'+di,split,world_index:ix,seed:w.seed,geometry_sha256:w.geometry_sha256,
      old_swapped:x.old,mid_swapped:x.mid,condition:x.condition,forced_detour_actions:r.forced,
      critical_step:r.checkpoint.index,critical_action:r.checkpoint.action,critical_input_sha256:H(r.checkpoint.input),
      probe_materials:r.probes.map(p=>p.material),probe_steps:r.probes.map(p=>p.index),final_action:r.final.action,
      host_terminal_success:true});
    }
    sets++;
   }
  }
  const main=split+(split==='TRAIN'?'_relevance_stream.jsonl':'_EVALUATOR_ONLY_relevance_stream.jsonl');
  const side=split+'_relevance_certificates_EVALUATOR_ONLY.jsonl';
  write(out,main,streams,manifest.files);write(out,side,certs,manifest.files);
  const setsN=streams.length/16;
  manifest.splits[split]={worlds:quick?Math.min(n,1):n,sets:setsN,episodes:streams.length,
   decisions:streams.reduce((n,x)=>n+x.steps.length,0),host_successes:streams.length,
   post_calibration_B_alerts:certs.filter(c=>c.condition.startsWith('realert_B')).length,
   irrelevant_A_alerts:certs.filter(c=>c.condition==='irrelevant_A').length,
   correct_no_alert_paints:certs.filter(c=>c.condition==='none').length,
   loss_masked_detour_moves:forced,duplicate_train_actions_masked:duplicated,
   delays:DELAYS[split]};
  total+=manifest.splits[split].decisions;
 }
 manifest.total_decisions=total;manifest.total_sets=sets;manifest.distinct_world_geometries=geometry.size;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
if(require.main===module){const [out,flag]=process.argv.slice(2);ok(out,'usage: node relevance_gate_v62.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(out),flag==='--quick').splits,null,2));}
module.exports={run,generate,COUNTS,DELAYS,CONDITIONS};
