#!/usr/bin/env node
'use strict';
/* Corpus Factory v61: two sequential host-executed B calibrations, a later
 * non-diagnostic alert, and 2x2 OLD/NEW physical-law counterfactual quartets.
 * DATA ONLY. Synthetic v53 graph host, NOT native CyberSand. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v51=require('./active_probe_v51.js'),v58=require('./offpolicy_repair_v58.js');
const v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error('V61 '+m)};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const COUNTS={TRAIN:12,DEV:6,HELDOUT:6};
const DELAYS={TRAIN:[48,96],DEV:[96,160],HELDOUT:[192,256]};
const parents=['deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','offpolicy_repair_v58.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
function run(w,oldSwapped,newSwapped,delay,salt){
 const arm=v53.ARMS[oldSwapped?2:1],pars=v53.armParams(arm),h=v53.prep(w,arm);
 let t=0,prior=[],initialProbe=0,firstErased=false,movesAfterFirst=0,secondAlert=-1;
 let probes=[],final=null,detoured=false,forced=0,excursions=0;
 const steps=[];
 function signals(){
  const n=v52.notice(h,pars);v53.injectB(h,arm,n,t);
  if(firstErased&&movesAfterFirst>=1&&secondAlert<0&&h.stage===1&&h.pos!==h.siteB){
   ok(h.painted[h.siteB]===undefined,'late warning leaves painted probe');
   h.laws.B=newSwapped?'SWAPPED':'NORMAL';
   h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
   secondAlert=t;
  }
 }
 function step(action,weight,focus=0){
  signals();const obs=v53.observe(h),input=v58.streamify(obs,prior);
  ok(obs.legal.includes(action),'illegal '+action+' at '+t);
  prior=obs.durable;steps.push({input,action,weight,focus_loss_weight:focus});
  const beforeStage=h.stage,at=h.pos,r=v52.command(h,action,pars);t++;
  if(beforeStage===1&&at===h.siteB&&action==='PAINT_A'){
   ok(r.material==='A'||r.material==='B','no physical outcome');
   probes.push({at:t,material:r.material});initialProbe++;
  }
  if(beforeStage===1&&at===h.siteB&&action==='ERASE'&&initialProbe===1)firstErased=true;
  if(firstErased&&secondAlert<0&&action.startsWith('MOVE_'))movesAfterFirst++;
  if(beforeStage===1&&at===h.w.goal2&&action.startsWith('PAINT_')&&detoured){
   final={index:steps.length-1,input,action,material:r.material,age:t-probes.at(-1).at};
  }
  ok(!h.done||h.success,'non-success host terminal');
 }
 function detour(){
  let left=delay,segment=0;
  while(left>0){
   const origin=h.pos,forward=[];let last=null;
   const depth=Math.min(4,left/2);
   for(let k=0;k<depth;k++){
    signals();const legal=v53.observe(h).legal.filter(x=>x.startsWith('MOVE_')&&x!==reverse[last]).sort();
    if(!legal.length)break;
    const a=legal[(w.seed+salt*17+segment*11+k*7)%legal.length];
    step(a,0,0);forward.push(a);last=a;
   }
   ok(forward.length>0,'detour blocked');if(forward.length>1)excursions++;
   for(let k=forward.length-1;k>=0;k--)step(reverse[forward[k]],0,0);
   ok(h.pos===origin,'detour retrace incorrect');left-=2*forward.length;segment++;
   ok(segment<400,'detour unbounded');
  }
  ok(h.pos===h.w.goal2,'detour left objective');forced=delay;
 }
 while(!h.done&&t<1050){
  signals();const obs=v53.observe(h),action=v51.teacher(obs);
  ok(obs.legal.includes(action),'illegal oracle '+action);
  if(!detoured&&h.stage===1&&h.pos===h.w.goal2&&action.startsWith('PAINT_')&&probes.length===2){
   ok(secondAlert>=0,'no second alert');detour();detoured=true;continue;
  }
  step(action,1,action.startsWith('PAINT_')&&h.stage===1&&h.pos===h.w.goal2?8:1);
 }
 ok(h.done&&h.success&&!h.protected_damage,'not successful');
 ok(secondAlert>=0&&probes.length===2&&final&&detoured,'missing second probe/final');
 ok(probes[0].material===(oldSwapped?'B':'A')&&probes[1].material===(newSwapped?'B':'A'),'probe did not track physical law');
 ok(final.material==='B'&&final.action===(newSwapped?'PAINT_A':'PAINT_B'),'goal law incorrect');
 ok(final.age>=delay&&steps.filter(x=>x.weight===0).length===delay,'memory/intervention span incorrect');
 ok(!final.input.events_delta.some(x=>x[1]==='PAINT_RESULT'&&x[4]==='B'),'physical event leaked at final input');
 return {steps,secondAlert,probes,final,forced,excursions};
}
function write(out,name,rows,files){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});files[name]={sha256:H(b),bytes:b.length,rows:rows.length};}
function generate(out,quick=false){
 ok(!fs.existsSync(out),'output already exists');fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/last-valid-physical-probe-v61/1',date:'2026-11-19',quick,
  synthetic_not_native_CyberSand:true,source_generator_sha256:H(fs.readFileSync(__filename)),
  parent_shas:Object.fromEntries(parents.map(f=>[f,H(fs.readFileSync(path.join(__dirname,f)))])),
  contract:{train:'TRAIN_last_probe_stream.jsonl only; DEV and HELDOUT evaluator-only.',
   model_visible:'Only ordered steps[].input and previous actions. Evaluator sidecar, surface-law booleans and certificates must NEVER be model inputs.',
   labels:'Every action accepted by synthetic host. Detour movements weight=0; second probe, safe erase and critical goal paint supervised.',
   challenge:'Independent OLD and NEW B law produces four matched causal arms; final current input identical; only latest physical calibration is valid after second alert.',
   split:'Group by geometry_sha256 and source_seed across releases; source v46 worlds indices 700+; no split cross-contamination.',
   metrics:'Report critical quartet accuracy, all-four-correct quartets, stale-first-only accuracy, autonomous completion, hidden-state ablation and neural CPU RSS/latency if trained.',
   causality:'Synthetic host executes both PAINT_A safe probes, erases both, then executes final goal paint. Second alert may change B law independent of prior law.'},
  limitations:['Synthetic graph host and externally injected second alert are NOT native CyberSand.',
   'Forced long retraces are masked interventions, not learned navigation demonstrations.',
   'A symbolic last-alert/last-probe receipt controller remains sufficient; no neural weights trained or native performance measured.',
   'No immutable Actions data artifacts; consumers reproduce with pinned source and SHA256 manifests.'],files:{},splits:{}};
 const geometry=new Map(),inputs=new Map();let all=0,quartets=0;
 for(const [split,n] of Object.entries(COUNTS)){
  const streams=[],certs=[];let repeats=0,interventions=0,excursions=0;
  for(let ix=0;ix<(quick?Math.min(2,n):n);ix++){
   const w=v46.candidate(split,700+ix).w,owner=geometry.get(w.geometry_sha256);
   ok(!owner||owner===split,'geometry split leakage');geometry.set(w.geometry_sha256,split);
   for(let di=0;di<DELAYS[split].length;di++){
    const delay=DELAYS[split][di],runs=[];
    for(const old of [false,true])for(const current of [false,true])runs.push({old,current,result:run(w,old,current,delay,di)});
    ok(runs.every(x=>eq(x.result.final.input,runs[0].result.final.input)),'quartet current input mismatch');
    ok(new Set(runs.map(x=>x.result.final.action)).size===2,'no final action discrimination');
    for(const x of runs){
     const ep=streams.length,rs=x.result;
     for(let j=0;j<rs.steps.length;j++){
      const row=rs.steps[j];if(!row.weight)continue;
      const key=H(row.input),seen=inputs.get(key);
      ok(!seen||seen.split===split,'input split leakage');
      if(seen&&seen.action===row.action&&split==='TRAIN'&&j!==rs.final.index){row.weight=0;row.focus_loss_weight=0;repeats++;}
      else if(!seen)inputs.set(key,{split,action:row.action});
     }
     ok(rs.steps[rs.final.index].weight===1&&rs.steps[rs.final.index].focus_loss_weight===8,'critical supervised label lost');
     streams.push({steps:rs.steps});interventions+=rs.forced;excursions+=rs.excursions;
     certs.push({episode:ep,pair_id:split+':'+ix+':'+di,split,source_index:ix,seed:w.seed,geometry_sha256:w.geometry_sha256,
      old_swapped:x.old,new_swapped:x.current,first_material:rs.probes[0].material,last_material:rs.probes[1].material,
      second_alert_step:rs.secondAlert,first_probe_step:rs.probes[0].at,second_probe_step:rs.probes[1].at,
      critical_step:rs.final.index,final_action:rs.final.action,calibration_age:rs.final.age,
      paired_current_input_sha256:H(rs.final.input),forced_detour_actions:rs.forced,
      host_terminal_success:true});
    }
    quartets++;
   }
  }
  const data=split+(split==='TRAIN'?'_last_probe_stream.jsonl':'_EVALUATOR_ONLY_last_probe_stream.jsonl');
  const side=split+'_last_probe_certificates_EVALUATOR_ONLY.jsonl';
  write(out,data,streams,manifest.files);write(out,side,certs,manifest.files);
  manifest.splits[split]={worlds:quick?Math.min(n,2):n,quartets:streams.length/4,episodes:streams.length,
   decisions:streams.reduce((a,e)=>a+e.steps.length,0),host_successes:streams.length,
   critical_command_A:certs.filter(c=>c.final_action==='PAINT_A').length,
   critical_command_B:certs.filter(c=>c.final_action==='PAINT_B').length,
   misleading_stale_first_probe:certs.filter(c=>c.first_material!==c.last_material).length,
   detour_interventions_loss_masked:interventions,nontrivial_excursions:excursions,
   duplicate_train_prefix_actions_loss_masked:repeats,delays:DELAYS[split]};
  all+=manifest.splits[split].decisions;
 }
 manifest.total_decisions=all;manifest.total_quartets=quartets;manifest.distinct_geometries=geometry.size;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage: node last_valid_probe_v61.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),flag==='--quick').splits,null,2));}
module.exports={run,generate,COUNTS,DELAYS};
