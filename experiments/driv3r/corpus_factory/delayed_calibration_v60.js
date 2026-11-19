#!/usr/bin/env node
'use strict';
/* Corpus Factory v60 — host-accepted, memory-stressing material decisions.
   DATA ONLY; synthetic v53 graph host != native CyberSand.
   Generate paired B-law contrasts with the SAME final instantaneous observation,
   after a long legal detour. Detour steps are interventions (zero loss), not
   falsely labelled optimal policy navigation. Physical calibration is actually
   executed, observed once, then absent from the final delta.
*/
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v51=require('./active_probe_v51.js'),v58=require('./offpolicy_repair_v58.js');
const v46=require('./mission_memory_v46.js');
const eq=(x,y)=>JSON.stringify(x)===JSON.stringify(y);
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(c,s)=>{if(!c)throw Error('v60 '+s)};
const pairs=[[1,2],[9,10],[14,15]];
const counts={TRAIN:12,DEV:6,HELDOUT:6};
const delays={TRAIN:[48,96,128],DEV:[64,128,160],HELDOUT:[96,192,256]};
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const deps=['deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js','offpolicy_repair_v58.js'];
const shaFile=p=>H(fs.readFileSync(p));
function choices(h){return v53.observe(h).legal.filter(z=>z.startsWith('MOVE_')).sort();}
function run(w,arm,delay,salt){
 const pars=v53.armParams(arm),h=v53.prep(w,arm);let previous=[],t=0,detoured=false;
 const steps=[];let phys=[],lastProbeAt=null,final=null,forced=0,branches=0;
 const append=(a,weight,focus=0)=>{
  const alert=v52.notice(h,pars);v53.injectB(h,arm,alert,t);
  const full=v53.observe(h),input=v58.streamify(full,previous);
  ok(full.legal.includes(a),'illegal action '+a+' step '+t);
  previous=full.durable;steps.push({input,action:a,weight,focus_loss_weight:focus});
  const before=h.stage,pos=h.pos,res=v52.command(h,a,pars);t++;
  if(a.startsWith('PAINT_')){
   phys.push({action:a,material:res.material,surface:before===0?'A':before===1?'B':null,site:h.g.coords[pos]});
   if(before===1&&pos!==h.w.goal2)lastProbeAt=t;
  }
  if(h.done)ok(h.success,'non-success host terminal');
  return {input,res,before,pos};
 };
 const step=(a,weight,focus)=>append(a,weight,focus);
 function detour(){
  // Construct nontrivial outward excursions and exact retraces; each branch is
  // host-accepted, with no invented observation and no synthetic action token.
  let remaining=delay,segment=0;
  while(remaining>0){
   const origin=h.pos,forward=[];
   const maxDepth=Math.min(4,remaining/2);
   let last=null;
   for(let k=0;k<maxDepth;k++){
    const available=choices(h).filter(a=>a!==reverse[last]);
    if(!available.length)break;
    const a=available[(w.seed+salt*17+segment*11+k*7)%available.length];
    step(a,0,0);forward.push(a);last=a;
   }
   ok(forward.length>0,'no detour legal move');
   if(forward.length>1)branches++;
   for(let k=forward.length-1;k>=0;k--)step(reverse[forward[k]],0,0);
   ok(h.pos===origin,'detour did not retrace');
   remaining-=2*forward.length;segment++;
   ok(segment<400,'detour unbounded');
  }
  ok(h.pos===h.w.goal2,'detour left B goal');
  forced=delay;
 }
 while(!h.done&&t<900){
  const alert=v52.notice(h,pars);v53.injectB(h,arm,alert,t);
  const full=v53.observe(h),a=v51.teacher(full);
  ok(full.legal.includes(a),'oracle illegal');
  if(!detoured&&h.stage===1&&h.pos===h.w.goal2&&a.startsWith('PAINT_')){
   ok(lastProbeAt!==null,'missing actual B probe');
   detour();detoured=true;continue;
  }
  const record=step(a,1,a.startsWith('PAINT_')&&h.stage===1&&h.pos===h.w.goal2?8:1);
  if(detoured&&!final&&h.stage===2&&a.startsWith('PAINT_')){
   final={index:steps.length-1,action:a,input:record.input,delay_accepted:record.input.recent.length,
    last_probe_age:t-lastProbeAt,material:record.res.material};
  }
 }
 ok(h.done&&h.success&&!h.protected_damage,'not host success');
 ok(detoured&&final&&forced===delay,'detour/goal missing');
 ok(steps[final.index].action===final.action&&final.material==='B','final paint not material B');
 ok(!final.input.events_delta.some(x=>x[1]==='PAINT_RESULT'&&x[4]==='B'),'final action leaks B probe receipt');
 ok(final.last_probe_age>=delay,'physical evidence not old');
 ok(steps.filter(x=>x.weight===0).length===delay,'forced detour weights wrong');
 return {steps,final,phys,branches,forced};
}
function write(out,name,rows,files){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});files[name]={sha256:H(b),bytes:b.length,rows:rows.length};}
function generate(out,{quick=false}={}){
 ok(!fs.existsSync(out),'output exists');fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/delayed-material-memory-v60/1',date:'2026-11-19',
  data_only:true,synthetic_not_native_CyberSand:true,quick,
  generator_sha256:shaFile(__filename),parent_shas:Object.fromEntries(deps.map(f=>[f,shaFile(path.join(__dirname,f))])),
  provenance:'Reconstructed v53 host/source geometry from v46 candidate(split,700+world_index), with v53 paired material-law arms; no host ABI additions.',
  input_contract:{visible:'Only ordered steps[].input and past actions; one-shot events_delta; NEVER supply evaluator-only certificates to model.',
   train:'TRAIN_delayed_memory_stream.jsonl only; DEV/HELDOUT are evaluator-only.',
   weight:'Host-forced exploration/retrace detour actions have weight=0. Physical calibration and final material decisions are supervised. Distinguish policy from intervention.',
   split:'Geometry/world ownership stable across v53/v56/v59; held-out world geometries never train. DEV/HELDOUT sealed.',
   challenge:'At B goal, paired final instantaneous inputs are byte-identical but one requires PAINT_A and the other PAINT_B. Old one-shot B paint receipt is causally necessary.',
   scoring:'Report paired both-correct and per-arm action accuracy, plus closed-loop host completion after final material action; do not use teacher success as neural performance.'},
  limitations:['Synthetic host; forced detours are navigation interventions, not optimal movement demonstrations.',
   'No actual target SLM trained and no native CyberSand gameplay demonstrated.',
   'Host is a graph with immutable walls and surface laws between probes; long detours alone do not prove realistic open-world memory.',
   'No immutable Actions publication or neural CPU latency/RSS in this corpus lane.'],files:{},splits:{}};
 const geometry=new Map(),splitInput=new Map();let total=0,allPairs=0;
 for(const [split,n] of Object.entries(counts)){
  const streams=[],certs=[],critical=[];let repeated=0,forced=0,valid=0,branches=0;
  for(let ix=0;ix<(quick?Math.min(2,n):n);ix++){
   const w=v46.candidate(split,700+ix).w,owner=geometry.get(w.geometry_sha256);
   ok(!owner||owner===split,'geometry leakage');geometry.set(w.geometry_sha256,split);
   for(let qi=0;qi<pairs.length;qi++){
    const [ai,bi]=pairs[qi],delay=delays[split][qi];
    const left=run(w,v53.ARMS[ai],delay,qi),right=run(w,v53.ARMS[bi],delay,qi);
    ok(eq(left.final.input,right.final.input),'paired final input differs split '+split+' idx '+ix+' pair '+qi);
    ok(left.final.action!==right.final.action,'no opposing material label');
    ok(left.final.last_probe_age>=delay&&right.final.last_probe_age>=delay,'short memory');
    for(const [side,result] of [[0,left],[1,right]]){
     const episode=streams.length;
     // No exact repeated prefix is supervised twice; retain full context to train memory.
     for(let j=0;j<result.steps.length;j++){
      const row=result.steps[j],hash=H(row.input),old=splitInput.get(hash);
      if(row.weight===0)continue;
      ok(!old||old.split===split,'cross-split input collision');
      if(old&&old.action===row.action&&split==='TRAIN'&&row.weight>0&&j!==result.final.index){row.weight=0;row.focus_loss_weight=0;repeated++;}
      else if(!old)splitInput.set(hash,{split,action:row.action});
      else if(old.action!==row.action){/* Legal history-dependent label: do not deduplicate. */}
     }
     // Never erase supervised critical target even if repeated input occurs.
     ok(result.steps[result.final.index].weight===1,'critical final paint masked');
     streams.push({steps:result.steps});valid++;forced+=result.forced;branches+=result.branches;
     certs.push({episode,source_split:split,world_index:ix,source_seed:w.seed,geometry_sha256:w.geometry_sha256,
      pair_id:split+':'+ix+':'+qi,side,arm:v53.ARMS[side===0?ai:bi].id,delay_actions:delay,
      critical_step:result.final.index,final_action:result.final.action,calibration_age_accepted_actions:result.final.last_probe_age,
      paired_current_input_sha256:H(result.final.input),host_terminal_success:true,host_correct_goal_material:result.final.material,
      forced_host_accepted_detour_actions:result.forced,nontrivial_excursions:result.branches});
     critical.push({episode,step:result.final.index,action:result.final.action,pair_id:split+':'+ix+':'+qi});
    }
    allPairs++;
   }
  }
  const data=split+(split==='TRAIN'?'_delayed_memory_stream.jsonl':'_EVALUATOR_ONLY_delayed_memory_stream.jsonl');
  const info=split+'_delayed_memory_certificates_EVALUATOR_ONLY.jsonl';
  write(out,data,streams,manifest.files);write(out,info,certs,manifest.files);
  manifest.splits[split]={worlds:quick?Math.min(2,n):n,episodes:streams.length,material_contrast_pairs:streams.length/2,
   final_opposing_commands:streams.length,host_terminal_success:valid,decisions:streams.reduce((a,e)=>a+e.steps.length,0),
   intervention_actions_loss_masked:forced,nontrivial_excursions:branches,repeat_training_steps_loss_masked:repeated,
   contrast_ages_actions:delays[split]};
  total+=manifest.splits[split].decisions;
 }
 manifest.total_decisions=total;manifest.total_material_pairs=allPairs;manifest.distinct_geometry=geometry.size;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
if(require.main===module){const [out,quick]=process.argv.slice(2);ok(out,'usage: node delayed_calibration_v60.js OUTPUT_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(out),{quick:quick==='--quick'}).splits,null,2));}
module.exports={generate,run,pairs,counts,delays};
