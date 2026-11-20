#!/usr/bin/env node
'use strict';
/**
 * DRIV3R Corpus Factory v58: host-executed off-policy repair curriculum.
 * DATA ONLY. The v53 graph host is synthetic and is not native CyberSand.
 *
 * Starts from a v56 physical-memory decision, forces the opposite LEGAL paint
 * command with zero supervised weight, then collects host-observed recovery.
 * Evaluator-only certificates are never included in policy inputs or TRAIN rows.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v51=require('./active_probe_v51.js'),v46=require('./mission_memory_v46.js');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const enc=o=>JSON.stringify(o);
const check=(v,s)=>{if(!v)throw new Error('V58 '+s)};
const splits=['TRAIN','DEV','HELDOUT'];
const read=p=>fs.readFileSync(p),lines=p=>read(p).toString('utf8').trim().split('\n').map(JSON.parse);
const names=(split,kind)=>kind==='episode' ? split+(split==='TRAIN'?'_causal_stream.jsonl':'_EVALUATOR_ONLY_causal_stream.jsonl') : split+'_'+kind+'_EVALUATOR_ONLY.jsonl';
const roots=['deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
function streamify(obs,prior) {
 check(Array.isArray(obs.durable) && Array.isArray(prior),'missing durable event list');
 check(obs.durable.length>=prior.length && enc(obs.durable.slice(0,prior.length))===enc(prior),'historical receipt prefix changed');
 const out={...obs};delete out.durable;out.v='synthetic-v54-event-stream';
 out.recent=out.recent.slice(-8);out.recent_positions=out.recent_positions.slice(-8);
 out.events_delta=obs.durable.slice(prior.length);
 return out;
}
function generateBranch(p,episode,ref,world) {
 const arm=v53.ARMS.find(a=>a.id===p.arm);check(arm,'arm missing');
 const par=v53.armParams(arm),h=v53.prep(world,arm);let previous=[];
 const original=episode.steps,steps=[];
 for(let t=0;t<=ref.source_step;t++) {
  const n=v52.notice(h,par);v53.injectB(h,arm,n,t);
  const obs=v53.observe(h),policy=streamify(obs,previous),saved=original[t];
  check(enc(policy)===enc(saved.input),'source observation mismatch '+p.arm+' step='+t);
  check(obs.legal.includes(saved.action),'illegal replay action');
  previous=obs.durable;
  if(t<ref.source_step){
   steps.push({input:saved.input,action:saved.action,weight:saved.weight,
    ...(saved.focus_loss_weight===undefined?{}:{focus_loss_weight:saved.focus_loss_weight})});
   v52.command(h,saved.action,par);continue;
  }
  check(saved.action==='PAINT_A'||saved.action==='PAINT_B','not a paint fork');
  check(h.pos===(h.stage===0?h.w.goal:h.w.goal2),'not protected goal');
  check(saved.action===ref.action,'reference disagrees with source');
  const wrong=saved.action==='PAINT_A'?'PAINT_B':'PAINT_A';
  check(obs.legal.includes(wrong),'opposite paint is illegal');
  steps.push({input:saved.input,action:wrong,weight:0,focus_loss_weight:0});
  const pre=h.stage,result=v52.command(h,wrong,par);
  check(h.stage===pre && result.material!==saved.input.surface && !h.success,'wrong paint made progress');
  const nonterminal=!h.done;
  let naiveStop=null;
  if(nonterminal){
   const naive=structuredClone(h);
   const beforeStop=naive.stage;
   v52.command(naive,'STOP',par);
   naiveStop={stage_before:beforeStop,stage_after:naive.stage,done:!!naive.done,success:!!naive.success};
   check(!naiveStop.success && naiveStop.stage_after===beforeStop,'naive STOP unexpectedly succeeded');
  }
  const continuation=[];
  if(nonterminal){
   for(let k=0;k<8 && !h.done;k++) {
    const notice=v52.notice(h,par);v53.injectB(h,arm,notice,t+1+k);
    const actual=v53.observe(h),input=streamify(actual,previous);
    check(input.legal.length && input.legal.includes(v51.teacher(actual)),'recovery action not legal');
    const next=v51.teacher(actual);previous=actual.durable;
    steps.push({input,action:next,weight:1,focus_loss_weight:next==='STOP'?1:6});
    const before=h.stage,receipt=v52.command(h,next,par);
    continuation.push({action:next,material:receipt.material??null,stage_before:before,stage_after:h.stage,
     accepted:h.accepted,success:!!h.success,done:!!h.done});
   }
   check(h.done && h.success && continuation.length>=2,'nonterminal recovery failed');
   check(continuation[0].action==='ERASE' && continuation.at(-1).action==='STOP','invalid recovery sequence');
  }
  return {steps,cert:{wrong_action:wrong,wrong_result:result.material,nonterminal,
   recovered:h.done&&h.success,terminal_damage:!nonterminal&&h.done&&!h.success,
   recovery_actions:continuation.map(x=>x.action),recovery_events:continuation,naive_stop_control:naiveStop,
   intervention_index:ref.source_step,original_action:saved.action,final_stage:h.stage}};
 }
 throw Error('unreachable');
}
function build(source,out){
 source=path.resolve(source);out=path.resolve(out);check(!fs.existsSync(out),'refuses output overwrite');
 const upstream=JSON.parse(read(path.join(source,'manifest.json')));
 check(upstream.schema==='driv3r/corpus_factory/physical-memory-causality-v56/1','wrong parent schema');
 for(const [f,v] of Object.entries(upstream.files)){
  const b=read(path.join(source,f));check(sha(b)===v.sha256&&b.length===v.bytes,'parent hash mismatch '+f);
 }
 fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/offpolicy-repair-v58/1',date:'2026-11-19',
  data_only:true,synthetic_not_native_CyberSand:true,generator_sha256:sha(read(__filename)),
  source_v56_manifest_sha256:sha(read(path.join(source,'manifest.json'))),
  source_v56_shards:Object.fromEntries(Object.entries(upstream.files).map(([f,v])=>[f,v.sha256])),
  host_module_sha256:Object.fromEntries(roots.map(f=>[f,sha(read(path.join(__dirname,f)))])),
  contract:{
   train:'TRAIN_offpolicy_repair_stream.jsonl ONLY. Full historical source prefixes plus forced wrong material paint (weight=0), followed by actual host-replayed ERASE/corrected PAINT/STOP (weight>0).',
   model_visible:'Only ordered steps[].input + past observed actions. All source identities, outcomes and forks are in EVALUATOR_ONLY sidecars; never pass them to a policy.',
   intervention:'Deliberately WRONG paint action, host-executed and loss-masked, not a teacher-supervision target. Keep model recurrent state across the action; do not add evaluator metadata.',
   dev:'DEV_EVALUATOR_ONLY_offpolicy_repair_stream.jsonl for development scoring; no training or hyperparameter optimization.',
   heldout:'HELDOUT_EVALUATOR_ONLY_offpolicy_repair_stream.jsonl sealed; no training, selection or prompt tuning.',
   scope:'Only recoverable surface-B wrong-paint episodes become trainable complete episodes. Terminal surface-A wrong-paints form evaluator-only failure-control certificates, NOT training examples or recoveries.',
   evaluation:'Score off-policy ERASE->correct PAINT->STOP recovery separately from original correct-paint pair accuracy and whole-episode autonomous terminal success. This dataset is teacher-forced and synthetic.',
   split:'Preserve parent v56 world geometry ownership; cross-version worlds must remain grouped by source split.',
   weights:'All original source weights retained; intervention zero; recovery ERASE and corrected PAINT receive focus_loss_weight=6; terminal STOP=1. No evaluator label exposed.'},
  files:{},splits:{},limitations:[
   'Recovery uses v51 symbolic oracle after a forced wrong action and therefore is a demonstration, not autonomous learned policy recovery.',
   'v53 synthetic host makes A material errors terminal while B errors are recoverable; this is not a native CyberSand rule.',
   'Source-selected v56 worlds and scripted preludes still admit simple parsers and do not constitute new geometry.',
   'No pinned target SLM weights updated, CPU neural latency/RSS, native CyberSand playback or immutable Actions corpus artifacts.'
  ]};
 const owners=new Map();
 for(const split of splits){
  const eps=lines(path.join(source,names(split,'episode'))),pairs=lines(path.join(source,names(split,'material_pairs'))),prov=lines(path.join(source,names(split,'provenance')));
  check(eps.length===prov.length,'provenance mismatch');
  const records=[],certs=[],seen=new Set();let terminal=0,wrong=0,recovery=0,decisions=0;
  for(const pair of pairs){
   check(pair.category==='physically_certified_material_memory'&&pair.references.length===2,'invalid pair');
   for(let q=0;q<2;q++){
    const ref=pair.references[q],p=prov[ref.v56_causal_episode],ep=eps[ref.v56_causal_episode];
    const key=split+':'+ref.v56_causal_episode+':'+ref.source_step;
    check(!seen.has(key),'duplicate fork');seen.add(key);
    check(p.split===split&&ep.steps[ref.source_step].action===pair.actions[q],'bad source ref');
    const owner=owners.get(p.geometry_sha256);check(!owner||owner===split,'cross split world leak');owners.set(p.geometry_sha256,split);
    const w=v46.candidate(split,700+p.world_index).w;
    check(w.geometry_sha256===p.geometry_sha256&&w.seed===p.source_seed,'bad geometry seed');
    const result=generateBranch(p,ep,{source_step:ref.source_step,action:pair.actions[q]},w);
    const {cert,steps}=result;wrong++;
    check(cert.original_action!==cert.wrong_action,'wrong branch is correct');
    const meta={pair_id:pair.pair_id,reference:q,source_episode_index:ref.v56_causal_episode,source_step:ref.source_step,
     world_index:p.world_index,geometry_sha256:p.geometry_sha256,source_seed:p.source_seed,arm:p.arm,
     surface:ep.steps[ref.source_step].input.surface,instant_observation_sha256:pair.instant_observation_sha256,
     prior_identical_suffix_steps:pair.identical_past_input_action_suffix_steps,calibration_memory_age_steps:ref.calibration_memory_age_steps,...cert};
    if(cert.nonterminal){
     check(meta.surface==='B'&&cert.recovered,'unexpected recovered surface');
     records.push({steps});certs.push({...meta,trainable_episode_index:records.length-1});
     decisions+=steps.length;recovery++;
    }else{
     check(meta.surface==='A'&&cert.terminal_damage,'unexpected terminal outcome');terminal++;
     certs.push({...meta,trainable_episode_index:null});
    }
   }
  }
  const data=split+(split==='TRAIN'?'_offpolicy_repair_stream.jsonl':'_EVALUATOR_ONLY_offpolicy_repair_stream.jsonl');
  const meta=split+'_offpolicy_fork_certificates_EVALUATOR_ONLY.jsonl';
  for(const [file,rows] of [[data,records],[meta,certs]]){
   const b=Buffer.from(rows.map(enc).join('\n')+'\n');fs.writeFileSync(path.join(out,file),b,{flag:'wx'});
   manifest.files[file]={sha256:sha(b),bytes:b.length,rows:rows.length};
  }
  manifest.splits[split]={source_material_pairs:pairs.length,wrong_accepted_forks:wrong,irreversible_wrong_A:terminal,
   recoverable_wrong_B:recovery,recovered_complete_episodes:records.length,demonstration_decisions:decisions,
   zero_weight_forced_wrong_paints:records.length,naive_stop_successes:0, host_recovery_erase:recovery,host_recovery_corrected_paint:recovery,
   host_recovery_stop:recovery};
 }
 const b=Buffer.from(JSON.stringify(manifest,null,2)+'\n');fs.writeFileSync(path.join(out,'manifest.json'),b,{flag:'wx'});
 return manifest;
}
if(require.main===module){const [source,out]=process.argv.slice(2);check(source&&out,'usage: node offpolicy_repair_v58.js V56_CORPUS NEW_OUTPUT_DIR');console.log(JSON.stringify(build(source,out).splits,null,2));}
module.exports={build,generateBranch,streamify};
