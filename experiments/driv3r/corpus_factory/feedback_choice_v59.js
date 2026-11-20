#!/usr/bin/env node
'use strict';
/**
 * Corpus Factory v59 — balanced receipt-conditioned B-goal continuation.
 * DATA ONLY; synthetic v53 graph host, NOT native CyberSand.
 * Deliberately executes one correct or incorrect B paint on the very same
 * history, masks both interventions, and learns the host-observable response:
 * STOP when goal paint was correct; ERASE, corrected PAINT, STOP otherwise.
 * Entire event-stream episodes retained, evaluator metadata segregated.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v58=require('./offpolicy_repair_v58.js');
const v53=require('./deferred_surface_v53.js');
const v52=require('./realert_v52.js');
const v51=require('./active_probe_v51.js');
const v46=require('./mission_memory_v46.js');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const check=(b,s)=>{if(!b)throw Error('v59 '+s)};
const enc=JSON.stringify;
const read=f=>fs.readFileSync(f);
const jsonlines=f=>read(f).toString('utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const splits=['TRAIN','DEV','HELDOUT'];
const sourceName=(s,type)=> type==='episodes'?s+(s==='TRAIN'?'_causal_stream.jsonl':'_EVALUATOR_ONLY_causal_stream.jsonl'):s+'_'+type+'_EVALUATOR_ONLY.jsonl';
const deps=['offpolicy_repair_v58.js','deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js','empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
function correctFork(provenance, episode, ref, world){
 const arm=v53.ARMS.find(z=>z.id===provenance.arm);check(arm,'missing arm');
 const pars=v53.armParams(arm),host=v53.prep(world,arm),steps=[];let previous=[];
 for(let t=0;t<=ref.source_step;t++){
  const alert=v52.notice(host,pars);v53.injectB(host,arm,alert,t);
  const full=v53.observe(host),input=v58.streamify(full,previous),src=episode.steps[t];
  check(enc(input)===enc(src.input),'correct fork source replay diverged at '+t);
  check(full.legal.includes(src.action),'source action illegal');previous=full.durable;
  if(t<ref.source_step){
   steps.push({input:src.input,action:src.action,weight:src.weight,
    ...(src.focus_loss_weight===undefined?{}:{focus_loss_weight:src.focus_loss_weight})});
   v52.command(host,src.action,pars);continue;
  }
  check(host.stage===1 && host.pos===host.w.goal2 && src.input.surface==='B','not B goal');
  check(src.action===ref.action && /^PAINT_[AB]$/.test(src.action),'correct action not verified');
  // Both arms are forced; do NOT supervise the fork action as a model prediction.
  steps.push({input:src.input,action:src.action,weight:0,focus_loss_weight:0});
  const result=v52.command(host,src.action,pars);
  check(result.material==='B'&&host.stage===2&&!host.done,'correct physical B paint did not complete stage');
  const nextAlert=v52.notice(host,pars);v53.injectB(host,arm,nextAlert,t+1);
  const fullNext=v53.observe(host),next=v58.streamify(fullNext,previous);
  check(fullNext.legal.includes('STOP')&&v51.teacher(fullNext)==='STOP','no legal success STOP');
  steps.push({input:next,action:'STOP',weight:1,focus_loss_weight:6});
  v52.command(host,'STOP',pars);
  check(host.done&&host.success&&!host.protected_damage,'correct fork not terminal success');
  return {steps,physical:{correct_action:src.action,correct_material:result.material,
   before_stage:1,after_paint_stage:2,final_stage:host.stage,success:host.success,
   post_paint_receipt_seen:next.events_delta.some(x=>x[1]==='PAINT_RESULT'&&x[2]===src.action&&x[3]==='B'&&x[4]==='B'),
   post_paint_mission_complete_seen:next.events_delta.some(x=>x[1]==='MISSION_COMPLETE')}};
 }
 throw Error('unreachable');
}
function writeRows(out,filename,rows,files){const b=Buffer.from(rows.map(enc).join('\n')+'\n');fs.writeFileSync(path.join(out,filename),b,{flag:'wx'});files[filename]={sha256:sha(b),bytes:b.length,rows:rows.length};}
function generate(source,out){
 source=path.resolve(source);out=path.resolve(out);
 check(!fs.existsSync(out),'refuses output overwrite');
 const parent=JSON.parse(read(path.join(source,'manifest.json')));
 check(parent.schema==='driv3r/corpus_factory/physical-memory-causality-v56/1','wrong source schema');
 for(const [f,m] of Object.entries(parent.files)){const b=read(path.join(source,f));check(b.length===m.bytes&&sha(b)===m.sha256,'source shard mismatch '+f);}
 fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/receipt-conditioned-continuation-v59/1',date:'2026-11-19',
  data_only:true,synthetic_not_native_CyberSand:true,
  source_v56_manifest_sha256:sha(read(path.join(source,'manifest.json'))),
  parent_shard_sha256:Object.fromEntries(Object.entries(parent.files).map(([f,m])=>[f,m.sha256])),
  generator_sha256:sha(read(__filename)),host_module_sha256:Object.fromEntries(deps.map(f=>[f,sha(read(path.join(__dirname,f)))])),
  contract:{
   training:'TRAIN_receipt_choice_stream.jsonl ONLY. Complete episodes: identical historical prefixes, then equally many forced-correct and forced-wrong B paints (both loss-masked), then observed-result-specific STOP versus ERASE/PAINT/STOP.',
   visibility:'Only steps[].input, prior model-observed actions and labelled training decisions; NEVER supply sidecar certificates, source references, branch_arm labels or underlying hidden material laws.',
   action_abi:'No new action tokens. Host-legal MOVE, PAINT_A/PAINT_B, ERASE, STOP. Post-action receipt observed in events_delta only after accepted paint.',
   split:'Use original v56 geometry ownership, group by geometry/world across v54-v59; HELDOUT never used for training or model selection.',
   weights:'Forced correct AND wrong intervention actions have weight=0 and focus_loss_weight=0. All post-intervention actions weight=1, focus_loss_weight=6; redundant identical TRAIN episode-prefix actions zero-weighted but preserved for recurrent state.',
   scoring:'Conditioned binary choice: after forced correct paint STOP, after forced wrong B paint ERASE then corrected PAINT and STOP. Score separately from holistic autonomous mission success; never count teacher replay as neural score.',
   evaluation:'Use paired source world and action memory; compare correct/wrong postaction receipt, monitor both exact sequence and host success. Positive interventions prohibit unconditional ERASE policies; negative prohibit unconditional STOP policies.'},
  limitations:['Synthetic v53 host, not native CyberSand; surfaces, A/B protection and law flips are hypothetical.',
   'The material state is visible in painted/current receipts; a simple symbolic host-aware rule may still solve continuation.',
   'Branch choice is an external intervention; actions at fork are not neural decisions and are masked from supervised loss.',
   'Inherited v56 worlds and scripted physical calibration; not new topology, not real SLM training, no neural CPU latency/RSS.',
   'No immutable Actions dataset artifact; output generated locally from exact pinned v56 source.'],files:{},splits:{}};
 const geometryOwners=new Map();let all=0;
 for(const split of splits){
  const eps=jsonlines(path.join(source,sourceName(split,'episodes'))),
        prov=jsonlines(path.join(source,sourceName(split,'provenance'))),
        pairs=jsonlines(path.join(source,sourceName(split,'material_pairs')));
  check(eps.length===prov.length,'source episodes/provenance length mismatch');
  const stream=[],certs=[],seen=new Set(),prefixSeen=new Set();
  let wrong=0,right=0,maskedPrefix=0,decisions=0,critical=0;
  for(const pair of pairs){
   check(pair.references.length===2 && pair.actions.length===2,'invalid source pair');
   for(let refIndex=0;refIndex<2;refIndex++){
    const ref=pair.references[refIndex],index=ref.v56_causal_episode,p=prov[index],ep=eps[index];
    if(ref.surface!=='B')continue;
    const key=index+':'+ref.source_step;check(!seen.has(key),'duplicate source intervention');seen.add(key);
    check(p.split===split && ref.surface==='B' && pair.actions[refIndex]===ep.steps[ref.source_step].action,'invalid B reference');
    const old=geometryOwners.get(p.geometry_sha256);check(!old||old===split,'geometry split leakage');geometryOwners.set(p.geometry_sha256,split);
    const w=v46.candidate(split,700+p.world_index).w;
    check(w.seed===p.source_seed&&w.geometry_sha256===p.geometry_sha256,'world lineage invalid');
    const bad=v58.generateBranch(p,ep,{source_step:ref.source_step,action:ref.action||pair.actions[refIndex]},w);
    const good=correctFork(p,ep,{source_step:ref.source_step,action:pair.actions[refIndex]},w);
    check(bad.cert.nonterminal&&bad.cert.recovered&&!bad.cert.terminal_damage,'wrong B fork failed');
    check(enc(bad.steps.slice(0,ref.source_step).map(s=>[s.input,s.action]))===enc(good.steps.slice(0,ref.source_step).map(s=>[s.input,s.action])),'prefix mismatch');
    check(enc(bad.steps[ref.source_step].input)===enc(good.steps[ref.source_step].input),'fork observation mismatch');
    check(bad.steps[ref.source_step].action!==good.steps[ref.source_step].action,'same intervention');
    check(good.physical.post_paint_receipt_seen && good.physical.post_paint_mission_complete_seen,'missing correct-host receipts');
    check(bad.steps[ref.source_step+1].input.events_delta.some(x=>x[1]==='PAINT_RESULT'&&x[3]==='A'&&x[4]==='B'),'wrong-host receipt invisible');
    for(const [branchType,result] of [['correct',good],['wrong',bad]]){
     const steps=structuredClone(result.steps),armIndex=branchType==='correct'?0:1;
     let running='episode-start',maskedHere=0;
     for(let j=0;j<steps.length;j++){
      const s=steps[j];check(s.input.v==='synthetic-v54-event-stream','not event-stream');
      check(!('durable' in s.input),'full historical receipts leaked');
      check(s.input.legal.includes(s.action),'illegal host action');
      if(j===ref.source_step)check(s.weight===0 && s.focus_loss_weight===0,'forced paint not masked');
      const canonical=sha(running+'\0'+enc(s.input)+'\0'+s.action);
      running=canonical;
      if(split==='TRAIN'&&j<ref.source_step&&prefixSeen.has(canonical)){
       if(s.weight!==0){s.weight=0;s.focus_loss_weight=0;maskedHere++;}
      }else if(split==='TRAIN'&&j<ref.source_step)prefixSeen.add(canonical);
     }
     maskedPrefix+=maskedHere;
     for(const s of steps.slice(ref.source_step+1))s.focus_loss_weight=6;
     const post=steps.slice(ref.source_step+1).map(s=>s.action);
     check(enc(post)===enc(branchType==='correct'?['STOP']:['ERASE',pair.actions[refIndex],'STOP']),'post paint branch condition violated');
     for(const s of steps.slice(ref.source_step+1)){check(s.weight===1&&s.focus_loss_weight===6,'postintervention supervision missing');critical++;}
     stream.push({steps});decisions+=steps.length;
     certs.push({pair_id:pair.pair_id,source_episode_index:index,source_step:ref.source_step,
      geometry_sha256:p.geometry_sha256,source_seed:p.source_seed,world_index:p.world_index,
      arm:p.arm,source_action:pair.actions[refIndex],branch_type:branchType,branch_index:armIndex,
      intervention_action:steps[ref.source_step].action,source_ref:refIndex,
      trainable_episode_index:stream.length-1,material_surface:'B',
      postintervention_actions:post,synthetic_terminal_success:true,
      postintervention_physical_result:branchType==='correct'?'B':'A',
      postintervention_host_receipt:true,
      previous_identical_suffix_steps:pair.identical_past_input_action_suffix_steps,
      prior_last_calibration_age_steps:ref.calibration_memory_age_steps,
      prior_prefix_repeat_masked:maskedHere,
      original_v58_fork_recovered:branchType==='wrong'?bad.cert.recovered:null});
     branchType==='correct'?right++:wrong++;
    }
   }
  }
  check(right===wrong && right===seen.size,'imbalanced binary branch');
  const data=split+(split==='TRAIN'?'_receipt_choice_stream.jsonl':'_EVALUATOR_ONLY_receipt_choice_stream.jsonl');
  const evidence=split+'_receipt_choice_certificates_EVALUATOR_ONLY.jsonl';
  writeRows(out,data,stream,manifest.files);writeRows(out,evidence,certs,manifest.files);
  manifest.splits[split]={source_physical_pairs:pairs.length,source_B_goal_decisions:seen.size,
   correct_intervention_episodes:right,wrong_intervention_episodes:wrong,
   host_terminal_successes:stream.length,episodes:stream.length,decisions,
   forced_paints_zero_weight:stream.length,correct_postaction_STOP:right,
   wrong_postaction_ERASE:wrong,wrong_postaction_corrected_PAINT:wrong,wrong_postaction_STOP:wrong,
   supervised_postintervention_decisions:critical,
   repeated_train_prefix_decisions_loss_masked:maskedPrefix};
  all+=decisions;
 }
 manifest.unique_geometry_worlds=geometryOwners.size;manifest.total_decisions=all;
 const m=Buffer.from(JSON.stringify(manifest,null,2)+'\n');fs.writeFileSync(path.join(out,'manifest.json'),m,{flag:'wx'});
 return manifest;
}
if(require.main===module){const [source,out]=process.argv.slice(2);check(source&&out,'usage node feedback_choice_v59.js V56_CORPUS NEW_OUTPUT_DIR');console.log(JSON.stringify(generate(source,out).splits,null,2));}
module.exports={generate,correctFork};
