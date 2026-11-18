#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v57: independent physical wrong-command fork certificates.
 * Data/evaluator-only. No model training. v53 SYNTHETIC graph host != native CyberSand.
 */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const v53 = require('./deferred_surface_v53.js');
const v52 = require('./realert_v52.js');
const v51 = require('./active_probe_v51.js');
const v46 = require('./mission_memory_v46.js');
const splits = ['TRAIN','DEV','HELDOUT'];
const hash = x => crypto.createHash('sha256').update(x).digest('hex');
const assert = (x,m) => {if(!x)throw Error('V57 '+m);};
const read = f => fs.readFileSync(f);
const lines = f => read(f).toString('utf8').trim().split('\n').map(JSON.parse);
const stable = x => JSON.stringify(x);
function fname(split,type) {
 if(type==='episodes') return split + (split==='TRAIN'?'_causal_stream.jsonl':'_EVALUATOR_ONLY_causal_stream.jsonl');
 return split+'_'+type+'_EVALUATOR_ONLY.jsonl';
}
const roots=['deferred_surface_v53.js','realert_v52.js','active_probe_v51.js','online_recovery_v50.js','contextual_probe_v49.js', 'empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'];
function replayReference(split,p,episode,ref,world) {
 const arm=v53.ARMS.find(a=>a.id===p.arm);assert(!!arm,'arm '+p.arm);
 const h=v53.prep(world,arm),par=v53.armParams(arm);
 const idx=ref.source_step, steps=episode.steps;
 assert(idx>=0&&idx<steps.length-1,'step range');
 for(let t=0;t<=idx;t++) {
  const notice=v52.notice(h,par);v53.injectB(h,arm,notice,t);
  const obs=v53.observe(h), action=v51.teacher(obs);
  assert(action===steps[t].action,'source action mismatch '+split+'/'+p.arm+'/'+t);
  assert(stable(obs.p)===stable(steps[t].input.p),'source position drift');
  assert(obs.legal.includes(action),'host action illegal');
  if(t<idx) { v52.command(h,action,par);continue; }
  assert(action==='PAINT_A'||action==='PAINT_B','not a material decision');
  assert(steps[t].input.surface===ref.surface,'material surface drift');
  const target=ref.surface, goal=h.stage===0?h.w.goal:h.w.goal2;
  assert(h.pos===goal,'not at protected mission goal');
  const before={stage:h.stage,accepted:h.accepted,success:!!h.success,done:!!h.done};
  const chosen=v52.command(h,action,par);
  const after={stage:h.stage,accepted:h.accepted,success:!!h.success,done:!!h.done,protected_damage:!!h.protected_damage};
  assert(chosen.material===target && after.stage!==before.stage,'correct action did not advance protected mission');
  return {action,correct_result:chosen.material,pre:before,correct_post:after,
   active_warning:obs.durable.some(e=>e[1]==='SURFACE_ALERT'&&e[2]===target),
   site:obs.p,accepted_action_number:before.accepted,stage:before.stage};
 }
 throw Error('unreachable');
}
function forkWrong(split,p,episode,ref,world) {
 const arm=v53.ARMS.find(a=>a.id===p.arm),par=v53.armParams(arm),h=v53.prep(world,arm);
 for(let t=0;t<=ref.source_step;t++) {
  const notice=v52.notice(h,par);v53.injectB(h,arm,notice,t);
  const obs=v53.observe(h),a=v51.teacher(obs);
  assert(a===episode.steps[t].action,'wrong-arm reconstruction diverged');
  if(t<ref.source_step){v52.command(h,a,par);continue;}
  const wrong=a==='PAINT_A'?'PAINT_B':'PAINT_A';
  assert(obs.legal.includes(wrong),'opposite command not legal');
  const before={stage:h.stage,done:!!h.done,success:!!h.success,accepted:h.accepted};
  const r=v52.command(h,wrong,par);
  const after={stage:h.stage,done:!!h.done,success:!!h.success,accepted:h.accepted,
    protected_damage:!!h.protected_damage};
  assert(r.material!==ref.surface,'wrong action unexpectedly produced correct target material');
  assert(after.stage===before.stage && !after.success,'wrong action advanced mission');
  assert(after.accepted===before.accepted+1,'wrong action not accepted');
  const continuation=[];
  if(!h.done) {
   for(let k=0;k<8&&!h.done;k++) {
    const n=v52.notice(h,par);v53.injectB(h,arm,n,ref.source_step+1+k);
    const i=v53.observe(h),repair=v51.teacher(i);
    assert(i.legal.includes(repair),'recovery oracle chose illegal action');
    const at={at:i.p,action:repair,stage_before:h.stage,material_before:h.painted[h.pos]??null};
    const result=v52.command(h,repair,par);
    continuation.push({...at,material_result:result.material??null,stage_after:h.stage,
      terminal:!!h.done,success:!!h.success});
   }
   assert(h.done&&h.success,'nonterminal wrong paint not recovered within eight host actions');
   assert(continuation.length>=2 && continuation[0].action==='ERASE','recovery did not erase bad material');
  }
  return {wrong_action:wrong,wrong_result:r.material,wrong_post:after,
   wrong_caused_terminal_failure:after.done&&!after.success,
   wrong_caused_nonterminal_bad_paint:!after.done&&!after.success,
   oracle_recovery_actions:continuation.map(x=>x.action),
   oracle_recovery_events:continuation,
   oracle_recovery_terminal_success:continuation.length>0&&h.done&&h.success};
 }
 throw Error('unreachable');
}
function build(source,out){
 source=path.resolve(source);out=path.resolve(out);assert(!fs.existsSync(out),'refuses output overwrite');
 const parent=JSON.parse(read(path.join(source,'manifest.json')));
 assert(parent.schema==='driv3r/corpus_factory/physical-memory-causality-v56/1','requires exact v56 schema');
 for(const [f,desc] of Object.entries(parent.files)) {
  const b=read(path.join(source,f));assert(hash(b)===desc.sha256 && b.length===desc.bytes,'source hash '+f);
 }
 fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/physical-fork-v57/1',date:'2026-11-18',
  purpose:'Host-replayed positive and opposite-command branches for every v56 causal material-memory comparison; evaluator-only evidence, not new policies.',
  synthetic_not_native_CyberSand:true,data_only:true,
  generator_sha256:hash(read(__filename)),
  parent_v56_manifest_sha256:hash(read(path.join(source,'manifest.json'))),
  parent_v56_shard_sha256:Object.fromEntries(Object.entries(parent.files).map(([f,d])=>[f,d.sha256])),
  host_module_sha256:Object.fromEntries(roots.map(f=>[f,hash(read(path.join(__dirname,f)))])),
  contract:{training:'Continue consuming original v56 TRAIN causal streams; v57 provides no extra training or policy labels.',
   model_visible:'NONE of the v57 evaluator-only fork certificates. v56 input observations remain the only policy observations.',
   evaluation:'Check correct chosen action and legal opposite action at identical current input; report both physical material outcomes, mission stage progress, terminal damage and nonterminal incorrect paint separately.',
   splits:'Never mix DEV/HELDOUT evaluator-only records into training; retain v53-v56 source split and geometry lineage.',
   interpretation:'A one-step synthetic-host branch intervention, NOT closed-loop learned policy success. Correct-action progress and opposite-action regression are replayed in the original host code.'},
  files:{},splits:{},limitations:[
   'v57 does not generate fresh geometry or train an SLM, and inherits v56 short-history concentration and symbolic-parse shortcuts.',
   'One wrong action per fork; a deterministic oracle recovery tests physical recoverability but does not show whether a trained model can recover.',
   'Synthetic host protected-paint behavior is not native CyberSand and must not be claimed as a gameplay result.',
   'No pinned LFM/Sprocket neural training, neural CPU RSS/latency, or immutable Actions corpus artifact publication.']};
 const geo=new Map(),worlds=new Map();
 for(const split of splits) {
  const episodes=lines(path.join(source,fname(split,'episodes')));
  const provenance=lines(path.join(source,fname(split,'provenance')));
  const pairs=lines(path.join(source,fname(split,'material_pairs')));
  assert(episodes.length===provenance.length,'provenance length mismatch');
  const entries=[];let count=0,term=0,nonterm=0,success=0,recovered=0;
  for(const pair of pairs) {
   assert(pair.category==='physically_certified_material_memory','wrong pair category');
   assert(pair.references.length===2&&pair.actions.length===2,'pair reference size');
   const c=[];
   for(let q=0;q<2;q++) {
    const ref=pair.references[q],p=provenance[ref.v56_causal_episode],ep=episodes[ref.v56_causal_episode];
    assert(p.split===split,'source split mismatch');
    assert(ep.steps[ref.source_step].action===pair.actions[q],'source label mismatch');
    const wkey=split+':'+p.world_index;
    if(!worlds.has(wkey)) worlds.set(wkey,v46.candidate(split,700+p.world_index).w);
    const w=worlds.get(wkey);
    assert(w.geometry_sha256===p.geometry_sha256 && w.seed===p.source_seed,'world provenance mismatch');
    const owner=geo.get(p.geometry_sha256);assert(!owner||owner===split,'cross-split geometry overlap');geo.set(p.geometry_sha256,split);
    const a=replayReference(split,p,ep,ref,w),b=forkWrong(split,p,ep,ref,w);
    assert(a.action===pair.actions[q],'correct action mismatch');
    assert(a.correct_result===ref.surface&&b.wrong_result!==ref.surface,'physical fork not discriminative');
    count++;success+=Number(a.correct_post.stage!==a.pre.stage);
    term+=Number(b.wrong_caused_terminal_failure);nonterm+=Number(b.wrong_caused_nonterminal_bad_paint);recovered+=Number(b.oracle_recovery_terminal_success);
    c.push({arm:p.arm,world_index:p.world_index,episode_index:ref.v56_causal_episode,source_step:ref.source_step,
      input_sha256:pair.instant_observation_sha256,calibration_memory_age_steps:ref.calibration_memory_age_steps,
      ...a,...b});
   }
   assert(c[0].action!==c[1].action && c[0].correct_result===c[1].correct_result,'pair physical contrast missing');
   entries.push({pair_id:pair.pair_id,split,same_instant_input_sha256:pair.instant_observation_sha256,
    identical_past_input_action_suffix_steps:pair.identical_past_input_action_suffix_steps,
    counterfactuals:c});
  }
  const name=split+'_host_forks_EVALUATOR_ONLY.jsonl',b=Buffer.from(entries.map(stable).join('\n')+'\n');
  fs.writeFileSync(path.join(out,name),b,{flag:'wx'});
  manifest.files[name]={sha256:hash(b),bytes:b.length,rows:entries.length};
  manifest.splits[split]={paired_contrasts:entries.length,correct_action_forks:count,host_confirmed_correct_mission_advances:success,
   wrong_action_forks:count,wrong_terminal_failures:term,wrong_nonterminal_bad_paints:nonterm,wrong_nonterminal_oracle_recoveries:recovered,
   host_confirmed_wrong_mission_advances:0,
   equal_past_32step_pairs:entries.filter(x=>x.identical_past_input_action_suffix_steps>=32).length};
 }
 const b=Buffer.from(JSON.stringify(manifest,null,2)+'\n');fs.writeFileSync(path.join(out,'manifest.json'),b,{flag:'wx'});
 return manifest;
}
if(require.main===module){const [source,out]=process.argv.slice(2);assert(source&&out,'usage: node physical_fork_v57.js V56_CORPUS NEW_OUTPUT_DIR');console.log(JSON.stringify(build(source,out).splits,null,2));}
module.exports={build,replayReference,forkWrong};
