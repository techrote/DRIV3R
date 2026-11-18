'use strict';
// DRIV3R Corpus Factory v33: real host-accepted counterfactual bad-first navigation
// and exact optimal recovery, without model training. Synthetic host, NOT CyberSand.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js');
const host=require('./causal_gate_v26.js');
const view=require('./bounded_view_v30.js');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:8,DEV:4,HELDOUT:4};
const ALT=['BACKTRACK','MOVE_E','MOVE_S','MOVE_W'];
function cleanRanks(planner,h){
  const ranks=planner.rank([h.pos,h.prev,h.status,h.latched]);
  const legal=host.options(h.w,[h.pos,h.prev,h.status,h.latched]).map(x=>x.action).sort();
  if(JSON.stringify(ranks.map(x=>x.action).sort())!==JSON.stringify(legal))throw Error('oracle action menu mismatch');
  return ranks;
}
function optionsCertificate(ranks){
  const best=ranks[0].cost;
  if(!Number.isFinite(best))throw Error('terminal suffix unreachable');
  return ranks.map(r=>({action:r.action,remaining_cost:Number.isFinite(r.cost)?r.cost:null,
    regret:Number.isFinite(r.cost)?r.cost-best:null}));
}
function chooseWrong(ranks,g,v,arm){
  const legal=ranks.filter(r=>ALT.includes(r.action)&&Number.isFinite(r.cost)&&r.cost-ranks[0].cost>=2)
    .sort((a,b)=>a.action.localeCompare(b.action));
  if(legal.length<2)throw Error('not enough finite detrimental legal alternatives');
  return legal[(g+v+arm)%legal.length];
}
function replay(w,planner,split,g,v,arm,pair,mode,emit){
  const prepared=base.prepare(w,split,g,v,arm),h=prepared.h;
  const first=cleanRanks(planner,h),best=first[0];
  if(best.action!==pair.first[arm].action)throw Error('paired oracle mismatch');
  const alternative=chooseWrong(first,g,v,arm);
  let accepted_intervention=null;
  if(mode==='wrong-first'){
    const receipt=host.apply(h,alternative.action);
    if(!receipt.accepted)throw Error('off-policy first action was not host legal');
    accepted_intervention={action:alternative.action,receipt,optimal_cost:best.cost,
      alternate_cost:alternative.cost,strict_regret:alternative.cost-best.cost};
  } else if(mode!=='clean')throw Error('unknown mode');
  const eid=`${split}-v33-g${g}-v${v}-arm${arm}-${mode}`;
  h.id=eid;
  const initial=host.observe(h),initial_history=clone(h.history);
  const initial_input_sha256=hash(initial);
  const certificate=optionsCertificate(cleanRanks(planner,h));
  let prevEvents=h.history.events.length,prevPositions=h.history.positions.length,step=0,erases=0;
  let previousCost=cleanRanks(planner,h)[0].cost;
  while(!h.done){
    const obs=host.observe(h),rank=cleanRanks(planner,h),target=rank[0];
    if(step>200||!Number.isFinite(target.cost))throw Error('bounded recovery exceeded');
    if(target.cost!==previousCost)throw Error('suffix-distance witness mismatch');
    const bounded=view.compress(obs);
    if(JSON.stringify(bounded).length>8192)throw Error('bounded model input >8192 characters');
    const row={schema:'driv3r/corpus_factory/offpolicy-recovery-v33/decision',episode_id:eid,step,
      state:Object.fromEntries(Object.entries(obs).filter(([k])=>k!=='history')),
      events_added:clone(obs.history.events.slice(prevEvents)),positions_added:obs.history.positions.slice(prevPositions),
      model_input_sha256:hash(obs),bounded_model_input:bounded,bounded_model_input_sha256:hash(bounded),
      target:target.action,oracle_action_costs:optionsCertificate(rank),training_weight:1};
    prevEvents=obs.history.events.length;prevPositions=obs.history.positions.length;
    const receipt=host.apply(h,target.action);
    if(!receipt.accepted)throw Error('oracle action rejected by host');
    row.receipt=receipt;emit(row,obs);if(target.action==='ERASE')erases++;
    previousCost--;step++;
  }
  if(previousCost!==0||erases!==2||host.correct(w,h.status).wrong!==0||host.correct(w,h.status).count<2)
    throw Error('terminal failed');
  const expected=mode==='clean'?best.cost:alternative.cost-1;
  if(step!==expected)throw Error(`not shortest successful suffix ${step} vs ${expected}`);
  const firstAfter=mode==='clean'?best.action:certificate.find(x=>x.regret===0)?.action;
  return {schema:'driv3r/corpus_factory/offpolicy-recovery-v33/episode',episode_id:eid,split,mode,
    family_id:`${split}-g${g}-v${v}`,world_group:`${split}-${g}`,
    synthetic_host_not_native:true,world_sha256:w.world_sha256,geometry_sha256:w.geometry_sha256,
    hidden_interventions:{older_wrong_target_index:arm,latest_wrong_target_index:4},
    older_history_accepted_actions:prepared.accepted,off_policy_accepted_intervention:accepted_intervention,
    initial_history,initial_full_observation_sha256:initial_input_sha256,
    no_memory_alias_sha256:pair.restricted_hash,last_paint_alias_sha256:pair.last_paint_hash,
    first_legal_action_regrets:optionsCertificate(first),first_after_intervention_optimal:firstAfter,
    optimal_clean_suffix_cost:best.cost,accepted_bad_first_cost:alternative.cost,
    realised_first_action_regret:mode==='wrong-first'?alternative.cost-best.cost:0,
    first_action_required:best.action,steps:step,erasures:erases,terminal_success:true,
    difficulty:{nodes:w.nodes.length,paint_targets:5,durable_history_required:true,
      accepted_bad_first_action:mode==='wrong-first',remote_history_lag_at_least:160,
      strict_gate_causal_control:pair.causal[arm].strict_regret>=2}};
}
function shard(dest,name,arr){const content=Buffer.from(arr.map(JSON.stringify).join('\n')+'\n');
  fs.writeFileSync(path.join(dest,name),content);return {rows:arr.length,bytes:content.length,sha256:hash(content)};}
function generate(dest,{quick=false}={}){
  if(fs.existsSync(dest))throw Error('refuse overwrite '+dest);
  fs.mkdirSync(dest,{recursive:true});
  const manifestSplits={},shards={},inputOwners=new Map(),geometryOwners=new Map();
  for(const [split,totalGroups] of Object.entries(SPLITS)){
    const eps=[],records=[],seen=new Map(),rejections=[],pairDescriptors=[],groupCount=quick?1:totalGroups;
    let badActions=0,badRegret=0,cleanSteps=0,recoverySteps=0;
    for(let g=0;g<groupCount;g++)for(let v=0;v<8;v++){
      const w=base.world(split,g,v),planner=base.oracle(w),pair=base.qualify(w,planner,split,g,v);
      if(!pair){rejections.push({g,v,reason:'v32 strict four-way or causal qualifier failed'});continue;}
      const geoOwner=geometryOwners.get(w.geometry_sha256);
      if(geoOwner&&geoOwner!==split)throw Error('cross-split geometry overlap');
      geometryOwners.set(w.geometry_sha256,split);
      pairDescriptors.push({family_id:`${split}-g${g}-v${v}`,world_sha256:w.world_sha256,
        first_actions:pair.first.map(x=>x.action),strict_frozen_gate_regret:pair.causal.map(x=>x.strict_regret)});
      for(let arm=0;arm<4;arm++)for(const mode of ['clean','wrong-first']){
        const ep=replay(w,planner,split,g,v,arm,pair,mode,(row)=>{
          const earlier=seen.get(row.model_input_sha256),owner=inputOwners.get(row.model_input_sha256);
          if(earlier!==undefined&&earlier!==row.target)throw Error('conflicting identical-input labels');
          if(owner&&owner!==split)throw Error('cross-split input overlap');
          if(earlier!==undefined&&split==='TRAIN')row.training_weight=0;
          else seen.set(row.model_input_sha256,row.target);
          inputOwners.set(row.model_input_sha256,split);records.push(row);
        });
        if(mode==='wrong-first'){badActions++;badRegret+=ep.realised_first_action_regret;recoverySteps+=ep.steps;}
        else cleanSteps+=ep.steps;
        eps.push(ep);
      }
    }
    if(!pairDescriptors.length)throw Error('no valid families');
    for(const suffix of ['episodes','decisions']){
      const name=`${split}_${suffix}.jsonl`;shards[name]=shard(dest,name,suffix==='episodes'?eps:records);
    }
    manifestSplits[split]={groups:groupCount,worlds_considered:groupCount*8,qualified_four_way_families:pairDescriptors.length,
      rejected:rejections,episodes:eps.length,clean_episodes:eps.filter(x=>x.mode==='clean').length,
      offpolicy_recovery_episodes:eps.filter(x=>x.mode==='wrong-first').length,
      decisions:records.length,corrective_erasures:eps.reduce((n,e)=>n+e.erasures,0),
      accepted_detrimental_first_actions:badActions,total_proven_first_action_regret:badRegret,
      clean_suffix_steps:cleanSteps,recovery_suffix_steps:recoverySteps,
      train_zero_weight:records.filter(x=>x.training_weight===0).length,
      unique_full_inputs:seen.size,first_actions_clean:eps.filter(x=>x.mode==='clean').reduce((d,e)=>(d[e.first_action_required]=(d[e.first_action_required]||0)+1,d),{}),
      strict_gate_causal_clean:eps.filter(x=>x.mode==='clean'&&x.difficulty.strict_gate_causal_control).length};
  }
  const manifest={schema:'driv3r/corpus_factory/offpolicy-recovery-v33/manifest',synthetic_host_not_native_CyberSand:true,
    generator:'offpolicy_recovery_v33.js',generator_sha256:hash(fs.readFileSync(__filename)),
    dependencies:['causal_shortcut_v32.js','causal_gate_v26.js','outcome_forks_v25.js','bounded_view_v30.js'].map(file=>({file,sha256:hash(fs.readFileSync(path.join(__dirname,file)))})),
    quick,oracle:'exact shortest successful suffix by v32 reverse BFS, five ternary target statuses, dynamic latched gate',
    challenge:'Matched four-way remote-error history; legal wrong-first navigation is genuinely executed, then independently certified optimal terminal recovery; per-legal-action regret available as supervised preference targets.',
    information_separation:'oracle_action_costs, accepted intervention, episode metadata and labels are teacher-only; never supply as model observation.',
    caveats:['Synthetic host, not native CyberSand.','Older remote wrong paint and gate opener are scripted host-accepted actions.','Only navigation mistakes injected after the agent start; no autonomous neural policy evaluated.','Model latency/RSS and checkpoint are not measured.'],
    first_action_recent64_uniform_ceiling:0.25,splits:manifestSplits,shards};
  fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,replay,chooseWrong,optionsCertificate};
if(require.main===module){if(!process.argv[2])throw Error('usage: node offpolicy_recovery_v33.js FRESH_DIR [--quick]');
  console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
