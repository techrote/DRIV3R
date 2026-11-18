'use strict';
// DRIV3R Corpus Factory v36. Synthetic host environmental GOAL_REVISION at a
// legal between-actions boundary, followed by 160 accepted movement actions.
// NOT CyberSand-native. No policy learner; teacher labels are exact BFS suffixes.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js');
const host=require('./causal_gate_v26.js');
const view=require('./bounded_view_v30.js');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:3,DEV:2,HELDOUT:2};
const GOALS=[{name:'control_quota_2',quota:2},{name:'revised_quota_3',quota:3}];
const MODES=['expert','accepted_bad_first'];
function assert(ok,msg){if(!ok)throw Error(msg);}
function legalGo(h,to){
 const q=[h.pos],from=new Map([[h.pos,-1]]);
 for(let i=0;i<q.length&&!from.has(to);i++)for(const n of h.w.adjacency[q[i]])if(!from.has(n)&&host.edgeAllowed(h.w,q[i],n,h.latched)){from.set(n,q[i]);q.push(n);}
 assert(from.has(to),'legal route not found');
 const route=[];for(let n=to;n!==h.pos;n=from.get(n))route.push(n);route.reverse();
 for(const n of route){const action=host.options(h.w,[h.pos,h.prev,h.status,h.latched]).find(x=>x.next?.[0]===n);
  assert(action && host.apply(h,action.action).accepted,'movement not host-accepted');}
 return route.length;
}
function reviseGoal(h,quota){
 assert(!h.done&&h.pos===h.w.root&&h.latched===1,'revision outside permitted boundary');
 assert(!h.revisionApplied&&h.w.quota===2&&[2,3].includes(quota),'duplicate or invalid revision');
 const before=h.w.quota;h.w.quota=quota;h.revisionApplied=true;
 // This is a host-originating environment transition, NOT an agent action.
 const receipt={kind:'GOAL_REVISION',accepted:true,previous:{quota:before,exit:h.w.nodes[h.w.exit]},
  goal:{quota,exit:h.w.nodes[h.w.exit]},cause:'SYNTHETIC_ENVIRONMENT_SCHEDULE'};
 h.history.events.push(receipt);return receipt;
}
const PLAN_CACHE=new Map();
function planners(split,g,v,quota){
 const key=`${split}-${g}-${v}-${quota}`;
 if(!PLAN_CACHE.has(key)){
  const old=base.world(split,g,v),prior=base.oracle(old),now=base.world(split,g,v);now.quota=quota;
  PLAN_CACHE.set(key,{prior,after:quota===2?prior:base.oracle(now),geometry_sha256:now.geometry_sha256});
 }
 return PLAN_CACHE.get(key);
}
function start(split,g,v,arm,quota){
 const w=base.world(split,g,v);const prepared=base.prepare(w,split,g,v,arm),h=prepared.h;
 const cached=planners(split,g,v,quota);assert(cached.geometry_sha256===w.geometry_sha256,'planner/world geometry drift');
 const prior=cached.prior;const before=prior.rank([h.pos,h.prev,h.status,h.latched]);
 const receipt=reviseGoal(h,quota);let postAccepted=0;
 const north=w.nidx.get('0,-1');
 for(let i=0;i<80;i++){postAccepted+=legalGo(h,north);postAccepted+=legalGo(h,w.root);}
 assert(postAccepted===160&&h.pos===w.root&&h.prev===north,'revision aging not host-legally completed');
 const obs=host.observe(h);
 assert(obs.goal.quota===quota&&obs.history.events[0].goal.quota===2,'goal observation inconsistent');
 assert(!obs.history.events.slice(-64).some(x=>x.kind==='GOAL_REVISION'||x.action?.startsWith('PAINT_')),'revision/paint leaked into recent64');
 assert(host.correct(w,h.status).wrong===2,'wrong-material contamination lost');
 const after=cached.after;
 return {w,h,obs,receipt,postAccepted,before,after,prior,initial_receipts:clone(h.history),preparedAccepted:prepared.accepted};
}
function compress(obs){
 const x=view.compress(obs); // Base v30 survey contains the *old* goal. Correct to live public goal.
 x.v='synthetic-v36';x.goal=[obs.goal.quota,x.goal[1]];
 assert(x.goal[0]===obs.goal.quota,'stale revision in compressed view');
 assert(JSON.stringify(x).length<=8192,'bounded view too large');
 return x;
}
function withoutDurable(inp){return {...inp,durable:[]};}
// Conservative near-key deliberately PRESERVES all durable receipts, goal,
// material map, local painting and legality. Only budget and latest movement
// tail are omitted. Conflicting labels are fatal, never silently downweighted.
function nearKey(inp){const {left,recent,recent_positions,...rest}=inp;return hash(rest);}

function cert(rank){const best=rank[0].cost;assert(Number.isFinite(best),'no optimal suffix');
 return rank.map(x=>({action:x.action,cost:Number.isFinite(x.cost)?x.cost:null,
   regret:Number.isFinite(x.cost)?x.cost-best:null}));}
function chooseWrong(rank,g,v,arm,quota){
 const legal=rank.filter(x=>x.action.startsWith('MOVE_')||x.action==='BACKTRACK')
  .filter(x=>Number.isFinite(x.cost)&&x.cost>=rank[0].cost+2).sort((a,b)=>a.action.localeCompare(b.action));
 assert(legal.length>=1,'no strict navigational offpolicy option');
 return legal[(g+v+arm+quota)%legal.length];
}
function writer(dest,name){const fd=fs.openSync(path.join(dest,name),'wx');const h=crypto.createHash('sha256');let rows=0,bytes=0;
 return {write(value){const line=JSON.stringify(value)+'\n';fs.writeSync(fd,line);h.update(line);rows++;bytes+=Buffer.byteLength(line);},
  async finish(){fs.closeSync(fd);return {rows,bytes,sha256:h.digest('hex')};}};
}
function follow(item,mode,g,v,arm,emit){
 const {w,h,after,prior}=item;
 const rank0=after.rank([h.pos,h.prev,h.status,h.latched]);const best0=rank0[0];
 const wrong=chooseWrong(rank0,g,v,arm,w.quota);
 const initial=compress(host.observe(h));const initHash=hash(initial);
 let intervention=null;
 if(mode==='accepted_bad_first'){
  const ack=host.apply(h,wrong.action);assert(ack.accepted,'offpolicy action rejected');
  intervention={action:wrong.action,regret:wrong.cost-best0.cost,receipt:ack};
 }
 const trace=[];let step=0,erasures=0,frozenIllegal=0,frozenStrict=0,firstAction=null;
 while(!h.done){
  assert(step<200,'recovery runaway');
  const obs=host.observe(h),input=compress(obs),state=[h.pos,h.prev,h.status,h.latched],rank=after.rank(state),best=rank[0];
  assert(Number.isFinite(best.cost),'unreachable oracle state');
  const old=prior.rank(state)[0],actual=rank.find(x=>x.action===old.action);
  if(!actual)frozenIllegal++;else if(actual.cost>=best.cost+2)frozenStrict++;
  if(!firstAction)firstAction=best.action;
  const result=host.apply(h,best.action);assert(result.accepted,'oracle action rejected');
  trace.push({action:best.action,input_sha256:hash(input),suffix_cost:best.cost,
    receipt_sha256:hash(result)});
  emit({input,action:best.action,weight:1});
  if(best.action==='ERASE')erasures++;
  const next=h.done?0:after.rank([h.pos,h.prev,h.status,h.latched])[0].cost;
  assert(next===best.cost-1,'nonoptimal oracle suffix progression');step++;
 }
 assert(host.correct(w,h.status).wrong===0&&host.correct(w,h.status).count>=w.quota,'failed terminal');
 assert(step===(intervention?wrong.cost-1:best0.cost),'actual shortest post-intervention completion');
 assert(erasures===2,'incorrect erasure count');
 return {steps:step,erasures,first_action:firstAction,first_clean_action:best0.action,
  accepted_intervention:intervention,first_action_cost_certificate:cert(rank0),
  initial_input_sha256:initHash,initial_input:initial,
  goal_effect:{strict_frozen_quota2_decision_regrets:frozenStrict,
   frozen_quota2_illegal_actions:frozenIllegal},trace};
}
async function generate(dest,{quick=false}={}){
 assert(!fs.existsSync(dest),'refusing destination overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},allOwners=new Map(),geometryOwners=new Map();
 for(const [split,groups] of Object.entries(SPLITS)){
  const policies=writer(dest,`${split}_policy.jsonl`),episodes=writer(dest,`${split}_episodes.jsonl`);
  const challenge=split==='TRAIN'?null:writer(dest,`${split}_memory_challenge.jsonl`);
  const seen=new Map(),nearSeen=new Map(),familyRows=new Map(),worlds=new Set();
  let count=0,duplicates=0,nearDuplicates=0,weights=0,goalCausalEpisodes=0,goalStrictDecisions=0,badFirst=0,firstStrict=Infinity;
  let families=0,goalControls=0,goalRevisions=0;
  for(let g=0;g<(quick?1:groups);g++)for(let v=0;v<8;v++)for(const goal of GOALS){
   const group=`${split}-g${g}-v${v}-${goal.name}`;
   const starts=Array.from({length:4},(_,arm)=>start(split,g,v,arm,goal.quota));
   const initial=starts.map(x=>compress(x.obs));
   const histless=initial.map(x=>hash(withoutDurable(x)));
   assert(new Set(histless).size===1,'matched recent-only contexts not identical');
   assert(new Set(initial.map(hash)).size===4,'full memory aliases unexpectedly');
   assert(starts.every(x=>x.preparedAccepted===starts[0].preparedAccepted),'different arm prefix length');
   const first=starts.map(x=>x.after.rank([x.h.pos,x.h.prev,x.h.status,x.h.latched])[0].action);
   const maxFirst=Math.max(...[...new Set(first)].map(a=>first.filter(x=>x===a).length))/4;
   assert(maxFirst===(goal.quota===2?0.25:0.5),'unexpected recent-only action ceiling');
   const inputOwner=geometryOwners.get(starts[0].w.geometry_sha256);
   assert(!inputOwner||inputOwner===split,'cross split geometry overlap');geometryOwners.set(starts[0].w.geometry_sha256,split);
   worlds.add(starts[0].w.geometry_sha256);
   if(goal.quota===2)goalControls++;else goalRevisions++;
   families++;
   const grouped=[];
   for(let arm=0;arm<4;arm++)for(const mode of MODES){
    // Each episode requires its own fresh mutable host, not a reused state.
    const item=mode==='expert'?starts[arm]:start(split,g,v,arm,goal.quota);
    const collected=[];
    const run=follow(item,mode,g,v,arm,obj=>collected.push(obj));
    const id=`${group}-arm${arm}-${mode}`;
    if(goal.quota===3){assert(run.goal_effect.strict_frozen_quota2_decision_regrets>0 || run.goal_effect.frozen_quota2_illegal_actions>0,'revision did not causally change policy');goalCausalEpisodes++;}
    else assert(run.goal_effect.strict_frozen_quota2_decision_regrets===0&&run.goal_effect.frozen_quota2_illegal_actions===0,'false-positive stable goal effect');
    goalStrictDecisions+=run.goal_effect.strict_frozen_quota2_decision_regrets;
    if(mode==='accepted_bad_first'){badFirst++;assert(run.accepted_intervention.regret>=2,'bad move lacks regret');firstStrict=Math.min(firstStrict,run.accepted_intervention.regret);}
    if(mode==='expert')grouped.push({input:run.initial_input,action:run.first_clean_action});
    for(const row of collected){
      const inpHash=hash(row.input),prior=seen.get(inpHash),owner=allOwners.get(inpHash);
      assert(!owner||owner===split,'cross split exact input leak');allOwners.set(inpHash,split);
      if(prior!==undefined){assert(prior===row.action,'same bounded input, contradictory label');row.weight=0;duplicates++;}
      else seen.set(inpHash,row.action);
      const nk=nearKey(row.input),near=nearSeen.get(nk);
      if(near!==undefined){assert(near===row.action,'conservative near-context has opposite labels');nearDuplicates++;if(split==='TRAIN')row.weight=0;}
      else nearSeen.set(nk,row.action);
      if(row.weight)weights++;
      policies.write(row);count++;
    }
    episodes.write({schema:'driv3r/corpus_factory/goal-revision-v36/episode',id,split,group,arm,mode,goal_case:goal.name,
      synthetic_host_not_native:true,geometry_sha256:item.w.geometry_sha256,
      world_recipe_sha256:hash(item.w.recipes),goal_revision_receipt:item.receipt,
      prelude_accepted_actions:item.preparedAccepted,post_revision_accepted_moves:item.postAccepted,
      initial_history:item.initial_receipts,
      ...run,initial_input:undefined});
   }
   assert(grouped.length===4&&grouped.every((x,i)=>x.action===first[i]),'challenge/trace discrepancy');
   if(challenge)for(const x of grouped)challenge.write({input:x.input,historyless_input:withoutDurable(x.input),action:x.action});
   familyRows.set(group,{actions:first,historyless_max_accuracy:maxFirst});
  }
  shards[`${split}_policy.jsonl`]=await policies.finish();
  shards[`${split}_episodes.jsonl`]=await episodes.finish();
  if(challenge)shards[`${split}_memory_challenge.jsonl`]=await challenge.finish();
  splits[split]={geometry_groups:worlds.size,goal_families:families,stable_goal_families:goalControls,revised_goal_families:goalRevisions,
   episodes:families*4*MODES.length,decisions:count,unique_bounded_inputs:seen.size,
   weight_one:weights,weight_zero:count-weights,exact_duplicate_inputs:duplicates,near_duplicate_contexts:nearDuplicates,
   conservative_near_dedup_applied_to_training_only:true,goal_causal_episodes:goalCausalEpisodes,
   strict_goal_regret_decisions:goalStrictDecisions,accepted_detrimental_navigation:badFirst,
   min_detrimental_navigation_regret:firstStrict,
   challenge_examples:challenge?families*4:0,
   stable_recent_only_accuracy:0.25,revised_recent_only_accuracy:0.5};
 }
 const files=['goal_revision_v36.js','causal_shortcut_v32.js','causal_gate_v26.js','bounded_view_v30.js','outcome_forks_v25.js'];
 const manifest={schema:'driv3r/corpus_factory/goal-revision-v36/manifest',synthetic_host_not_native_CyberSand:true,quick,
  generator:files[0],generator_sha256:hash(fs.readFileSync(__filename)),
  dependencies:files.slice(1).map(file=>({file,sha256:hash(fs.readFileSync(path.join(__dirname,file)))})),
  provenance:{parent_v35:'trainer-safe-v35',design:'v32 graph/host with new host-originating goal revision after v32 accepted history, then 160 additional accepted moves',
    synthetic_host_only:true,revision_event_is_environment_not_agent_action:true},
  policy_contract:'Only row.input is model-visible; row.action is teacher label, row.weight is training loss weight (0 means exclude). Episodes and memory_challenge are teacher/eval files.',
  challenge_contract:'Within each 4-arm family identical recent/position/goal/legal/budget but different durable painting records. Exact recent-only first-action majority bound =25% stable quota2, 50% revised quota3. No learner claim.',
  goal_control:'Compare actual quota3 shortest-successful-suffix policy with frozen quota2 teacher at identical reachable states; strict >=2-action regret or invalid action is required in each revised episode. Stable quota2 is exact negative control.',
  non_goals:['Synthetic, not native CyberSand','No actual SLM training or autonomous neural rollouts','No claim of generalization to unseen geometry families under nonlinear renamings'],splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,start,compress,withoutDurable,nearKey,follow,reviseGoal,legalGo,hash};
if(require.main===module){const [dest,flag]=process.argv.slice(2);assert(dest,'usage: node goal_revision_v36.js FRESH_DIR [--quick]');
 generate(path.resolve(dest),{quick:flag==='--quick'}).then(x=>console.log(JSON.stringify(x.splits,null,2))).catch(e=>{console.error(e.stack);process.exitCode=1;});}
