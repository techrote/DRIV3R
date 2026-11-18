'use strict';
// Independent exact-content and host-action replay of v36. No model inference.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const gen=require('./goal_revision_v36.js'),base=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js');
const hash=gen.hash,assert=(x,m)=>{if(!x)throw Error(m);};
async function* records(file){const st=fs.createReadStream(file,{encoding:'utf8'}),rd=readline.createInterface({input:st,crlfDelay:Infinity});try{for await(const line of rd)if(line)yield JSON.parse(line);}finally{rd.close();st.destroy();}}
function fileMeta(file){const b=fs.readFileSync(file);return {bytes:b.length,sha256:hash(b)};}
function stable(x){return JSON.stringify(x);}
async function audit(dir){
 const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 assert(manifest.schema==='driv3r/corpus_factory/goal-revision-v36/manifest','wrong schema');
 assert(manifest.generator_sha256===hash(fs.readFileSync(path.join(__dirname,manifest.generator))),'generator source mismatch');
 for(const d of manifest.dependencies)assert(d.sha256===hash(fs.readFileSync(path.join(__dirname,d.file))),'dependency source mismatch '+d.file);
 for(const [name,stat]of Object.entries(manifest.shards)){
  const got=fileMeta(path.join(dir,name));assert(got.bytes===stat.bytes&&got.sha256===stat.sha256,'file checksum '+name);
 }
 const splitGeometry=new Map(),splitInputs=new Map(),result={};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const pol=records(path.join(dir,`${split}_policy.jsonl`))[Symbol.asyncIterator]();
  const summary={episodes:0,decisions:0,matched_families:new Map(),near_seen:new Map(),near_duplicates:0,goal_revisions:0,bad_first:0,strict_goal:0,weights:0,zero:0};
  let challengeIndex=0, firstGroup=[];
  for await(const ep of records(path.join(dir,`${split}_episodes.jsonl`))){
   const m=ep.id.match(/^(TRAIN|DEV|HELDOUT)-g(\d+)-v(\d+)-(control_quota_2|revised_quota_3)-arm([0-3])-(expert|accepted_bad_first)$/);
   assert(m&&m[1]===split,'bad episode metadata/id');
   const g=+m[2],v=+m[3],quota=m[4]==='control_quota_2'?2:3,arm=+m[5],mode=m[6];
   assert(ep.schema==='driv3r/corpus_factory/goal-revision-v36/episode'&&ep.synthetic_host_not_native===true,'provenance absent');
   const o=splitGeometry.get(ep.geometry_sha256);assert(!o||o===split,'geometry split collision');splitGeometry.set(ep.geometry_sha256,split);
   const item=gen.start(split,g,v,arm,quota),h=item.h;
   assert(item.w.geometry_sha256===ep.geometry_sha256,'geometry replay drift');
   assert(hash(item.initial_receipts)===hash(ep.initial_history),'host prefix mismatch');
   assert(ep.goal_revision_receipt.goal.quota===quota&&item.receipt.accepted,'revision receipt mismatch');
   assert(ep.post_revision_accepted_moves===160&&item.postAccepted===160,'missing host-accepted age');
   const original=host.observe(h),compact=gen.compress(original);
   assert(compact.goal[0]===quota&&original.goal.quota===quota,'stale live goal');
   assert(original.history.events[0].goal.quota===2,'initial survey overwritten');
   assert(!original.history.events.slice(-64).some(x=>x.kind==='GOAL_REVISION'||x.action?.startsWith('PAINT')),'recent64 leak');
   const initialRank=item.after.rank([h.pos,h.prev,h.status,h.latched]);
   assert(ep.first_clean_action===initialRank[0].action,'wrong first action certificate');
   if(mode==='accepted_bad_first'){
    assert(ep.accepted_intervention?.regret>=2,'missing regret intervention');
    const receipt=host.apply(h,ep.accepted_intervention.action);
    assert(receipt.accepted&&hash(receipt)===hash(ep.accepted_intervention.receipt),'unreplayable bad first action');summary.bad_first++;
   } else assert(ep.accepted_intervention===null,'unexpected intervention');
   let strict=0,invalid=0,erases=0;
   for(let i=0;i<ep.trace.length;i++){
    const next=await pol.next();assert(!next.done,'missing policy rows');const row=next.value;
    const keys=Object.keys(row).sort().join(',');assert(keys==='action,input,weight','policy row includes teacher metadata');
    assert(Object.keys(row.input).sort().join(',')===['v','p','legal','goal','ok','left','cells','edges','targets','protected','gate','local','recent','recent_positions','durable'].sort().join(','),'input schema contamination');
    assert(!stable(row.input).includes('episode_id')&&!stable(row.input).includes('oracle')&&!stable(row.input).includes('regret'),'leaked teacher keys');
    const observed=host.observe(h),modelInput=gen.compress(observed),trace=ep.trace[i];
    assert(stable(row.input)===stable(modelInput),'model observation mismatch '+ep.id+':'+i);
    assert(trace.input_sha256===hash(row.input),'source observation digest mismatch');
    const state=[h.pos,h.prev,h.status,h.latched],rank=item.after.rank(state);
    assert(rank[0].action===row.action&&rank[0].cost===trace.suffix_cost,'nonoracle teacher label');
    assert(row.action===trace.action&&row.input.legal.includes(row.action),'teacher illegal action');
    const old=item.prior.rank(state)[0],possible=rank.find(x=>x.action===old.action);
    if(!possible)invalid++;else if(possible.cost>=rank[0].cost+2)strict++;
    const ack=host.apply(h,row.action);assert(ack.accepted&&hash(ack)===trace.receipt_sha256,'host denied replay or receipt drift');
    if(row.action==='ERASE')erases++;
    const after=h.done?0:item.after.rank([h.pos,h.prev,h.status,h.latched])[0].cost;
    assert(after===rank[0].cost-1,'not shortest successful suffix');
    const inputHash=hash(row.input),owner=splitInputs.get(inputHash);assert(!owner||owner===split,'cross-split model-input collision');splitInputs.set(inputHash,split);
    const near=gen.nearKey(row.input),oldNear=summary.near_seen.get(near);
    if(oldNear!==undefined){assert(oldNear===row.action,'conflicting near-key labels');summary.near_duplicates++;}
    else summary.near_seen.set(near,row.action);
    const expectedWeight=split==='TRAIN'&&oldNear!==undefined?0:1;
    assert(row.weight===expectedWeight,'conservative TRAIN near-dedup mismatch');
    if(row.weight)summary.weights++;else summary.zero++;
    summary.decisions++;
   }
   assert(h.done&&host.correct(item.w,h.status).wrong===0&&host.correct(item.w,h.status).count>=quota,'not successfully terminated');
   assert(ep.steps===ep.trace.length&&erases===2&&ep.erasures===erases,'wrong terminal steps/erasures');
   assert(ep.goal_effect.strict_frozen_quota2_decision_regrets===strict&&ep.goal_effect.frozen_quota2_illegal_actions===invalid,'counterfactual frozen goal mismatch');
   if(quota===2)assert(strict===0&&invalid===0,'negative goal control effect');
   else {assert(strict+invalid>0,'no causal revised goal outcome');summary.goal_revisions++;summary.strict_goal+=strict;}
   if(mode==='expert'){
    const family=ep.group;const set=summary.matched_families.get(family)||[];
    set.push({arm,action:ep.first_clean_action,input:compact});summary.matched_families.set(family,set);
   }
   summary.episodes++;
  }
  assert((await pol.next()).done,'trailing policy records');
  const expected=manifest.splits[split];
  assert(summary.episodes===expected.episodes&&summary.decisions===expected.decisions,'count mismatch');
  assert(summary.weights===expected.weight_one&&summary.zero===expected.weight_zero,'sample weight mismatch');
  assert(summary.near_duplicates===expected.near_duplicate_contexts,'near duplicate metric mismatch');
  assert(summary.goal_revisions===expected.goal_causal_episodes&&summary.bad_first===expected.accepted_detrimental_navigation,'cohort mismatch');
  assert(summary.strict_goal===expected.strict_goal_regret_decisions,'causal strict tally');
  const groups=[...summary.matched_families].sort(([a],[b])=>a.localeCompare(b));
  assert(groups.length===expected.goal_families,'family count mismatch');
  const indexed=new Map();
  for(const [id,items] of groups){
   assert(items.length===4&&new Set(items.map(x=>x.arm)).size===4,'incomplete family');
   const aliases=items.map(x=>hash(gen.withoutDurable(x.input)));
   assert(new Set(aliases).size===1,'recent-only input not paired');
   assert(new Set(items.map(x=>hash(x.input))).size===4,'full-history pair collision');
   const frequencies=new Map();for(const item of items)frequencies.set(item.action,(frequencies.get(item.action)||0)+1);
   const bound=Math.max(...frequencies.values())/4,quota=items[0].input.goal[0];
   assert(bound===(quota===2?.25:.5),'wrong theoretical memory bound');
   indexed.set(id,items.sort((a,b)=>a.arm-b.arm));
  }
  if(split!=='TRAIN'){
   // First-action challenge is strictly evaluation-only, ordered by g/v/goal/arm.
   const k=records(path.join(dir,`${split}_memory_challenge.jsonl`))[Symbol.asyncIterator]();
   for(const [id,items]of [...indexed].sort(([a],[b])=>{
    function key(v){const x=v.match(/-g(\d+)-v(\d+)-(control_quota_2|revised_quota_3)$/);return [+x[1],+x[2],x[3]==='control_quota_2'?0:1]};
    const x=key(a),y=key(b);return x[0]-y[0]||x[1]-y[1]||x[2]-y[2];
   }))for(const item of items){
    const next=await k.next();assert(!next.done,'missing challenge row');
    assert(stable(next.value.input)===stable(item.input)&&next.value.action===item.action,'challenge full input discrepancy');
    assert(stable(next.value.historyless_input)===stable(gen.withoutDurable(item.input)),'historyless ablation mismatch');
    challengeIndex++;
   }
   assert((await k.next()).done&&challengeIndex===expected.challenge_examples,'unexpected challenge tail');
  }
  result[split]={episodes:summary.episodes,decisions:summary.decisions,goalCausalEpisodes:summary.goal_revisions,
   strictFrozenGoalRegrets:summary.strict_goal,challenge:challengeIndex,geometry:expected.geometry_groups};
 }
 return result;
}
async function main(){const [a,b]=process.argv.slice(2);assert(a,'usage node test_goal_revision_v36.js GENERATED_DIR [REPEAT_DIR]');
 const first=await audit(path.resolve(a));
 if(b){const m=JSON.parse(fs.readFileSync(path.join(a,'manifest.json'))),n=JSON.parse(fs.readFileSync(path.join(b,'manifest.json')));
  assert(stable(m)===stable(n),'two complete generations manifest mismatch');for(const name of Object.keys(m.shards))assert(fileMeta(path.join(a,name)).sha256===fileMeta(path.join(b,name)).sha256,'nondeterministic shard '+name);}
 console.log(JSON.stringify({independent_host_replay:'PASS',sha256_all_shards:'PASS',full_model_input_replay:'PASS',cross_split_leakage:'PASS',goal_revision_causal_and_negative:'PASS',reproducibility:b?'PASS':'NOT_RUN',splits:first},null,2));}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={audit};
