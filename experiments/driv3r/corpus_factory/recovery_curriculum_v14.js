/* DRIV3R Corpus Factory v14: off-policy recovery and in-flight objective changes.
 * External deterministic synthetic host; NOT CyberSand native evidence or ABI.
 * DATA ONLY; this module does not train a policy.
 * node recovery_curriculum_v14.js NEW_OUTPUT_DIR [styleSeeds=2]
 * Feed ONLY observe(host) or decision.model_input to a model; never privileged fields.
 */
"use strict";
const v13=require("./relay_memory_v13.js");
const crypto=require("node:crypto"),fs=require("node:fs"),path=require("node:path");
const SPLITS=["TRAIN","DEV","HELDOUT"];
const DIRS=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const key=p=>p.join(",");
const clone=x=>structuredClone(x);
function step(p,action){const d=DIRS.find(v=>v[0]===action);if(!d)throw Error("unknown move");return [p[0]+d[1],p[1]+d[2]];}
function actionFor(a,b){const d=DIRS.find(v=>a[0]+v[1]===b[0]&&a[1]+v[2]===b[1]);if(!d)throw Error("noncardinal route");return d[0];}
function distanceToGoal(w,closed){
 const dist=new Map([[key(w.goal),0]]),q=[w.goal];
 for(let i=0;i<q.length;i++)for(const a of v13.legal(w,q[i],closed)){
  const t=step(q[i],a),k=key(t);if(!dist.has(k)){dist.set(k,dist.get(key(q[i]))+1);q.push(t);}
 }
 return dist;
}
function scriptedPrefix(host,mode){
 const w=host.w,closed=host.closed,wrong=closed==="WEST"?-1:1;
 const actions=[];const take=a=>{const r=v13.apply(host,a);if(!r.ok)throw Error("illegal scripted prefix "+mode+" "+a);actions.push(a);};
 if(mode==="closed_gate_probe"){
  for(let i=0;i<w.span;i++)take(wrong<0?"MOVE_WEST":"MOVE_EAST");
  for(let y=w.jy;y<w.by-4;y++)take("MOVE_SOUTH");
  const blocked=v13.apply(host,"MOVE_SOUTH");
  if(blocked.ok||blocked.reason!=="host_illegal_action")throw Error("expected shutter refusal");
  host.episode.initial_model_input.history.events.push({step:host.episode.initial_model_input.history.executed_positions.length+host.travelled.length-2,
   kind:"host_rejected_move",action:"MOVE_SOUTH",reason:"host_illegal_action"});
 } else if(mode==="shutter_reversal"){
  const originalOpen=closed==="WEST"?1:-1;
  for(let i=0;i<w.span;i++)take(originalOpen<0?"MOVE_WEST":"MOVE_EAST");
  for(let y=w.jy;y<w.by-Math.max(5,Math.floor(w.height/2));y++)take("MOVE_SOUTH");
  host.closed=closed==="WEST"?"EAST":"WEST";
  host.episode.initial_model_input.history.events.push({step:host.episode.initial_model_input.history.executed_positions.length+host.travelled.length-2,
   kind:"remote_shutter_change_receipt",line:"ALPHA",event:"inversion",observed_by_host:true});
 } else if(mode==="goal_retarget"){
  const originalOpen=closed==="WEST"?1:-1;
  for(let i=0;i<w.span;i++)take(originalOpen<0?"MOVE_WEST":"MOVE_EAST");
  for(let y=w.jy;y<w.by-Math.max(5,Math.floor(w.height/2));y++)take("MOVE_SOUTH");
  w.goal=[w.cx+wrong*w.span,w.by-5];
  host.episode.initial_model_input.goal=w.goal.slice();
  host.episode.initial_model_input.objective="reach the newly marked survey target without entering protected cells";
  host.episode.initial_model_input.history.objective=host.episode.initial_model_input.objective;
  host.episode.initial_model_input.history.events.push({step:host.episode.initial_model_input.history.executed_positions.length+host.travelled.length-2,
   kind:"objective_retarget_receipt",new_goal:w.goal.slice(),observed_by_host:true});
 } else throw Error("unknown intervention");
 if(host.done)throw Error("scripted prefix already finished");
 return actions;
}
function createHost(episode){
 const host=v13.createHost(clone(episode.parent));const actions=scriptedPrefix(host,episode.intervention);
 if(JSON.stringify(actions)!==JSON.stringify(episode.prefix_actions))throw Error("prefix source drift");
 if(JSON.stringify(v13.observe(host))!==JSON.stringify(episode.first_model_input))throw Error("initial recovery observation drift");
 return host;
}
const observe=host=>v13.observe(host);
const apply=(host,action)=>v13.apply(host,action);
function sampleTimes(len){const times=new Set([0,1,2,3,len-1,Math.floor(len/3),Math.floor(2*len/3)]);
 for(let i=0;i<8;i++)times.add(Math.floor(i*(len-1)/7));return [...times].filter(x=>x>=0&&x<len).sort((a,b)=>a-b);}
function makeRecovery(parent,intervention){
 const host=v13.createHost(clone(parent)),prefix=scriptedPrefix(host,intervention),initial=observe(host);
 const planned=v13.route(host.w,host.position,host.closed);
 if(!planned||planned.length<2||key(planned[planned.length-1])!==key(host.w.goal))throw Error("unreachable recovery");
 const moves=planned.slice(1).map((p,i)=>actionFor(planned[i],p));
 const ep={schema:"driv3r/corpus-factory-recovery-v14",source_schema:parent.schema,
  split:parent.split,task:intervention,difficulty:intervention==="closed_gate_probe"?"D6-rejected-probe-recovery":"D7-inflight-replanning",
  episode_id:parent.episode_id+"/"+intervention,pair_id:parent.pair_id,group_id:parent.group_id,
  topology_group:parent.topology_group,provenance:{generator:"recovery_curriculum_v14.js",
   parent_generator:"relay_memory_v13.js",parent_episode_id:parent.episode_id,
   parent_world:parent.provenance.world_id,synthetic_only:true},
  parent:{initial_model_input:clone(parent.initial_model_input),privileged_host:clone(parent.privileged_host)},
  intervention,prefix_actions:prefix,prefix_moves:prefix.length,
  rejected_prefix_actions:intervention==="closed_gate_probe"?1:0,
  first_model_input:initial,privileged_host:{closed_after:host.closed,goal_after:host.w.goal.slice()},
  expert_recovery:{actions:moves,positions:planned,shortest_steps:moves.length},native_receipt:false,
  model_input_contract:"Only first_model_input or observe(host); NEVER parent.privileged_host/expert_recovery/metadata"};
 return ep;
}
function replayAndLabel(ep){
 const host=createHost(ep),dist=distanceToGoal(host.w,host.closed),want=ep.expert_recovery.actions;
 if(dist.get(key(host.position))!==want.length)throw Error("recovery not shortest");
 const rows=[],times=new Set(sampleTimes(want.length));let strict=0,legalMistakes=0;
 for(let i=0;i<want.length;i++){
  const obs=observe(host),target=want[i],distance=dist.get(key(host.position));
  if(!obs.legal_actions.includes(target)||dist.get(key(step(host.position,target)))!==distance-1)
   throw Error("oracle move invalid/suboptimal");
  if(times.has(i)){
   const strictlyLonger=[],tieAlternates=[];
   for(const a of obs.legal_actions){if(a===target)continue;
    const remainder=dist.get(key(step(host.position,a)));
    if(remainder===undefined||1+remainder>distance)strictlyLonger.push(a);
    else if(1+remainder===distance)tieAlternates.push(a);
    else throw Error("distance triangle inequality");
   }
   strict+=strictlyLonger.length;legalMistakes+=tieAlternates.length;
   rows.push({schema:ep.schema,split:ep.split,task:ep.task,difficulty:ep.difficulty,
    episode_id:ep.episode_id,pair_id:ep.pair_id,group_id:ep.group_id,topology_group:ep.topology_group,
    time_index:i,model_input:obs,target,hard_negatives:{strictly_longer_legal:strictlyLonger,
     illegal_or_blocked:DIRS.map(d=>d[0]).filter(a=>!obs.legal_actions.includes(a))},
    controls:{last64_only:v13.shortHistory(obs,64),last1_only:v13.shortHistory(obs,1)},
    annotation:{equally_optimal_other_actions:tieAlternates,remaining_shortest_steps:distance},
    excluded_model_fields:["task","difficulty","episode_id","pair_id","group_id","topology_group",
     "split","time_index","target","hard_negatives","controls","annotation"],native_receipt:false});
  }
  const r=apply(host,target);if(!r.ok)throw Error("oracle recovery refused");
 }
 if(!host.done||key(host.position)!==key(host.w.goal)||host.applied!==want.length+ep.prefix_moves||
    host.illegal_attempts!==ep.rejected_prefix_actions)throw Error("recovery replay failed");
 return {rows,recovery_moves:want.length,prefix_moves:ep.prefix_moves,strict,legalMistakes};
}
function generateDataset(styleSeeds=2){
 if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>8)throw Error("styleSeeds must be 1..8");
 const prior=v13.generateDataset(styleSeeds),episodes={TRAIN:[],DEV:[],HELDOUT:[]},decisions={TRAIN:[],DEV:[],HELDOUT:[]};
 const counts={},byTopology=new Map(),seenEpisodeIds=new Set(),seenInputs=new Set();
 let accepted=0,recovered=0,strict=0,rejected=0,tieAlternates=0;
 for(const split of SPLITS){
  for(const parent of prior.episodes[split])for(const mode of ["closed_gate_probe","shutter_reversal","goal_retarget"]){
   const ep=makeRecovery(parent,mode);
   if(seenEpisodeIds.has(ep.episode_id))throw Error("duplicate episode ID");seenEpisodeIds.add(ep.episode_id);
   if(byTopology.has(ep.topology_group)&&byTopology.get(ep.topology_group)!==split)throw Error("topology leakage");
   byTopology.set(ep.topology_group,split);
   const result=replayAndLabel(ep);
   for(const row of result.rows){
    const sig=crypto.createHash("sha256").update(JSON.stringify(row.model_input)).digest("hex");
    if(seenInputs.has(sig))throw Error("duplicate model input across dataset");seenInputs.add(sig);
   }
   accepted+=result.prefix_moves;recovered+=result.recovery_moves;rejected+=ep.rejected_prefix_actions;
   strict+=result.strict;tieAlternates+=result.legalMistakes;
   episodes[split].push(ep);decisions[split].push(...result.rows);
  }
  counts[split]={episodes:episodes[split].length,decisions:decisions[split].length,
   world_groups:new Set(episodes[split].map(x=>x.group_id)).size,
   topologies:new Set(episodes[split].map(x=>x.topology_group)).size,
   interventions:Object.fromEntries(["closed_gate_probe","shutter_reversal","goal_retarget"].map(k=>[k,episodes[split].filter(e=>e.intervention===k).length]))};
 }
 return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-recovery-v14",version:"off-policy-recovery-v14",
  provenance:"Synthetic deterministic host derived from v13; NOT canonical CyberSand ABI or native receipts",
  parent_source_git_blob:"b49afca90c307d8126c8f4c1becba667b7ced6f1",style_seeds:styleSeeds,counts,
  accepted_intervention_moves:accepted,oracle_recovery_moves:recovered,rejected_prefix_moves:rejected,
  protected_entries:0,illegal_oracle_moves:0,strict_hard_negative_actions:strict,tie_optimal_alternates:tieAlternates,
  grouping:"Inherited v13 topology/world split; all three interventions and counterfactual siblings grouped",
  memory_limit:"Long prelude retained but v14 does NOT claim last64 alias or history necessity after interventions",
  limitations:["Scripted error/retarget/shutter intervention; model chooses only subsequent recovery actions",
   "Synthetic grid/material, no native CyberSand ABI or brush semantics",
   "Oracle BFS labels, not primary SLM results; correlated recovery decisions by world",
   "Do not pass IDs, intervention labels, expert, or privileged host to model",
   "HELDOUT is sealed; use TRAIN for fitting and DEV for selection"]}};
}
function writeDataset(dest,styleSeeds=2){
 if(fs.existsSync(dest))throw Error("refuse to overwrite destination");
 const out=generateDataset(styleSeeds),shards={};fs.mkdirSync(dest,{recursive:true});
 for(const split of SPLITS)for(const kind of ["episodes","decisions"]){
  const file=`${split}_${kind}.jsonl`,body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
  fs.writeFileSync(path.join(dest,file),body,{flag:"wx"});shards[file]={rows:out[kind][split].length,
   bytes:Buffer.byteLength(body),sha256:crypto.createHash("sha256").update(body).digest("hex"),
   usage:split==="TRAIN"?"TRAIN only":"evaluation only: never optimize"};
 }
 const manifest={...out.summary,generator:"recovery_curriculum_v14.js",shards,
  model_input_contract:"Only observe(host) or decision.model_input; exclude other row and episode fields"};
 fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
 return manifest;
}
if(require.main===module){
 if(!process.argv[2])throw Error("Usage: node recovery_curriculum_v14.js NEW_OUTPUT_DIR [styleSeeds=2]");
 const n=process.argv[3]===undefined?2:Number(process.argv[3]);
 console.log(JSON.stringify(writeDataset(process.argv[2],n).counts));
}
module.exports={createHost,observe,apply,makeRecovery,replayAndLabel,generateDataset,writeDataset};
