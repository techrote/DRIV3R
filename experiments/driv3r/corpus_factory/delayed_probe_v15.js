/* DRIV3R Corpus Factory v15 — delayed probe / shutter inversion parity.
 * External, deterministic synthetic host. NOT CyberSand native ABI or native evidence.
 * DATA LANE: BFS labels only; no model optimization.
 * node delayed_probe_v15.js NEW_OUTPUT_DIR [styleSeeds=2]
 */
"use strict";
const v13=require("./relay_memory_v13.js");
const crypto=require("node:crypto"),fs=require("node:fs"),path=require("node:path");
const SPLITS=["TRAIN","DEV","HELDOUT"];
const DIRS=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const clone=x=>structuredClone(x),key=p=>p.join(",");
const hash=x=>crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
function step(p,a){const d=DIRS.find(d=>d[0]===a);if(!d)throw Error("unknown move");return [p[0]+d[1],p[1]+d[2]];}
function actionFor(a,b){const d=DIRS.find(d=>a[0]+d[1]===b[0]&&a[1]+d[2]===b[1]);if(!d)throw Error("noncardinal oracle step");return d[0];}
function washout(host){
 // 7 identical stem excursions ensure last 64 locations and any short receipts alias.
 const actions=[];
 for(let r=0;r<7;r++){
  for(let i=0;i<9;i++){if(!v13.apply(host,"MOVE_NORTH").ok)throw Error("neutral north refused");actions.push("MOVE_NORTH");}
  for(let i=0;i<9;i++){if(!v13.apply(host,"MOVE_SOUTH").ok)throw Error("neutral south refused");actions.push("MOVE_SOUTH");}
 }
 return actions;
}
function script(host,side,flip){
 if(!["WEST","EAST"].includes(side)||![0,1].includes(flip))throw Error("invalid intervention");
 const h=host.episode.initial_model_input, w=host.w;
 h.objective="reach the southern merge after the distant shutter test and any remote inversion";
 h.history.objective=h.objective;
 // The old relay is intentionally absent: only the actual probe and inversion receipts
 // supply the long-range causal variables. Remove the v13 synthetic boot/toggle events.
 h.history.events=h.history.events.filter(e=>e.kind==="ambient_sensor");
 const actions=[];const take=a=>{if(!v13.apply(host,a).ok)throw Error("scripted legal action refused: "+a);actions.push(a);};
 const outward=side==="WEST"?"MOVE_WEST":"MOVE_EAST", inward=side==="WEST"?"MOVE_EAST":"MOVE_WEST";
 for(let i=0;i<w.span;i++)take(outward);
 for(let y=w.jy;y<w.by-4;y++)take("MOVE_SOUTH");
 const attempted=v13.apply(host,"MOVE_SOUTH"),result=attempted.ok?"accepted":"blocked";
 if(!attempted.ok&&attempted.reason!=="host_illegal_action")throw Error("unexpected refusal reason");
 if(attempted.ok){actions.push("MOVE_SOUTH");take("MOVE_NORTH");}
 const receiptStep=h.history.executed_positions.length+host.travelled.length-2;
 h.history.events.push({step:receiptStep,kind:"remote_probe_receipt",side,result});
 for(let y=w.by-4;y>w.jy;y--)take("MOVE_NORTH");
 for(let i=0;i<w.span;i++)take(inward);
 if(key(host.position)!==key(w.start))throw Error("failed to return to junction");
 if(flip)host.closed=host.closed==="WEST"?"EAST":"WEST";
 h.history.events.push({step:h.history.executed_positions.length+host.travelled.length-2,
  kind:"remote_inversion_receipt",line:"ALPHA",inversions:flip});
 actions.push(...washout(host));
 if(key(host.position)!==key(w.start)||host.done)throw Error("washout location drift");
 return {actions,probe_result:result,host_refusals:attempted.ok?0:1,washed_out_moves:126};
}
function eventControl(obs,include){
 const h=obs.history;
 return {...obs,history:{...h,executed_positions:h.executed_positions.slice(-64),
  events:h.events.filter(e=>include.includes(e.kind)).map(({step,...receipt})=>receipt)}};
}
const eventKinds=["remote_probe_receipt","remote_inversion_receipt"];
function makeEpisode(parent,side,flip){
 const host=v13.createHost(clone(parent));const prefix=script(host,side,flip),first=v13.observe(host);
 const route=v13.route(host.w,host.position,host.closed);
 if(!route||key(route.at(-1))!==key(host.w.goal))throw Error("no post-inversion goal route");
 const actions=route.slice(1).map((p,i)=>actionFor(route[i],p));
 const ep={schema:"driv3r/corpus-factory-delayed-probe-v15",split:parent.split,
  difficulty:"D8-delayed-probe-inversion-64-alias",task:"delayed_shutter_probe_parity",
  episode_id:`v15/${parent.provenance.world_id}/${side}/base${parent.episode_id.at(-1)}/flip${flip}`,
  quartet_id:`v15/${parent.provenance.world_id}/${side}`,
  group_id:parent.group_id,topology_group:parent.topology_group,
  provenance:{generator:"delayed_probe_v15.js",parent_source_git_blob:"b49afca90c307d8126c8f4c1becba667b7ced6f1",
   parent_episode_id:parent.episode_id,world_id:parent.provenance.world_id,synthetic_only:true},
  parent:{initial_model_input:clone(parent.initial_model_input),privileged_host:clone(parent.privileged_host)},
  intervention:{probe_side:side,inversion_count:flip},prefix_actions:prefix.actions,
  prefix_receipt:{result:prefix.probe_result,refused_moves:prefix.host_refusals},
  first_model_input:first,expert:{actions,positions:route,optimal_moves:actions.length},
  excluded_from_model:["intervention","prefix_receipt","parent","expert","provenance","episode_id","group_id","quartet_id","split"],
  native_receipt:false};
 return ep;
}
function createHost(ep){
 const host=v13.createHost(clone(ep.parent));const prefix=script(host,ep.intervention.probe_side,ep.intervention.inversion_count);
 if(hash(prefix.actions)!==hash(ep.prefix_actions)||prefix.probe_result!==ep.prefix_receipt.result||
    prefix.host_refusals!==ep.prefix_receipt.refused_moves)throw Error("script mismatch");
 if(hash(v13.observe(host))!==hash(ep.first_model_input))throw Error("observation mismatch");
 return host;
}
const observe=host=>v13.observe(host),apply=(host,a)=>v13.apply(host,a);
function replayAndLabel(ep){
 const host=createHost(ep),rows=[],moves=ep.expert.actions;
 const chosen=new Set([0,1,2,3,Math.floor(moves.length/4),Math.floor(moves.length/2),
  Math.floor(3*moves.length/4),moves.length-2,moves.length-1]);
 let strictlyLonger=0;
 for(let i=0;i<moves.length;i++){
  const obs=observe(host),target=moves[i];
  if(!obs.legal_actions.includes(target))throw Error("oracle target illegal");
  const best=v13.route(host.w,host.position,host.closed);
  if(!best||best.length-1!==moves.length-i)throw Error("oracle not shortest");
  if(chosen.has(i)){
   const negative=[],ties=[];
   for(const a of obs.legal_actions){if(a===target)continue;
    const remain=v13.route(host.w,step(host.position,a),host.closed);
    if(!remain||remain.length>best.length-1)negative.push(a);
    else if(remain.length===best.length-1)ties.push(a);
    else throw Error("distance inconsistency");
   }
   strictlyLonger+=negative.length;
   rows.push({schema:ep.schema,split:ep.split,task:ep.task,difficulty:ep.difficulty,
    episode_id:ep.episode_id,quartet_id:ep.quartet_id,group_id:ep.group_id,topology_group:ep.topology_group,
    time_index:i,model_input:obs,target,hard_negatives:{strictly_longer_legal:negative,
     blocked_or_illegal:DIRS.map(d=>d[0]).filter(a=>!obs.legal_actions.includes(a))},
    controls:{last64_only:v13.shortHistory(obs,64),last1_only:v13.shortHistory(obs,1),
     event_history:eventControl(obs,eventKinds),probe_only:eventControl(obs,["remote_probe_receipt"]),
     inversion_only:eventControl(obs,["remote_inversion_receipt"])},
    annotation:{equally_optimal_other_actions:ties,remaining_shortest_steps:best.length-1},
    excluded_model_fields:["task","difficulty","episode_id","quartet_id","group_id","topology_group",
     "split","time_index","target","hard_negatives","controls","annotation"],native_receipt:false});
  }
  if(!apply(host,target).ok)throw Error("oracle move refused");
 }
 if(!host.done||key(host.position)!==key(host.w.goal)||host.illegal_attempts!==ep.prefix_receipt.refused_moves)
  throw Error("oracle replay did not finish safely");
 return {rows,moves:moves.length,strictlyLonger};
}
function checkQuartets(quartets){
 let count=0;
 for(const [id,eps] of quartets){
  if(eps.length!==4)throw Error("quartet cardinality "+id);
  const first=eps.map(ep=>ep.first_model_input),target=eps.map(ep=>ep.expert.actions[0]);
  if(new Set(first.map(obs=>hash(v13.shortHistory(obs,64)))).size!==1)throw Error("last64 leak "+id);
  if(new Set(first.map(obs=>hash(eventControl(obs,eventKinds)))).size!==4)throw Error("event pair conflation");
  if(target.filter(a=>a==="MOVE_WEST").length!==2||target.filter(a=>a==="MOVE_EAST").length!==2)
   throw Error("quartet target imbalance");
  for(const mode of [["remote_probe_receipt"],["remote_inversion_receipt"]]){
   const classes=new Map();
   for(let i=0;i<4;i++){
    const sig=hash(eventControl(first[i],mode));
    const labels=classes.get(sig)||[];labels.push(target[i]);classes.set(sig,labels);
   }
   if(classes.size!==2||[...classes.values()].some(ls=>ls.length!==2||new Set(ls).size!==2))
    throw Error("one-event ablation does not force 50% "+mode);
  }
  count++;
 }
 return count;
}
function generateDataset(styleSeeds=2){
 if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>8)throw Error("styleSeeds 1..8");
 const parent=v13.generateDataset(styleSeeds),episodes={},decisions={},counts={},quartets=new Map(),worldSplits=new Map(),topologySplits=new Map();
 const worldViews=new Map(),seenInputs=new Set();let moves=0,refused=0,strict=0,washed=0;
 for(const split of SPLITS){
  episodes[split]=[];decisions[split]=[];
  for(const src of parent.episodes[split].filter(ep=>ep.episode_id.includes("/toggle0/"))){
   for(const side of ["WEST","EAST"])for(const flip of [0,1]){
    const ep=makeEpisode(src,side,flip),sig=hash(ep.first_model_input.local_material_7x7);
    if(worldViews.has(sig)&&worldViews.get(sig)!==ep.group_id)throw Error("duplicate start-view across world groups");
    worldViews.set(sig,ep.group_id);
    if(worldSplits.has(ep.group_id)&&worldSplits.get(ep.group_id)!==split)throw Error("world leakage");
    worldSplits.set(ep.group_id,split);
    if(topologySplits.has(ep.topology_group)&&topologySplits.get(ep.topology_group)!==split)throw Error("topology leakage");
    topologySplits.set(ep.topology_group,split);
    if(!quartets.has(ep.quartet_id))quartets.set(ep.quartet_id,[]);quartets.get(ep.quartet_id).push(ep);
    const r=replayAndLabel(ep);moves+=r.moves;strict+=r.strictlyLonger;
    refused+=ep.prefix_receipt.refused_moves;washed+=126;
    for(const row of r.rows){const h=hash(row.model_input);
     if(seenInputs.has(h))throw Error("duplicate full model input");seenInputs.add(h);
    }
    episodes[split].push(ep);decisions[split].push(...r.rows);
   }
  }
  counts[split]={episodes:episodes[split].length,decisions:decisions[split].length,
   world_groups:new Set(episodes[split].map(ep=>ep.group_id)).size,
   quartets:new Set(episodes[split].map(ep=>ep.quartet_id)).size,
   topology_groups:new Set(episodes[split].map(ep=>ep.topology_group)).size};
 }
 const n=checkQuartets(quartets);
 return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-delayed-probe-v15",version:"delayed-probe-v15",
  source:"read-only v13 synthetic host, not CyberSand native ABI",source_git_blob:"b49afca90c307d8126c8f4c1becba667b7ced6f1",
  style_seeds:styleSeeds,counts,counterfactual_quartets:n,expert_moves:moves,host_refused_probe_moves:refused,
  neutral_washout_moves:washed,strict_hard_negatives:strict,illegal_oracle_moves:0,protected_entries:0,
  split_policy:"source world and topology-disjoint; correlated probe/inversion quartets in one split",
  model_view:"Only observe(host) or row.model_input; never episode fields or row controls/targets",
  alias:"Each quartet has identical last64/current material/legal/goals, 2 WEST/2 EAST expert labels; probe-only and inversion-only each mathematically capped at 50% on paired first actions",
  limitations:["Synthetic v13 corridor grammar and scripted probe/inversion/washout, not native CyberSand",
   "One remote accepted/rejected probe per episode, at most one inversion; no stochastic physics",
   "Only post-intervention trajectory is model-controllable; oracle labels are not SLM results",
   "All decisions within a quartet/world correlated; evaluate by world groups, not row count",
   "Proof of initial-action alias does not prove full closed-loop learnability; HELDOUT sealed"]}};
}
function writeDataset(dest,styleSeeds=2){
 if(fs.existsSync(dest))throw Error("refuse overwrite");
 const out=generateDataset(styleSeeds);fs.mkdirSync(dest,{recursive:true});const shards={};
 for(const split of SPLITS)for(const kind of ["episodes","decisions"]){
  const filename=`${split}_${kind}.jsonl`,body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
  fs.writeFileSync(path.join(dest,filename),body,{flag:"wx"});shards[filename]={rows:out[kind][split].length,
   bytes:Buffer.byteLength(body),sha256:crypto.createHash("sha256").update(body).digest("hex"),
   usage:split==="TRAIN"?"TRAIN only":"evaluation only, never optimize"};
 }
 const manifest={...out.summary,generator:"delayed_probe_v15.js",shards};
 fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
 return manifest;
}
if(require.main===module){if(!process.argv[2])throw Error("Usage: node delayed_probe_v15.js NEW_OUTPUT_DIR [styleSeeds=2]");
 console.log(JSON.stringify(writeDataset(process.argv[2],process.argv[3]===undefined?2:Number(process.argv[3])).counts));}
module.exports={makeEpisode,createHost,observe,apply,eventControl,replayAndLabel,checkQuartets,generateDataset,writeDataset};
