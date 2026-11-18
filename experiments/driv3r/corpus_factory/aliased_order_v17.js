/* DRIV3R Corpus Factory v17: controlled prefix/suffix alias with held-out SET separation.
 * DATA-ONLY external synthetic host; NOT native CyberSand ABI/receipts.
 * Uses v13 Git blob b49afca90c307d8126c8f4c1becba667b7ced6f1.
 * Derived from v16 blob 5d64ac0558349f5efe7f40ff866204006a2a1fa7.
 * Usage: node aliased_order_v17.js NEW_OUTPUT_DIR [styleSeeds=2]
 */
"use strict";
const v13=require("./relay_memory_v13.js");
const crypto=require("node:crypto"),fs=require("node:fs"),path=require("node:path");
const SPLITS=["TRAIN","DEV","HELDOUT"];
const DIRS=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const clone=x=>structuredClone(x),positionKey=x=>x.join(",");
const sha=x=>crypto.createHash("sha256").update(JSON.stringify(x)).digest("hex");
const PARENT_SOURCE_BLOB="b49afca90c307d8126c8f4c1becba667b7ced6f1";
const TOKENS=["SET_WEST","SET_EAST",...Array(10).fill("TOGGLE")];
function permutations(tokens){
 const out=[];
 if(tokens.length!==12||tokens.filter(x=>x==="SET_WEST").length!==1||
    tokens.filter(x=>x==="SET_EAST").length!==1)throw Error("unsupported token bag");
 for(let w=0;w<12;w++)for(let e=0;e<12;e++){
  if(w===e)continue;const a=Array(12).fill("TOGGLE");
  a[w]="SET_WEST";a[e]="SET_EAST";out.push(a);
 }
 return out.sort((a,b)=>a.join("/").localeCompare(b.join("/")));
}
const ORDERS=permutations(TOKENS);
if(ORDERS.length!==132)throw Error("expected 132 noncommutative orderings");
// Predeclared 36 disjoint sequences; all contain exactly the same 12-token bag,
// two common initial TOGGLEs, and two common terminal TOGGLEs.
// TRAIN: adjacent SETs; DEV: one TOGGLE separating SETs;
// HELDOUT: two to six TOGGLEs separating SETs. This is distribution shift,
// not evidence that any neural model has generalised.
// W=SET_WEST, E=SET_EAST, T=TOGGLE. Labels balanced in each split.
const ORDER_CODES={
 TRAIN:["TTEWTTTTTTTT", "TTTTEWTTTTTT", "TTTTTTEWTTTT", "TTTTTTTTEWTT", "TTTTTTTWETTT", "TTTTTWETTTTT", "TTTEWTTTTTTT", "TTTTTEWTTTTT", "TTTTTTTEWTTT", "TTTTTTTTWETT", "TTTTTTWETTTT", "TTTTWETTTTTT"],
 DEV:["TTTETWTTTTTT", "TTTTTETWTTTT", "TTTTTTTETWTT", "TTTTTTWTETTT", "TTTTWTETTTTT", "TTWTETTTTTTT", "TTETWTTTTTTT", "TTTTETWTTTTT", "TTTTTTETWTTT", "TTTTTTTWTETT", "TTTTTWTETTTT", "TTTWTETTTTTT"],
 HELDOUT:["TTETTWTTTTTT", "TTTTETTWTTTT", "TTTETTWTTTTT", "TTTTTETTWTTT", "TTTETTTWTTTT", "TTETTTWTTTTT", "TTETTTTWTTTT", "TTTETTTTWTTT", "TTTETTTTTWTT", "TTETTTTTWTTT", "TTETTTTTTWTT", "TTWTTTTTTETT"],
};
const decode=c=>[...c].map(x=>({W:"SET_WEST",E:"SET_EAST",T:"TOGGLE"})[x]);
const ORDERS_BY_SPLIT=Object.fromEntries(SPLITS.map(s=>[s,ORDER_CODES[s].map(decode)]));
if(new Set(Object.values(ORDER_CODES).flat()).size!==36||
   Object.values(ORDERS_BY_SPLIT).some(orders=>orders.length!==12))throw Error("order split leakage");
function applyGateCommand(closed,cmd){
 if(cmd==="SET_WEST")return "WEST";
 if(cmd==="SET_EAST")return "EAST";
 if(cmd==="TOGGLE")return closed==="WEST"?"EAST":"WEST";
 throw Error("unknown gate instruction");
}
function closedFor(order){return order.reduce(applyGateCommand,"WEST");}
function modelHistoryProjection(obs,n){return v13.shortHistory(obs,n);}
function eventProjection(obs,mode){
 const events=obs.history.events;
 const alpha=events.filter(e=>e.line==="ALPHA"&&e.kind==="remote_instruction");
 let selected;
 switch(mode){
  case "bag": selected=events.map(({step,...e})=>e).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));break;
  case "first": selected=alpha.slice(0,1).map(({step,...e})=>e);break;
  case "first2": selected=alpha.slice(0,2).map(({step,...e})=>e);break;
  case "first3": selected=alpha.slice(0,3).map(({step,...e})=>e);break;
  case "last": selected=alpha.slice(-1).map(({step,...e})=>e);break;
  case "last2": selected=alpha.slice(-2).map(({step,...e})=>e);break;
  case "last3": selected=alpha.slice(-3).map(({step,...e})=>e);break;
  case "no_events": selected=[];break;
  default:throw Error("unknown projection");
 }
 return {...obs,history:{...obs.history,executed_positions:obs.history.executed_positions.slice(-64),events:selected}};
}
function actionFor(a,b){const d=DIRS.find(d=>a[0]+d[1]===b[0]&&a[1]+d[2]===b[1]);
 if(!d)throw Error("non-cardinal expert movement");return d[0];}
function nextPos(p,a){const d=DIRS.find(d=>d[0]===a);if(!d)throw Error("unknown move");return [p[0]+d[1],p[1]+d[2]];}
function runPrefix(host,order,noise){
 const initial=host.episode.initial_model_input;
 initial.objective="reach the southern merge after obeying ordered remote shutter instructions";
 initial.history={objective:initial.objective,executed_positions:[host.w.start.slice()],events:[]};
 host.closed="WEST"; // deterministic initial remote-gate state, invisible at the junction
 const actions=[];const events=initial.history.events;
 const take=a=>{const receipt=v13.apply(host,a);if(!receipt.ok)throw Error("script move refused: "+a);actions.push(a);};
 let seq=0;
 for(const command of order){
  for(let i=0;i<9;i++)take("MOVE_NORTH");
  host.closed=applyGateCommand(host.closed,command);
  const timestamp=initial.history.executed_positions.length+host.travelled.length-2;
  events.push({step:timestamp,kind:"remote_instruction",line:"ALPHA",command});
  // Noncausal sensor noise, same multiset and timestamp for all sibling permutations.
  const filler=["TOGGLE","SET_EAST","SET_WEST","TOGGLE"];
  for(let j=0;j<noise;j++){
   const n=(seq*noise+j)%filler.length;
   events.push({step:timestamp,kind:"remote_instruction",line:"BETA",command:filler[n],ignored_by_alpha:true});
  }
  for(let i=0;i<9;i++)take("MOVE_SOUTH");seq++;
 }
 for(let i=0;i<7;i++){
  for(let k=0;k<9;k++)take("MOVE_NORTH");
  for(let k=0;k<9;k++)take("MOVE_SOUTH");
 }
 if(positionKey(host.position)!==positionKey(host.w.start)||host.done||actions.length!==342)
  throw Error("prefix did not return to start");
 if(host.closed!==closedFor(order))throw Error("gate command execution inconsistent");
 return {actions,commands:order.slice(),closed:host.closed,noise};
}
function makeEpisode(parent,order,orderIndex,noise){
 const host=v13.createHost(clone(parent)),prefix=runPrefix(host,order,noise);
 const start=v13.observe(host),route=v13.route(host.w,host.position,host.closed);
 if(!route||positionKey(route.at(-1))!==positionKey(host.w.goal))throw Error("goal unreachable");
 const actions=route.slice(1).map((p,i)=>actionFor(route[i],p));
 return {schema:"driv3r/corpus-factory-aliased-order-v17",split:parent.split,
  task:"aliased_order_with_set_gap_transfer",difficulty:"D10-controlled-set-gap-transfer-64-alias",
  episode_id:`v17/${parent.provenance.world_id}/order${orderIndex}`,
  family_id:`v17/${parent.provenance.world_id}`,group_id:parent.group_id,topology_group:parent.topology_group,
  provenance:{generator:"aliased_order_v17.js",parent_source_git_blob:PARENT_SOURCE_BLOB,
   parent_episode_id:parent.episode_id,world_id:parent.provenance.world_id,synthetic_only:true},
  parent:{initial_model_input:clone(parent.initial_model_input),privileged_host:clone(parent.privileged_host)},
  intervention:{commands:order.slice(),noise},prefix_actions:prefix.actions,
  initial_model_input:start,expert:{actions,positions:route,optimal_moves:actions.length},
  excluded_from_model:["parent","intervention","expert","provenance","episode_id","family_id","split","group_id","topology_group"],
  native_receipt:false};
}
function createHost(ep){
 const host=v13.createHost(clone(ep.parent));
 const prefix=runPrefix(host,ep.intervention.commands,ep.intervention.noise);
 if(sha(prefix.actions)!==sha(ep.prefix_actions)||sha(v13.observe(host))!==sha(ep.initial_model_input))
  throw Error("host reconstruction mismatch");
 return host;
}
const observe=h=>v13.observe(h),apply=(h,a)=>v13.apply(h,a);
function replayAndLabel(ep){
 const host=createHost(ep),rows=[],acts=ep.expert.actions;
 const chosen=new Set([0,1,2,3,Math.floor(acts.length/4),Math.floor(acts.length/2),
  Math.floor(acts.length*3/4),acts.length-2,acts.length-1]);
 let negCount=0;
 for(let t=0;t<acts.length;t++){
  const obs=observe(host),target=acts[t],route=v13.route(host.w,host.position,host.closed);
  if(!obs.legal_actions.includes(target)||!route||route.length-1!==acts.length-t)
   throw Error("expert move illegal or suboptimal");
  if(chosen.has(t)){
   const negatives=[],ties=[];
   for(const a of obs.legal_actions){if(a===target)continue;
    const after=v13.route(host.w,nextPos(host.position,a),host.closed);
    if(!after||after.length>route.length-1)negatives.push(a);
    else if(after.length===route.length-1)ties.push(a);
    else throw Error("route step contradiction");
   }
   negCount+=negatives.length;
   rows.push({schema:ep.schema,split:ep.split,task:ep.task,difficulty:ep.difficulty,
    episode_id:ep.episode_id,family_id:ep.family_id,group_id:ep.group_id,topology_group:ep.topology_group,
    time_index:t,model_input:obs,target,
    hard_negatives:{strictly_longer_legal:negatives,blocked_or_illegal:DIRS.map(d=>d[0]).filter(a=>!obs.legal_actions.includes(a))},
    controls:{last64_only:modelHistoryProjection(obs,64),last1_only:modelHistoryProjection(obs,1),
     event_bag:eventProjection(obs,"bag"),first_alpha:eventProjection(obs,"first"),
     last_alpha:eventProjection(obs,"last"),last2_alpha:eventProjection(obs,"last2"),
     first2_alpha:eventProjection(obs,"first2"),first3_alpha:eventProjection(obs,"first3"),
     last3_alpha:eventProjection(obs,"last3"),
     event_blind:eventProjection(obs,"no_events")},
    annotation:{remaining_shortest_steps:route.length-1,equally_optimal_other_actions:ties},
    excluded_model_fields:["split","task","difficulty","episode_id","family_id","group_id","topology_group",
     "time_index","target","hard_negatives","controls","annotation"],native_receipt:false});
  }
  if(!apply(host,target).ok)throw Error("host rejected expert");
 }
 if(!host.done||host.illegal_attempts!==0)throw Error("expert failed replay");
 return {rows,moves:acts.length,hardNegatives:negCount};
}
function ceiling(episodes,project){
 const bins=new Map();
 for(const ep of episodes){const key=sha(project(ep.initial_model_input));
  const a=ep.expert.actions[0],counts=bins.get(key)||{};counts[a]=(counts[a]||0)+1;bins.set(key,counts);}
 return {correct:[...bins.values()].reduce((n,b)=>n+Math.max(...Object.values(b)),0),
  denominator:episodes.length,classes:bins.size};
}
function checkFamilies(families){
 let count=0;const theoretical={};
 for(const [id,eps] of families){
  if(eps.length!==12)throw Error("missing permutation "+id);
  const targets=eps.map(ep=>ep.expert.actions[0]);
  if(targets.filter(a=>a==="MOVE_WEST").length!==6||targets.filter(a=>a==="MOVE_EAST").length!==6)
   throw Error("family target imbalance");
  const obs=eps.map(ep=>ep.initial_model_input);
  if(new Set(obs.map(o=>sha(modelHistoryProjection(o,64)))).size!==1)
   throw Error("last64 alias violation "+id);
  for(const mode of ["bag","no_events"]){
   if(new Set(obs.map(o=>sha(eventProjection(o,mode)))).size!==1)
    throw Error("event bag / blind alias violation "+id);
  }
  const stats={bag:ceiling(eps,o=>eventProjection(o,"bag")),
   first:ceiling(eps,o=>eventProjection(o,"first")),last:ceiling(eps,o=>eventProjection(o,"last")),
   first2:ceiling(eps,o=>eventProjection(o,"first2")),first3:ceiling(eps,o=>eventProjection(o,"first3")),
   last2:ceiling(eps,o=>eventProjection(o,"last2")),last3:ceiling(eps,o=>eventProjection(o,"last3"))};
  if(stats.bag.correct!==6||stats.bag.classes!==1)
   throw Error("bag cannot be predictive: "+id);
  for(const mode of ["first","first2","last","last2"])
   if(stats[mode].correct!==6||stats[mode].classes!==1)
    throw Error("controlled prefix/suffix leakage "+mode+": "+id);
  const split=eps[0].split;
  if(!theoretical[split])theoretical[split]=stats;
  else if(sha(theoretical[split])!==sha(stats))throw Error("unstable within-split control ceilings");
  count++;
 }
 return {families:count,controls:theoretical};
}
function generateDataset(styleSeeds=2){
 if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>8)throw Error("styleSeeds must be 1..8");
 const parent=v13.generateDataset(styleSeeds),episodes={},decisions={},counts={},families=new Map();
 const splitsByWorld=new Map(),splitsByTopology=new Map(),inputHashes=new Set(),viewHashes=new Map();
 let expertMoves=0,hardNegatives=0;
 for(const split of SPLITS){
  episodes[split]=[];decisions[split]=[];
  for(const src of parent.episodes[split].filter(e=>e.episode_id.includes("/toggle0/base0"))){
   const noise=split==="TRAIN"?1:split==="DEV"?2:3;
   for(let oi=0;oi<ORDERS_BY_SPLIT[split].length;oi++){
    const ep=makeEpisode(src,ORDERS_BY_SPLIT[split][oi],oi,noise),o=ep.initial_model_input;
    if(splitsByWorld.has(ep.group_id)&&splitsByWorld.get(ep.group_id)!==split)throw Error("world leakage");
    splitsByWorld.set(ep.group_id,split);
    if(splitsByTopology.has(ep.topology_group)&&splitsByTopology.get(ep.topology_group)!==split)
     throw Error("topology leakage");
    splitsByTopology.set(ep.topology_group,split);
    const sight=sha(o.local_material_7x7);
    if(viewHashes.has(sight)&&viewHashes.get(sight)!==ep.group_id)throw Error("duplicate world view across groups");
    viewHashes.set(sight,ep.group_id);
    if(!families.has(ep.family_id))families.set(ep.family_id,[]);families.get(ep.family_id).push(ep);
    const replay=replayAndLabel(ep);expertMoves+=replay.moves;hardNegatives+=replay.hardNegatives;
    for(const row of replay.rows){const key=sha(row.model_input);
     if(inputHashes.has(key))throw Error("duplicate complete model input");inputHashes.add(key);
    }
    episodes[split].push(ep);decisions[split].push(...replay.rows);
   }
  }
  counts[split]={episodes:episodes[split].length,decisions:decisions[split].length,
   world_groups:new Set(episodes[split].map(x=>x.group_id)).size,
   families:new Set(episodes[split].map(x=>x.family_id)).size,
   topology_groups:new Set(episodes[split].map(x=>x.topology_group)).size,
   alpha_receipts_per_episode:12,beta_decoys_per_alpha:split==="TRAIN"?1:split==="DEV"?2:3};
 }
 const proof=checkFamilies(families);
 return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-aliased-order-v17",version:"aliased-order-v17",
  source:"external v13 deterministic host; NOT native CyberSand ABI",parent_source_git_blob:PARENT_SOURCE_BLOB,
  counts,style_seeds:styleSeeds,noncommutative_families:proof.families,expert_moves:expertMoves,
  scripted_prefix_moves:342*Object.values(episodes).flat().length,
  strict_hard_negatives:hardNegatives,illegal_oracle_moves:0,protected_entries:0,
  first_action_control_ceiling_by_split:proof.controls,
  source_worlds:parent.summary.counts,split_policy:"world/topology-disjoint; 12 correlated permutations in one split",
  challenge:"Every 12-case family shares identical latest64, goal, legal menu, local terrain, full event bag, first-two and last-two ALPHA instructions; six opposed labels; TRAIN adjacent SETs, DEV gap=2, HELDOUT gap>=3; BETA decoys noncausal",
  usage:"TRAIN is optimizer-only; DEV tuning-only; HELDOUT sealed until fixed model selection",
  model_view:"Only observe(host) or row.model_input, never parent/intervention/expert/controls/metadata",
  limitations:["Scripted sensor-event prelude; only post-prefix rollout model-controllable",
   "One deterministic U-loop grammar, no native simulation or verified CyberSand ABI",
   "TRAIN/DEV/HELDOUT SET separation is deliberately distribution-shifted; split labels are not natural event frequencies",
   "Host environmental gate instructions are synthetic events, NOT model-writable actions",
   "A controlled last-two/first-two alias is exact; first-three/last-three controls can remain predictive, with split-specific ceilings recorded",
   "V13-derived world families recur across v16 and v17; never pool different-version splits without group-level re-splitting",
   "Correlated permutations and path rows; report world-grouped/paired held-out results, not iid row confidence",
   "No trained SLM results or material-paint acceptance"]}};
}
function writeDataset(dest,styleSeeds=2){
 if(fs.existsSync(dest))throw Error("refuse overwrite");const out=generateDataset(styleSeeds);
 fs.mkdirSync(dest,{recursive:true});const shards={};
 for(const split of SPLITS)for(const kind of ["episodes","decisions"]){
  const filename=`${split}_${kind}.jsonl`,body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
  fs.writeFileSync(path.join(dest,filename),body,{flag:"wx"});
  shards[filename]={rows:out[kind][split].length,bytes:Buffer.byteLength(body),
   sha256:crypto.createHash("sha256").update(body).digest("hex"),
   usage:split==="TRAIN"?"optimizer TRAIN only":split==="DEV"?"DEV selection only":"HELDOUT evaluation only"};
 }
 const manifest={...out.summary,generator:"aliased_order_v17.js",shards};
 fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});return manifest;
}
if(require.main===module){if(!process.argv[2])throw Error("Usage: node aliased_order_v17.js NEW_OUTPUT_DIR [styleSeeds=2]");
 console.log(JSON.stringify(writeDataset(process.argv[2],process.argv[3]===undefined?2:Number(process.argv[3])).counts));}
module.exports={permutations,ORDERS,applyGateCommand,closedFor,runPrefix,makeEpisode,createHost,observe,apply,
 ORDERS_BY_SPLIT,eventProjection,replayAndLabel,ceiling,checkFamilies,generateDataset,writeDataset};
