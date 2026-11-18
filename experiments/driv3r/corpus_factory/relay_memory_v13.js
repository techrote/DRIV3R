/* DRIV3R Corpus Factory v13: two-event relay/gate memory, external synthetic host.
 * DATA ONLY; not native CyberSand. Node 18+:
 * node relay_memory_v13.js NEW_OUTPUT_DIR [styleSeeds=4]
 * Never pass privileged_host, expert, controls, or row metadata to a model.
 */
"use strict";
const DIRS=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const SPECS={
 TRAIN:{grammar:"plain",spans:[9,12,15,18],heights:[20,25,30,35]},
 DEV:{grammar:"spur",spans:[21,24],heights:[22,28]},
 HELDOUT:{grammar:"double-spur",spans:[27,30],heights:[24,29,34]}
};
const posKey=p=>p.join(",");
function hash32(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0;}
function worldFor(grammar,span,height,seed){
 const cx=span+9,jy=11,by=jy+height,w=2*span+19,h=by+7,open=new Set();
 const add=(x,y)=>open.add(x+","+y);
 for(let y=0;y<=jy;y++)add(cx,y);
 for(let side of [-1,1]){
  for(let d=0;d<=span;d++){add(cx+side*d,jy);add(cx+side*d,by);}
  for(let y=jy;y<=by;y++)add(cx+side*span,y);
  let offsets=grammar==="double-spur"?[Math.floor(height/3),Math.floor(2*height/3)]:grammar==="spur"?[Math.floor(height/2)]:[];
  for(let offset of offsets)for(let d=1;d<=3;d++)add(cx+side*(span+d),jy+offset);
 }
 for(let y=by;y<=by+3;y++)add(cx,y);
 const style=hash32(`v13/style/${grammar}/${span}/${height}/${seed}`),wiring=hash32(`v13/wiring/${grammar}/${span}/${height}/${seed}`)%2;
 return {grammar,span,height,seed,cx,jy,by,w,h,open,style,wiring,start:[cx,jy],goal:[cx,by+3],
  topology:`${grammar}/${span}x${height}`,world_id:`${grammar}/${span}x${height}/${seed}`};
}
function tile(w,p,closed){
 const k=posKey(p),[x,y]=p;
 if(x<0||y<0||x>=w.w||y>=w.h)return "#";
 if(w.open.has(k)){
  if(y===w.by-3&&Math.abs(x-w.cx)===w.span)return (x<w.cx?"WEST":"EAST")===closed?"X":"G";
  return x<w.cx?"s":x>w.cx?"m":".";
 }
 if(hash32("v13/protect/"+w.style+"/"+k)%17===0)return "P";
 return ["#","r","s","m","w","#","#"][hash32("v13/tile/"+w.style+"/"+k)%7];
}
function legal(w,p,closed){
 return DIRS.filter(([_,dx,dy])=>{const q=[p[0]+dx,p[1]+dy];return w.open.has(posKey(q))&&tile(w,q,closed)!=="X";}).map(d=>d[0]);
}
function view7(w,p,closed){return Array.from({length:7},(_,r)=>Array.from({length:7},(_,c)=>tile(w,[p[0]+c-3,p[1]+r-3],closed)).join(""));}
function move(p,action){const d=DIRS.find(x=>x[0]===action);if(!d)throw Error("unknown action");return [p[0]+d[1],p[1]+d[2]];}
function route(w,from,closed){
 const start=posKey(from),goal=posKey(w.goal),q=[from],parent=new Map([[start,null]]),coords=new Map([[start,from]]);
 for(let i=0;i<q.length;i++){
  const p=q[i],k=posKey(p);if(k===goal){const path=[];let cursor=k;while(cursor!==null){path.push(coords.get(cursor));cursor=parent.get(cursor);}return path.reverse();}
  for(const action of legal(w,p,closed)){const t=move(p,action),tk=posKey(t);if(!parent.has(tk)){parent.set(tk,k);coords.set(tk,t);q.push(t);}}
 }
 return null;
}
function shortHistory(obs,n=64){const h=obs.history,L=h.executed_positions.length;return {...obs,history:{...h,executed_positions:h.executed_positions.slice(-n),events:h.events.filter(e=>e.step>=L-n)}};}
function ancientPrelude(w,base,toggle){
 const trace=[w.start.slice()];
 for(let repeat=0;repeat<12;repeat++){
  for(let d=1;d<=9;d++)trace.push([w.cx,w.jy-d]);
  for(let d=8;d>=0;d--)trace.push([w.cx,w.jy-d]);
 }
 const n=trace.length;
 const events=[
  {step:2,kind:"remote_relay_boot",line:"ALPHA",binary_state:base,west_is_low_channel:w.wiring===0},
  {step:37,kind:"remote_relay_toggle",line:"ALPHA",toggle_count:toggle},
  {step:n-49,kind:"ambient_sensor",payload:"unchanged"},
  {step:n-8,kind:"ambient_sensor",payload:"unchanged"}
 ];
 if(n-37<64||n<=150)throw Error("prelude too short");
 return {executed_positions:trace,events,objective:"reach the south merge while avoiding the remote shutter; relay ALPHA may invert channel routing"};
}
function makeEpisode(w,split,base,toggle){
 const active=base^toggle^w.wiring,closed=active===0?"WEST":"EAST",history=ancientPrelude(w,base,toggle);
 const path=route(w,w.start,closed);if(!path||path.length<3)throw Error("oracle unreachable");
 const actions=path.slice(1).map((p,i)=>{
  const a=path[i],d=DIRS.find(x=>a[0]+x[1]===p[0]&&a[1]+x[2]===p[1]);if(!d)throw Error("noncardinal route");return d[0];
 });
 const initial={schema:"driv3r/synthetic-observation-v13",position:w.start.slice(),goal:w.goal.slice(),
  objective:history.objective,local_material_7x7:view7(w,w.start,closed),legal_actions:legal(w,w.start,closed),history};
 return {schema:"driv3r/corpus-factory-v13",episode_id:`v13/${w.world_id}/toggle${toggle}/base${base}`,
  pair_id:`v13/${w.world_id}/toggle${toggle}`,group_id:`v13/${w.world_id}`,topology_group:w.topology,
  split,task:"composed_remote_relay_shutter",difficulty:"D7-two-remote-events-64-alias",native_receipt:false,
  provenance:{generator:"relay_memory_v13.js",world_id:w.world_id,grammar:w.grammar,
   span:w.span,height:w.height,style_seed:w.seed,style_hash32:w.style.toString(16),
   synthetic_prelude_steps:history.executed_positions.length-1},
  initial_model_input:initial,privileged_host:{grammar:w.grammar,span:w.span,height:w.height,seed:w.seed,
   closed_gate_side:closed},expert:{actions,positions:path,optimal_moves:actions.length,
   proof:"synthetic BFS shortest path under resolved remote relay/shutter state"},
  model_input_contract:"ONLY observe(host) or initial_model_input; never privileged_host/expert/metadata"};
}
function createHost(ep){
 const m=ep.privileged_host,w=worldFor(m.grammar,m.span,m.height,m.seed);
 return {episode:ep,w,closed:m.closed_gate_side,position:ep.initial_model_input.position.slice(),
  travelled:[ep.initial_model_input.position.slice()],applied:0,illegal_attempts:0,done:false};
}
function observe(host){
 const first=host.episode.initial_model_input,h=first.history;
 return {...first,position:host.position.slice(),local_material_7x7:view7(host.w,host.position,host.closed),
  legal_actions:legal(host.w,host.position,host.closed),history:{...h,executed_positions:h.executed_positions.concat(host.travelled.slice(1))}};
}
function apply(host,action){
 if(host.done)return {ok:false,reason:"already_finished"};
 if(!observe(host).legal_actions.includes(action)){host.illegal_attempts++;return {ok:false,reason:"host_illegal_action"};}
 host.position=move(host.position,action);host.travelled.push(host.position.slice());host.applied++;
 if(tile(host.w,host.position,host.closed)==="P")throw Error("protected entry");
 host.done=posKey(host.position)===posKey(host.w.goal);
 return {ok:true,done:host.done};
}
function samples(ep){
 const host=createHost(ep),length=ep.expert.actions.length,rows=[];
 const chosen=new Set([0,1,2,3,Math.floor(length/4),Math.floor(length/2),Math.floor(3*length/4),length-2,length-1]);
 for(let t=0;t<length;t++){
  const obs=observe(host),action=ep.expert.actions[t];
  if(chosen.has(t)){
   const best=route(host.w,host.position,host.closed);if(!best)throw Error("unreachable current");
   const negatives=obs.legal_actions.filter(a=>{
    if(a===action)return false;
    const rest=route(host.w,move(host.position,a),host.closed);
    return !rest||1+rest.length-1>best.length-1;
   });
   rows.push({schema:ep.schema,split:ep.split,task:ep.task,episode_id:ep.episode_id,
    pair_id:ep.pair_id,group_id:ep.group_id,topology_group:ep.topology_group,
    time_index:t,model_input:obs,target:action,
    hard_negatives:{provably_longer_legal:negatives},
    controls:{last64_only:shortHistory(obs),last1_only:shortHistory(obs,1),
     zero_history:{...obs,history:{objective:obs.objective}}},
    excluded_model_fields:["split","episode_id","pair_id","group_id","topology_group","target",
     "hard_negatives","controls","time_index"],native_receipt:false});
  }
  if(!apply(host,action).ok)throw Error("oracle rejected");
 }
 if(!host.done||host.applied!==ep.expert.optimal_moves||host.illegal_attempts!==0)throw Error("expert failed");
 return {rows,applied:host.applied};
}
function generateDataset(styleSeeds=4){
 if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>24)throw Error("styleSeeds 1..24");
 const episodes={TRAIN:[],DEV:[],HELDOUT:[]},decisions={TRAIN:[],DEV:[],HELDOUT:[]},counts={},topologies=new Map(),pairChecks=new Map(),viewSignatures=[],seenIds=new Set();
 let applied=0,rejected_near_duplicates=0;
 for(const [split,spec] of Object.entries(SPECS))for(const span of spec.spans)for(const height of spec.heights){
  const topology=`${spec.grammar}/${span}x${height}`;
  if(topologies.has(topology))throw Error("topology reused");topologies.set(topology,split);
  for(let seed=0;seed<styleSeeds;seed++){
   const w=worldFor(spec.grammar,span,height,seed),sig=view7(w,w.start,"WEST").join("");
   if(viewSignatures.some(old=>{let d=0;for(let i=0;i<sig.length;i++)if(sig[i]!==old[i]&&++d>4)return false;return d<=4;})){
    rejected_near_duplicates++;continue;
   }
   viewSignatures.push(sig);
   for(let toggle=0;toggle<=1;toggle++)for(let base=0;base<=1;base++){
    const ep=makeEpisode(w,split,base,toggle);if(seenIds.has(ep.episode_id))throw Error("duplicate episode");seenIds.add(ep.episode_id);
    const host=createHost(ep),initial=observe(host);
    if(JSON.stringify(initial)!==JSON.stringify(ep.initial_model_input))throw Error("inconsistent initial observation");
    const initialAction=ep.expert.actions[0],short=JSON.stringify(shortHistory(initial));
    const other=pairChecks.get(ep.pair_id);
    if(other){if(other.short!==short||other.action===initialAction)throw Error("paired alias/label failed");}
    else pairChecks.set(ep.pair_id,{short,action:initialAction});
    if(!initial.legal_actions.includes(initialAction))throw Error("illegal expert initial");
    const result=samples(ep);applied+=result.applied;episodes[split].push(ep);decisions[split].push(...result.rows);
   }
  }
 }
 for(const split of Object.keys(SPECS))counts[split]={episodes:episodes[split].length,
  pairs:new Set(episodes[split].map(x=>x.pair_id)).size,world_groups:new Set(episodes[split].map(x=>x.group_id)).size,
  decisions:decisions[split].length};
 const total=Object.values(episodes).flat().length;
 if(pairChecks.size!==total/2||!total)throw Error("pair count");
 return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-v13",version:"relay-memory-v13",
  source:"external deterministic synthetic host; NOT native CyberSand ABI",native_receipt:false,
  style_seeds_requested:styleSeeds,counts,applied_oracle_moves:applied,protected_entries:0,
  counterfactual_pairs:pairChecks.size,rejected_near_duplicate_worlds:rejected_near_duplicates,
  split_policy:"topology/geometry disjoint; siblings and world styles stay within each partition",
  causal_challenge:"two remote relay events compose with world wiring; identical last64 and local observation, opposite shortest route",
  limitations:["Expert paths are synthetic BFS labels, NOT trained model results",
   "Prelude is scripted synthetic history; model controls only after junction",
   "No native CyberSand action/observation ABI or brush acceptance",
   "Downstream decisions from each episode correlated; evaluate by world group",
   "HELDOUT sealed: never use for optimisation or checkpoint selection"]}};
}
function main(){
 const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
 const dest=process.argv[2],n=process.argv[3]===undefined?4:Number(process.argv[3]);
 if(!dest||!Number.isInteger(n)||n<1||n>24)throw Error("Usage: node relay_memory_v13.js NEW_OUTPUT_DIR [styleSeeds 1..24]");
 if(fs.existsSync(dest))throw Error("refuse to overwrite existing destination");
 const out=generateDataset(n);fs.mkdirSync(dest,{recursive:true});const shards={};
 for(const split of ["TRAIN","DEV","HELDOUT"])for(const kind of ["episodes","decisions"]){
  const file=`${split}_${kind}.jsonl`,body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
  fs.writeFileSync(path.join(dest,file),body,{flag:"wx"});
  shards[file]={rows:out[kind][split].length,bytes:Buffer.byteLength(body),
   sha256:crypto.createHash("sha256").update(body).digest("hex"),
   usage:split==="TRAIN"?"TRAIN only":"evaluation only; NEVER optimize"};
 }
 const manifest={...out.summary,generator:"relay_memory_v13.js",shards,
  model_input_contract:"Only observe(host) or row.model_input; never episode.expert/privileged_host/metadata/controls"};
 fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
 console.log(JSON.stringify({output:dest,counts:out.summary.counts,pairs:out.summary.counterfactual_pairs,
  applied_moves:out.summary.applied_oracle_moves,shards}));
}
if(require.main===module)main();
module.exports={worldFor,tile,legal,view7,route,makeEpisode,createHost,observe,apply,shortHistory,generateDataset};
