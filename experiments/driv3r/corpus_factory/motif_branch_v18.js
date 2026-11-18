/* DRIV3R Corpus Factory v18: synthetic closed-loop motif editing after ordered relay events.
 * DATA ONLY: independent fictional host. NOT the native CyberSand action/observation ABI.
 * Uses v17 order partitions (blob 014a25750f368f68b3397d0e413f7faca5ea4c67).
 * Usage: node motif_branch_v18.js NEW_OUTPUT_DIR [worldSeeds=2]
 */
"use strict";
const v17=require("./aliased_order_v17.js");
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
const SPLITS=["TRAIN","DEV","HELDOUT"];
const MOVES=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const PAINT=["SAND","METAL","WATER","STONE"],PKEY=p=>p.join(","),clone=x=>structuredClone(x);
const hash=x=>crypto.createHash("sha256").update(typeof x==="string"?x:JSON.stringify(x)).digest("hex");
const SRC="014a25750f368f68b3397d0e413f7faca5ea4c67";
function world(split,seed){
 const length=({TRAIN:8,DEV:10,HELDOUT:12})[split]+(seed%2),cx=6,jy=12;
 const w={split,seed,cx,jy,length,open:new Set(),protected:new Set(),paintable:new Set(),start:[cx,jy],exit:[cx+length+2,jy]};
 const add=(x,y)=>w.open.add(`${x},${y}`);
 for(let y=3;y<=jy+4;y++)add(cx,y);
 for(let x=1;x<=w.exit[0];x++)add(x,jy);
 for(let side of [-1,1]){
  const y=jy+side*2,bypass=y+side;
  for(let x=cx;x<=cx+length;x++)add(x,y);
  for(let x=cx+3;x<=cx+5;x++)add(x,bypass);
  add(cx+3,y);add(cx+5,y);
  const block=PKEY([cx+4,y]);w.protected.add(block);
  for(let dx of [2,3,5,6,7])w.paintable.add(PKEY([cx+dx,y]));
 }
 // At (cx+4,y) navigation must use the one-cell side bypass; blocked cells cannot be painted.
 for(const b of w.protected)if(!w.open.has(b))throw Error("protected missing from floor");
 return w;
}
function tile(w,paint,p){const k=PKEY(p);if(w.protected.has(k))return "P";
 if(w.open.has(k))return paint[k]?({SAND:"s",METAL:"m",WATER:"w",STONE:"t"})[paint[k]]:".";
 // Deterministic per-world background: different styles never enter playable cells.
 return ["#","r","#","o","#"][(parseInt(hash(`v18/${w.split}/${w.seed}/${k}`).slice(0,8),16))%5];
}
function local(w,paint,p){return Array.from({length:7},(_,r)=>Array.from({length:7},(_,c)=>tile(w,paint,[p[0]+c-3,p[1]+r-3])).join(""));}
function legal(w,paint,p){let choices=MOVES.filter(([_,dx,dy])=>w.open.has(PKEY([p[0]+dx,p[1]+dy]))&&!w.protected.has(PKEY([p[0]+dx,p[1]+dy]))).map(x=>x[0]);
 if(w.paintable.has(PKEY(p))){choices.push(...PAINT.map(x=>"PAINT_"+x));if(paint[PKEY(p)])choices.push("ERASE");}
 return choices;}
function shortest(w,from,to){const k=PKEY(from),q=[from],prev=new Map([[k,null]]),coords=new Map([[k,from]]);
 for(let i=0;i<q.length;i++){
  const p=q[i],a=PKEY(p);if(a===PKEY(to)){const out=[];let at=a;while(at!==null){out.push(coords.get(at));at=prev.get(at);}return out.reverse();}
  for(const [_,dx,dy] of MOVES){const np=[p[0]+dx,p[1]+dy],nk=PKEY(np);if(!prev.has(nk)&&w.open.has(nk)&&!w.protected.has(nk)){prev.set(nk,a);coords.set(nk,np);q.push(np);}}
 }return null;}
function prelude(w,order,noise){const positions=[w.start.slice()],events=[];let tick=0;
 for(let i=0;i<19;i++){
  for(let d=1;d<=9;d++)positions.push([w.cx,w.jy-d]);
  for(let d=8;d>=0;d--)positions.push([w.cx,w.jy-d]);
  if(i<12){tick++;
   events.push({step:positions.length-1,kind:"remote_instruction",line:"ALPHA",command:order[i]});
   const filler=["TOGGLE","SET_NORTH","SET_SOUTH","TOGGLE"];
   for(let j=0;j<noise;j++)events.push({step:positions.length-1,kind:"remote_instruction",line:"BETA",command:filler[(i*noise+j)%4],noncausal:true});
  }
 }
 if(positions.length!==343||PKEY(positions.at(-1))!==PKEY(w.start))throw Error("prelude length");
 return {executed_positions:positions,events,objective:"Follow the remote ordered relay, paint the five-cell alternating motif in its selected branch, then reach the eastern exit; protected cells must remain untouched"};
}
const resolved=order=>v17.closedFor(order)==="WEST"?"NORTH":"SOUTH";
const targetInk=(lane,i)=>PAINT[(lane==="NORTH"?0:1)+(i%2)];
function makeEpisode(w,order,orderIndex,noise){
 const lane=resolved(order),y=w.jy+(lane==="NORTH"?-2:2),paintTargets=Object.fromEntries([2,3,5,6,7].map((dx,i)=>[PKEY([w.cx+dx,y]),targetInk(lane,i)]));
 const history=prelude(w,order,noise),initial={schema:"driv3r/synthetic-authoring-observation-v18",position:w.start.slice(),exit:w.exit.slice(),objective:history.objective,
  rules:{alpha_start:"WEST",alpha_commands:"SET_WEST replaces state with WEST; SET_EAST replaces with EAST; TOGGLE swaps; process chronologically; BETA is ignored",
   lane_selection:"Final WEST -> NORTH branch; final EAST -> SOUTH branch",
   motif:"NORTH uses SAND,METAL alternating; SOUTH uses METAL,WATER alternating; paint each of five editable cells; no other cells may remain painted",
   material_actions:"PAINT_* overwrites current editable cell; ERASE clears; protected P cannot be entered or painted"},
  local_material_7x7:local(w,{},w.start),legal_actions:legal(w,{},w.start),history};
 const family=`v18/${w.split}/${w.length}/${w.seed}`;
 return {schema:"driv3r/corpus-factory-motif-branch-v18",split:w.split,episode_id:`${family}/order${orderIndex}`,family_id:family,group_id:family,
  topology_group:`v18/${w.split}/length${w.length}`,difficulty:"D11-old-ordered-relay-branch-and-protected-paint",
  provenance:{generator:"motif_branch_v18.js",synthetic_only:true,order_source_git_blob:SRC,style_seed:w.seed},
  initial_model_input:initial,privileged_host:{split:w.split,seed:w.seed,lane,target:paintTargets},
  scripted_environment_events:order,model_input_contract:"ONLY observe(host) or row.model_input; no private episode fields",native_receipt:false};
}
function createHost(ep){const w=world(ep.privileged_host.split,ep.privileged_host.seed),h={episode:ep,w,position:w.start.slice(),paint:{},trail:[w.start.slice()],receipts:[],invalid:0,steps:0,done:false};
 if(hash(observe(h))!==hash(ep.initial_model_input))throw Error("episode observation mismatch");return h;}
function observe(h){const base=h.episode.initial_model_input;
 return {...base,position:h.position.slice(),local_material_7x7:local(h.w,h.paint,h.position),legal_actions:legal(h.w,h.paint,h.position),
  history:{...base.history,executed_positions:base.history.executed_positions.concat(h.trail.slice(1)),events:base.history.events.concat(h.receipts)}};}
function finish(h){const target=h.episode.privileged_host.target;
 return PKEY(h.position)===PKEY(h.w.exit)&&Object.keys(h.paint).length===Object.keys(target).length&&Object.entries(target).every(([k,v])=>h.paint[k]===v);}
function apply(h,action){if(h.done)return {ok:false,reason:"finished"};const available=legal(h.w,h.paint,h.position);
 if(!available.includes(action)){h.invalid++;return {ok:false,reason:"illegal_action"};}
 const old=h.position.slice(),move=MOVES.find(x=>x[0]===action);
 if(move){h.position=[old[0]+move[1],old[1]+move[2]];h.trail.push(h.position.slice());}
 else {const k=PKEY(old);if(action==="ERASE")delete h.paint[k];else h.paint[k]=action.slice(6);
  h.receipts.push({step:343+h.steps,kind:"synthetic_paint_receipt",position:old,action,accepted:true});h.trail.push(old);}
 h.steps++;h.done=finish(h);return {ok:true,done:h.done};}
function pathActions(path){const out=[];for(let i=1;i<path.length;i++){
  const a=path[i-1],b=path[i],dir=MOVES.find(x=>a[0]+x[1]===b[0]&&a[1]+x[2]===b[1]);if(!dir)throw Error("non-cardinal route");out.push(dir[0]);}return out;}
function oracle(h){const ep=h.episode,lane=ep.privileged_host.lane,y=h.w.jy+(lane==="NORTH"?-2:2),acts=[];
 for(const dx of [2,3,5,6,7]){
  const end=[h.w.cx+dx,y],route=shortest(h.w,h.position,end);if(!route)throw Error("unreachable paint target");
  for(const a of pathActions(route)){if(!apply(h,a).ok)throw Error("oracle move refused");acts.push(a);}
  const paint="PAINT_"+ep.privileged_host.target[PKEY(end)];if(!apply(h,paint).ok)throw Error("oracle paint refused");acts.push(paint);
 }
 const route=shortest(h.w,h.position,h.w.exit);if(!route)throw Error("unreachable exit");
 for(const a of pathActions(route)){if(!apply(h,a).ok)throw Error("oracle move refused");acts.push(a);}
 if(!h.done||h.invalid)throw Error("oracle incomplete");return acts;}
function replay(ep){const h=createHost(ep),guide=oracle(createHost(ep)),rows=[];
 for(let i=0;i<guide.length;i++){
  const obs=observe(h),target=guide[i];if(!obs.legal_actions.includes(target))throw Error("illegal label");
  // Every step is a valid trainer decision; sampled first/relevant painting steps are separately tagged.
  rows.push({schema:ep.schema,split:ep.split,episode_id:ep.episode_id,group_id:ep.group_id,family_id:ep.family_id,topology_group:ep.topology_group,
   time_index:i,model_input:obs,target,hard_negatives:{other_legal:obs.legal_actions.filter(a=>a!==target),illegal:MOVES.map(x=>x[0]).filter(a=>!obs.legal_actions.includes(a))},
   phase:target.startsWith("PAINT_")?"paint":target.startsWith("MOVE_")?"navigate":"edit",native_receipt:false,
   excluded_model_fields:["split","episode_id","group_id","family_id","topology_group","time_index","target","hard_negatives","phase"]});
  if(!apply(h,target).ok)throw Error("oracle replay rejected");
 }
 if(!h.done||h.invalid||hash(h.paint)!==hash(ep.privileged_host.target))throw Error("oracle replay mismatch");
 return {rows,actions:guide.length,paint:5};}
function recent(obs,n=64){const hist=obs.history,L=hist.executed_positions.length;
 return {...obs,history:{...hist,executed_positions:hist.executed_positions.slice(-n),events:hist.events.filter(e=>e.step>=L-n)}};}
function eventBag(obs){return {...obs,history:{...obs.history,executed_positions:obs.history.executed_positions.slice(-64),
  events:obs.history.events.map(({step,...e})=>e).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)))}};}
function generateDataset(seeds=2){if(!Number.isInteger(seeds)||seeds<1||seeds>6)throw Error("seeds 1..6");
 const episodes={},decisions={},counts={},seenFull=new Set(),groups=new Map(),topologies=new Map(),familyMap=new Map();let expertActions=0,paintReceipts=0;
 for(const split of SPLITS){episodes[split]=[];decisions[split]=[];const noise={TRAIN:1,DEV:2,HELDOUT:3}[split];
  for(let seed=0;seed<seeds+(split==="TRAIN"?2:0);seed++){
   const w=world(split,seed);for(let oi=0;oi<v17.ORDERS_BY_SPLIT[split].length;oi++){
    const ep=makeEpisode(w,v17.ORDERS_BY_SPLIT[split][oi],oi,noise),replayed=replay(ep);
    if(groups.has(ep.group_id)&&groups.get(ep.group_id)!==split)throw Error("world leakage");groups.set(ep.group_id,split);
    if(topologies.has(ep.topology_group)&&topologies.get(ep.topology_group)!==split)throw Error("topology leakage");topologies.set(ep.topology_group,split);
    const key=hash(ep.initial_model_input);if(seenFull.has(key))throw Error("duplicate full input");seenFull.add(key);
    if(!familyMap.has(ep.family_id))familyMap.set(ep.family_id,[]);familyMap.get(ep.family_id).push(ep);
    episodes[split].push(ep);decisions[split].push(...replayed.rows);expertActions+=replayed.actions;paintReceipts+=replayed.paint;
   }
  }
  counts[split]={episodes:episodes[split].length,decisions:decisions[split].length,
   families:new Set(episodes[split].map(x=>x.family_id)).size,topology_groups:new Set(episodes[split].map(x=>x.topology_group)).size};
 }
 for(const [id,eps] of familyMap){if(eps.length!==12)throw Error("wrong family size "+id);
  for(const key of [x=>hash(recent(x.initial_model_input,64)),x=>hash(eventBag(x.initial_model_input))])
   if(new Set(eps.map(key)).size!==1)throw Error("aliased projection leakage "+id);
  const labels=eps.map(x=>oracleFirst(x));if(labels.filter(x=>x==="MOVE_NORTH").length!==6||labels.filter(x=>x==="MOVE_SOUTH").length!==6)
   throw Error("family imbalance");
 }
 const controls={};
 for(const split of SPLITS){const eps=episodes[split];controls[split]={};
  for(const mode of ["bag","first2","last2","first3","last3"]){
   const bins=new Map();for(const ep of eps){const key=hash(v17.eventProjection(ep.initial_model_input,mode));
    const counts=bins.get(key)||{};const a=oracleFirst(ep);counts[a]=(counts[a]||0)+1;bins.set(key,counts);}
   controls[split][mode]={correct:[...bins.values()].reduce((n,x)=>n+Math.max(...Object.values(x)),0),denominator:eps.length,classes:bins.size};
  }
 }
 return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-motif-branch-v18",version:"motif-branch-v18",source:"independent synthetic authoring host; NOT native CyberSand ABI",
  order_source_git_blob:SRC,world_seeds:seeds,counts,families:familyMap.size,expert_actions:expertActions,accepted_oracle_paint_receipts:paintReceipts,
  protected_entries:0,illegal_oracle_actions:0,initial_first_action_last64_bag_ceiling:"50% by 12-way family",first_action_control_ceiling_by_split:controls,
  usage:"TRAIN optimizer only, DEV selection only, HELDOUT sealed until selection fixed",
  limitations:["Synthetic brush/navigation semantics unverified against native game", "Scripted environmental history, only editing phase policy-controllable",
  "One small corridor-and-bypass grammar", "World variants correlated within topology; no cross-version independence from v17 order sequences",
  "No real SLM trained or evaluated", "Wrong paint can be corrected with ERASE; host reveals no correctness receipt"]}};}
function oracleFirst(ep){const lane=ep.privileged_host.lane;return lane==="NORTH"?"MOVE_NORTH":"MOVE_SOUTH";}
function writeDataset(dest,seeds=2){if(fs.existsSync(dest))throw Error("refuse overwrite");const out=generateDataset(seeds);fs.mkdirSync(dest,{recursive:true});const shards={};
 for(const split of SPLITS)for(const kind of ["episodes","decisions"]){const name=`${split}_${kind}.jsonl`,body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
  fs.writeFileSync(path.join(dest,name),body,{flag:"wx"});shards[name]={rows:out[kind][split].length,bytes:Buffer.byteLength(body),sha256:hash(body),usage:split};}
 const manifest={...out.summary,generator:"motif_branch_v18.js",shards};fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});return manifest;}
if(require.main===module){if(!process.argv[2])throw Error("Usage: node motif_branch_v18.js NEW_OUTPUT_DIR [worldSeeds=2]");console.log(JSON.stringify(writeDataset(process.argv[2],process.argv[3]===undefined?2:Number(process.argv[3])).counts));}
module.exports={world,tile,local,legal,shortest,prelude,resolved,makeEpisode,createHost,observe,apply,oracle,replay,recent,eventBag,oracleFirst,generateDataset,writeDataset};
