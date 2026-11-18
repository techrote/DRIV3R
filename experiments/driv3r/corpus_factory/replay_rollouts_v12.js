/* DRIV3R Corpus Factory v12: source-pinned, external synthetic-host
 * sequential replay and group-held-out decision examples. DATA ONLY.
 * node replay_rollouts_v12.js NEW_OUTPUT_DIR [styleSeeds=2]
 * Synthetic != canonical native CyberSand ABI; no model weight training.
 */
"use strict";
const {generateDataset:baseV11}=require("./stateful_challenge_v11.js");
const DIRECTIONS=[["MOVE_NORTH",0,-1],["MOVE_EAST",1,0],["MOVE_SOUTH",0,1],["MOVE_WEST",-1,0]];
const key=p=>p[0]+","+p[1];
function h32(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function makeWorld(meta){
  const {span,height,grammar,style_hash32}=meta,cx=span+14,hy=19,by=hy+height;
  const w=2*span+29,h=by+9,open=new Set(),add=(x,y)=>open.add(x+","+y);
  for(let y=0;y<=hy;y++)add(cx,y);
  for(const side of [-1,1]){
    for(let d=0;d<=span;d++){add(cx+side*d,hy);add(cx+side*d,by)}
    for(let y=hy;y<=by;y++)add(cx+side*span,y);
    if(grammar!=="plain"){
      for(const v of grammar==="double-spur"?[Math.floor(height/3),Math.floor(2*height/3)]:[Math.floor(height/2)]){
        for(let d=1;d<=3;d++)add(cx+side*(span+d),hy+v);
      }
    }
  }
  for(let y=by;y<=by+4;y++)add(cx,y);
  return {cx,hy,by,w,h,open,style:parseInt(style_hash32,16)};
}
function tile(w,p){
  const [x,y]=p,k=key(p);
  if(x<0||y<0||x>=w.w||y>=w.h)return "#";
  if(!w.open.has(k)&&h32("v11-protect/"+w.style+"/"+k)%13===0)return "P";
  if(w.open.has(k))return x<w.cx?"s":x>w.cx?"m":".";
  return ["#","s","m","r","#","#","w"][h32("v11-mat/"+w.style+"/"+k)%7];
}
function window7(w,p){
  return Array.from({length:7},(_,row)=>
    Array.from({length:7},(_,col)=>tile(w,[p[0]+col-3,p[1]+row-3])).join(""));
}
function legal(w,p,gate){
  return DIRECTIONS.filter(([_,dx,dy])=>{
    const q=[p[0]+dx,p[1]+dy];
    return w.open.has(key(q))&&(!gate||key(q)!==key(gate));
  }).map(([name])=>name);
}
function actionFor(a,b){
  const d=DIRECTIONS.find(([_,dx,dy])=>b[0]-a[0]===dx&&b[1]-a[1]===dy);
  if(!d)throw Error("non-cardinal oracle path");return d[0];
}
function cost(w,p,law){
  if(!law||p[0]===w.cx)return 1;
  return (p[0]<w.cx)===(law==="DRY")?1:7;
}
function makeEpisode(row){
  if(!["delayed_shutter","delayed_material_law"].includes(row.task))throw Error("navigation only");
  const w=makeWorld(row.provenance),oracle=row.evidence.oracle;
  const path=oracle.path.map(p=>p.slice()),gate=row.task==="delayed_shutter"?oracle.closed_gate:null;
  const law=row.task==="delayed_material_law"?oracle.law:null;
  if(path.length<2||key(path[0])!==key(row.model_input.position)||
    key(path[path.length-1])!==key(row.model_input.goal))throw Error("invalid oracle endpoints");
  let total=0;
  for(let i=1;i<path.length;i++){
    if(!legal(w,path[i-1],gate).includes(actionFor(path[i-1],path[i])))
      throw Error("oracle violated host legality");
    if(tile(w,path[i])==="P")throw Error("protected entry");
    total+=cost(w,path[i],law);
  }
  if(total!==oracle.cost)throw Error("v11 oracle cost disagrees with replay: "+row.scenario_id);
  if(JSON.stringify(window7(w,path[0]))!==JSON.stringify(row.model_input.local_material_7x7))
    throw Error("v11 material mismatch: "+row.scenario_id);
  if(JSON.stringify(legal(w,path[0],gate))!==JSON.stringify(row.model_input.legal_actions))
    throw Error("v11 legal menu mismatch");
  return {
    schema:"driv3r/corpus-factory-replay-v12",episode_id:row.scenario_id,
    split:row.split,task:row.task,pair_id:row.pair_id,group_id:row.group_id,
    topology_group:row.topology_group,difficulty:row.difficulty,
    provenance:{base_version:"v11",generator:"replay_rollouts_v12.js",
      parent_generator:"stateful_challenge_v11.js",...row.provenance},
    initial_model_input:row.model_input,
    privileged_host:{world:row.provenance,closed_gate:gate,material_law:law},
    expert:{positions:path,actions:path.slice(1).map((p,i)=>actionFor(path[i],p)),
      total_traversal_cost:total,optimality:"v11 weighted shortest-path oracle"},
    native_receipt:false,model_input_only:"initial_model_input or observe(host); NEVER privileged_host/expert"
  };
}
function createHost(episode){
  const initial=episode.initial_model_input,w=makeWorld(episode.privileged_host.world);
  return {episode,w,position:initial.position.slice(),travelled:[initial.position.slice()],
    applied:0,spent:0,done:false,illegal_attempts:0};
}
function observe(host){
  const {episode,w,position,travelled}=host,first=episode.initial_model_input;
  const history={...first.history,executed_positions:
    first.history.executed_positions.concat(travelled.slice(1))};
  return {...first,position:position.slice(),local_material_7x7:window7(w,position),
    legal_actions:legal(w,position,episode.privileged_host.closed_gate),history};
}
function short32(observation){
  const full=observation.history,n=full.executed_positions.length;
  return {...observation,history:{...full,executed_positions:full.executed_positions.slice(-32),
    history_events:(full.history_events||[]).filter(e=>e.step>=n-32)}};
}
function apply(host,action){
  if(host.done)return {ok:false,reason:"already_finished"};
  if(!observe(host).legal_actions.includes(action)){
    host.illegal_attempts++;return {ok:false,reason:"host_illegal_action"};
  }
  const d=DIRECTIONS.find(v=>v[0]===action),next=[host.position[0]+d[1],host.position[1]+d[2]];
  if(tile(host.w,next)==="P")throw Error("protected cell movement");
  host.position=next;host.travelled.push(next);host.applied++;
  host.spent+=cost(host.w,next,host.episode.privileged_host.material_law);
  host.done=key(next)===key(host.episode.initial_model_input.goal);
  return {ok:true,done:host.done};
}
function generateRollouts(styleSeeds=2){
  if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>16)
    throw Error("styleSeeds must be 1..16");
  const source=baseV11(styleSeeds),episodes={TRAIN:[],DEV:[],HELDOUT:[]},
    decisions={TRAIN:[],DEV:[],HELDOUT:[]},counts={};
  const pairStarts=new Map(),topologySplits=new Map();let moves=0;
  for(const [split,rows] of Object.entries(source.shards)){
    for(const row of rows){
      if(row.task==="periodic_motif")continue; // v11 motif stays in its separately validated painting corpus
      const ep=makeEpisode(row),host=createHost(ep),length=ep.expert.actions.length;
      const sampleSteps=new Set([0,1,2,3,Math.floor(length/3),Math.floor(2*length/3),length-1]);
      for(let t=0;t<length;t++){
        const obs=observe(host),action=ep.expert.actions[t];
        if(t===0){
          const identity=JSON.stringify(short32(obs)),pair=pairStarts.get(ep.pair_id);
          if(pair){
            if(pair.short!==identity||pair.action===action)throw Error("paired history ablation failed");
          }else pairStarts.set(ep.pair_id,{short:identity,action});
        }
        if(sampleSteps.has(t)){
          decisions[split].push({schema:ep.schema,split,episode_id:ep.episode_id,
            group_id:ep.group_id,topology_group:ep.topology_group,pair_id:ep.pair_id,
            time_index:t,task:ep.task,difficulty:ep.difficulty,model_input:obs,target:action,
            controls:{last32_only:short32(obs),
              zero_history:{...obs,history:{objective:obs.objective}}},
            hard_negatives:{legal_but_suboptimal:obs.legal_actions.filter(x=>x!==action)},
            provenance:ep.provenance,native_receipt:false,
            excluded_model_fields:["split","episode_id","group_id","topology_group","pair_id",
              "time_index","target","controls","hard_negatives","provenance"]});
        }
        if(!apply(host,action).ok)throw Error("expert action rejected");
        moves++;
      }
      if(!host.done||host.spent!==ep.expert.total_traversal_cost||
        host.illegal_attempts!==0)throw Error("expert replay did not arrive optimally");
      episodes[split].push(ep);
      if(topologySplits.has(ep.topology_group)&&topologySplits.get(ep.topology_group)!==split)
        throw Error("topology split leakage");
      topologySplits.set(ep.topology_group,split);
    }
    counts[split]={episodes:episodes[split].length,
      pairs:new Set(episodes[split].map(e=>e.pair_id)).size,
      world_groups:new Set(episodes[split].map(e=>e.group_id)).size,
      decisions:decisions[split].length,
      mean_expert_moves:episodes[split].reduce((s,e)=>s+e.expert.actions.length,0)/episodes[split].length};
  }
  if(pairStarts.size!==Object.values(episodes).flat().length/2)throw Error("missing counterfactual sibling");
  return {episodes,decisions,summary:{schema:"driv3r/corpus-factory-replay-v12",
    source:"external synthetic host; derived from pinned v11 source, NOT native CyberSand",
    source_v11_blob:"be98237e63992da967b9a4ce309c549c11e67285",
    style_seeds_per_topology:styleSeeds,counts,applied_oracle_moves:moves,protected_entries:0,
    split_policy:"inherit v11 world-group and topology-disjoint TRAIN/DEV/HELDOUT",
    baseline:"v11 oracle-labelled only one initial action; v12 replays all later decisions",
    limitations:["Scripted prelude and oracle policy; no trained model closed-loop results",
      "Only navigation tasks (shutter, material law); v11 painting motif remains separate",
      "Synthetic action/observation grammar not native CyberSand ABI",
      "No model tuning on HELDOUT; privileged_host/expert must be inaccessible to model"]}};
}
function main(){
  const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
  const dest=process.argv[2],n=process.argv[3]===undefined?2:Number(process.argv[3]);
  if(!dest||!Number.isInteger(n)||n<1||n>16)throw Error("Usage: node replay_rollouts_v12.js NEW_OUTPUT_DIR [styleSeeds 1..16]");
  if(fs.existsSync(dest))throw Error("refuse to overwrite destination");
  const out=generateRollouts(n);fs.mkdirSync(dest,{recursive:true});
  const shards={};
  for(const split of ["TRAIN","DEV","HELDOUT"]){
    for(const kind of ["episodes","decisions"]){
      const filename=split+"_"+kind+".jsonl",body=out[kind][split].map(x=>JSON.stringify(x)).join("\n")+"\n";
      fs.writeFileSync(path.join(dest,filename),body,{flag:"wx"});
      shards[filename]={rows:out[kind][split].length,bytes:Buffer.byteLength(body),
        sha256:crypto.createHash("sha256").update(body).digest("hex"),
        usage:split==="TRAIN"?"TRAIN only":"evaluation-only; do not optimise"};
    }
  }
  const manifest={...out.summary,version:"v12",generator:"replay_rollouts_v12.js",shards,
    source_tracking:"Pin source Git blob in lane-local LATEST_HANDOFF.json"};
  fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
  process.stdout.write(JSON.stringify({output:dest,counts:out.summary.counts,shards})+"\n");
}
if(require.main===module)main();
module.exports={generateRollouts,createHost,observe,apply,makeWorld,window7,short32};
