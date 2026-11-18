/* DRIV3R Corpus Factory v9: cycle counterfactual challenge. Node.js 18+.
 External SYNTHETIC host only, NOT native CyberSand actions/receipts.
 The actions are host-local cardinal navigation primitives, NOT a game ABI replacement.
 Run: node cycle_memory_v9.js /new/output/path [seeds_per_topology=16].
 Output files TRAIN.jsonl, DEV.jsonl, HELDOUT.jsonl and manifest.json.
 DEV/HELDOUT are for evaluation ONLY; neither may enter optimizers.
 No model is fitted here. Never commit large generated shards/checkpoints to Git. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
function rng(seed){let x=(seed^0x9e3779b9)>>>0;return ()=>((x=Math.imul(x,1664525)+1013904223>>>0)/4294967296);}

function key(p){return p[0]+","+p[1];}

function fnv(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;}

function geometry(span,height){
 if(span<10||height<14)throw Error("too small for long-memory controls");
 let cx=span+5,hub=[cx,6],merge=[cx,height+6],open=new Set();
 function add(x,y){open.add(x+","+y);}
 for(let y=6;y<=height+6;y++)add(cx,y);
 for(const side of [-1,1]){for(let x=cx;x!==cx+side*(span+1);x+=side)add(x,6);
  for(let y=6;y<=height+6;y++)add(cx+side*span,y);
  for(let x=cx;x!==cx+side*(span+1);x+=side)add(x,height+6);
 }
 return {span,height,cx,hub,merge,open,topology:"two_cycle_"+span+"_"+height};
}

function world(span,height,seed){
 const g=geometry(span,height),r=rng(seed*31+span*101+height*17),blocked=new Set();
 const width=2*span+13,hh=height+13;
 while(blocked.size<32){let x=Math.floor(r()*width),y=Math.floor(r()*hh),k=x+","+y;if(!g.open.has(k)&&!(Math.abs(x-g.cx)<=4&&Math.abs(y-6)<=4))blocked.add(k);}
 const style=fnv("style:"+seed+":"+span+":"+height),law=(seed%2?"dry":"wet");
 return {...g,seed,blocked,width,hh,style,law};
}

function material(w,x,y){
 const k=x+","+y;
 if(w.blocked.has(k))return "P";
 if(w.open.has(k))return ".";
 let v=fnv(w.style+":"+x+":"+y)%7;return ["s","s","m","m","r","#","w"][v];
}

function localGrid(w,p,radius=3){
 let out=[];for(let dy=-radius;dy<=radius;dy++){let s="";for(let dx=-radius;dx<=radius;dx++)s+=material(w,p[0]+dx,p[1]+dy);out.push(s);}
 return out;
}

function neighbors(w,p){
 const ds=[["N",0,-1],["E",1,0],["S",0,1],["W",-1,0]];
 return ds.filter(z=>w.open.has((p[0]+z[1])+","+(p[1]+z[2]))&&!w.blocked.has((p[0]+z[1])+","+(p[1]+z[2]))).map(z=>z[0]);
}

function lobeTrace(w,side){
 let x=w.cx,y=6,path=[[x,y]],dx=side==="W"?-1:1;
 for(let i=0;i<w.span;i++){x+=dx;path.push([x,y]);}
 for(let i=0;i<w.height;i++){y++;path.push([x,y]);}
 for(let i=0;i<w.span;i++){x-=dx;path.push([x,y]);}
 for(let i=0;i<w.height;i++){y--;path.push([x,y]);}
 return path;
}

function enterTrace(w){
 let a=[];for(let y=w.merge[1];y>=6;y--)a.push([w.cx,y]);return a;
}

function checkTrace(w,path){
 if(!path.length)throw Error("empty trace");
 for(let i=0;i<path.length;i++){
  if(!w.open.has(key(path[i]))||w.blocked.has(key(path[i])))throw Error("protected/blocked host entry");
  if(i&&Math.abs(path[i][0]-path[i-1][0])+Math.abs(path[i][1]-path[i-1][1])!==1)throw Error("non-cardinal host step");
 }
 return path.length-1;
}

function makeExample(w,task,variant,split){
 const tested=variant==="W"?"W":"E",other=tested==="W"?"E":"W";
 let path=task==="memory"?lobeTrace(w,tested):enterTrace(w);
 checkTrace(w,path);
 const p=w.hub,menu=neighbors(w,p),local=localGrid(w,p),last12=path.slice(-12);
 if(key(path[path.length-1])!==key(p)||menu.join()!=="E,S,W")throw Error("illegal junction ABI");
 const obs={local_material_7x7:local,position:p,heading:"N",legal_cardinal_moves:menu,last12_positions:last12,
    global_goal_visible:false,hidden_probe_sites:true};
 const base={schema:"driv3r/corpus-v9",split,task,group_id:w.topology+":"+w.seed,
   topology_group:w.topology,scenario_id:w.topology+":"+w.seed+":"+task+":"+variant,
   source:"external-deterministic-synthetic",native_receipt:false,
   difficulty:task==="memory"?"D4-looped-history-aliased":"D3-remote-material-law",
   provenance:{generator:"cycle_memory_v9.js",grammar:"two-overlapping-cycles",style_seed:w.seed,
     span:w.span,height:w.height,host_applied_steps:path.length-1,world_style_hash32:w.style.toString(16)},
   observation:obs,legal_actions:menu.map(x=>"MOVE_"+({N:"NORTH",E:"EAST",S:"SOUTH",W:"WEST"}[x])),
   trajectory:{host_applied:true,positions:path,probe_failure:task==="memory"?tested:null,protected_entries:0},
   forbidden_training_fields:["scenario_id","group_id","topology_group","split","provenance","target","hard_negatives","trajectory"]};
 let label,context;
 if(task==="memory"){
   label="MOVE_"+(other==="W"?"WEST":"EAST");
   const count={W:0,E:0};count[tested]=1;
   context={executed_path: path, failed_probe:{arm:tested,report:"dead_end"},branch_visits:count};
 }else{
   const law=variant==="dry"?"dry":"wet";
   const costs={WEST:law==="dry"?1:4,EAST:law==="dry"?4:1};
   const best=costs.WEST<costs.EAST?"WEST":"EAST";
   label="MOVE_"+best;
   context={law,remote_survey:{WEST:"sand",EAST:"metal"},per_cell_traversal_cost:costs,
       objective:"choose lowest total cost to distant probe; equal geometric distances"};
 }
 const model_input={...obs,context};
 const inputShort={...obs,context:task==="memory"?{last12_positions:last12}:context};
 const negatives=base.legal_actions.filter(a=>a!==label);
 const illegal=["MOVE_NORTH","PAINT_PROTECTED","MOVE_DIAGONAL"];
 if(!base.legal_actions.includes(label))throw Error("oracle illegal");
 return {...base,model_input,target:label,hard_negatives:{legal_but_wrong:negatives,host_illegal:illegal},
   controls:{history_12_only:inputShort,history_zero:{...obs,context:task==="memory"?{}:context}},
   oracle:{rule:task==="memory"?"take branch not recorded as failed":"minimize surveyed traversal cost under law",
    verified_remote_probe_distance:w.span+Math.floor(w.height/2),hidden_target:task==="memory"?other:null}};
}

function specs(){
 const train=[],dev=[],heldout=[];
 for(const s of [12,16,20,24])for(const h of [14,18,22,26,30,34,38,42,46])train.push([s,h]);
 for(const h of [14,18,22,26,30,34,38,42,46,50,54,58])dev.push([28,h]);
 for(const s of [32,36,40])for(const h of [18,26,34,42])heldout.push([s,h]);
 return {TRAIN:train,DEV:dev,HELDOUT:heldout};
}

function produce(seedsPerTopology=16){
 let rows={TRAIN:[],DEV:[],HELDOUT:[]},detail={},shapes=new Map(),pairs=0,totalSteps=0;
 for(const [split,combos] of Object.entries(specs()))for(let index=0;index<combos.length;index++){
  const [span,height]=combos[index],shape=span+":"+height;
  if(shapes.has(shape))throw Error("topology leakage");shapes.set(shape,split);
  for(let s=0;s<seedsPerTopology;s++){
   const seed=fnv("cs-v9:"+shape+":"+s),w=world(span,height,seed),items=[
     makeExample(w,"memory","W",split),makeExample(w,"memory","E",split),
     makeExample(w,"material_law","dry",split),makeExample(w,"material_law","wet",split)];
   if(items[0].target===items[1].target||items[2].target===items[3].target)throw Error("noncontradictory pair");
   if(JSON.stringify(items[0].observation)!==JSON.stringify(items[1].observation))throw Error("local memory leakage");
   if(JSON.stringify(items[0].observation.last12_positions)!==JSON.stringify(items[1].observation.last12_positions))throw Error("short-memory leakage");
   if(JSON.stringify(items[0].controls.history_12_only)!==JSON.stringify(items[1].controls.history_12_only))throw Error("12-history leakage");
   if(JSON.stringify(items[2].observation)!==JSON.stringify(items[3].observation))throw Error("law observable drift");
   for(const row of items){if(row.legal_actions.includes("MOVE_NORTH")||!row.legal_actions.includes(row.target))throw Error("host menu mismatch");
     if(row.trajectory.protected_entries)throw Error("protected mutation");
     totalSteps+=row.provenance.host_applied_steps;
     rows[split].push(row);}
   pairs+=2;
  }
 }
 for(const [split,data] of Object.entries(rows)){
   const keys=new Set(data.map(r=>r.scenario_id));
   if(keys.size!==data.length)throw Error("duplicate scenario id");
   detail[split]={topologies:specs()[split].length,world_groups:data.length/4,pairs:data.length/2,rows:data.length,
    by_task:{memory:data.filter(x=>x.task==="memory").length,material_law:data.filter(x=>x.task==="material_law").length},
    max_prelude_steps:Math.max(...data.map(r=>r.provenance.host_applied_steps)),
    min_prelude_steps:Math.min(...data.map(r=>r.provenance.host_applied_steps))};
 }
 return {rows,summary:{schema:"driv3r/corpus-v9",source:"external synthetic only",native_receipt:false,
   control:"All counterfactual siblings remain in same split; topology (span,height) never crosses splits.",
   diagnostics:detail,total_pairs:pairs,total_host_applied_steps:totalSteps,
   short_memory_identical_for_every_memory_pair:true,no_protected_entries:true,no_leakage_across_topology_groups:true,
   negative_results:"This dataset proves input aliasing, NOT any trained policy's navigation competence."}};
}
function sha(s){return crypto.createHash("sha256").update(s).digest("hex");}
function main(){
 const output=process.argv[2],n=process.argv[3]===undefined?16:Number(process.argv[3]);
 if(!output||!Number.isInteger(n)||n<1||n>64)throw Error("Usage: node cycle_memory_v9.js NEW_OUTPUT_DIR [seeds 1..64]");
 if(fs.existsSync(output))throw Error("refuse overwrite existing destination");
 const {rows,summary}=produce(n);fs.mkdirSync(output,{recursive:true});
 const shards={};for(const [split,records] of Object.entries(rows)){
  const data=records.map(r=>JSON.stringify(r)).join("\n")+"\n",name=split+".jsonl";
  fs.writeFileSync(path.join(output,name),data,{flag:"wx"});
  shards[split]={file:name,sha256:sha(data),bytes:Buffer.byteLength(data),records:records.length,
   scope:split==="TRAIN"?"optimizer-allowed":"evaluation-only; DO NOT TRAIN"};
 }
 const manifest={...summary,seeds_per_topology:n,shards,
  explicit_input_contract:"Only model_input is a model feature; no top-level metadata, target, oracle, trajectory, hard_negatives or controls are inputs.",
  v9_scope:"partially-observed scripted host preludes and one labelled choice per task, not fully model-driven rollout",
  safety:"locally legal synthetic cardinal action menu; NOT verified native CyberSand ABI"};
 fs.writeFileSync(path.join(output,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
 process.stdout.write(JSON.stringify({output,shards,counts:manifest.diagnostics},null,2)+"\n");
}
if(require.main===module)main();
module.exports={world,lobeTrace,checkTrace,makeExample,produce,specs};
