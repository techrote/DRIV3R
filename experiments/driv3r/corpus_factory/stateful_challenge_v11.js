/* DRIV3R Corpus Factory v11: independent counterfactual stateful events.
 * DATA ONLY. Synthetic host, not native CyberSand or its canonical ABI.
 * Node 18+: node stateful_challenge_v11.js NEW_OUTPUT_DIR [styleSeeds=16]
 * Sealed HELDOUT, disjoint geometry and grammar, group-level counterfactuals.
 */
"use strict";
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
const sha256=x=>crypto.createHash("sha256").update(x).digest("hex");
function generateDataset(styleSeeds=16){
  if(!Number.isInteger(styleSeeds)||styleSeeds<1||styleSeeds>32)throw Error("styleSeeds must be 1..32");
  const dirs=[["N",0,-1],["E",1,0],["S",0,1],["W",-1,0]];
  const actions={N:"MOVE_NORTH",E:"MOVE_EAST",S:"MOVE_SOUTH",W:"MOVE_WEST"};
  const key=p=>p[0]+","+p[1];
  const h32=s=>{let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0};
  const specs={
    TRAIN:{grammar:"plain",spans:[11,14,17,20],heights:[18,24,30,36]},
    DEV:{grammar:"single-spur",spans:[23,26],heights:[21,27,33]},
    HELDOUT:{grammar:"double-spur",spans:[29,32],heights:[25,31,39]}
  };
  function world(grammar,span,height,seed){
    const cx=span+14,hy=19,by=hy+height,w=2*span+29,h=by+9,open=new Set(),protectedCells=new Set();
    const add=(x,y)=>open.add(x+","+y);
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
    const style=h32("v11-style/"+grammar+"/"+span+"/"+height+"/"+seed);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const p=x+","+y;if(!open.has(p)&&h32("v11-protect/"+style+"/"+p)%13===0)protectedCells.add(p);
    }
    return {grammar,span,height,seed,style,cx,hy,by,w,h,open,protectedCells,
      hub:[cx,hy],goal:[cx,by],id:"v11/"+grammar+"/"+span+"x"+height+"/"+seed,
      topology:grammar+"/"+span+"x"+height};
  }
  function material(w,p){
    const [x,y]=p;if(x<0||y<0||x>=w.w||y>=w.h)return "#";
    if(w.protectedCells.has(key(p)))return "P";
    const z=h32("v11-mat/"+w.style+"/"+key(p))%7;
    if(w.open.has(key(p)))return x<w.cx?"s":x>w.cx?"m":".";return ["#","s","m","r","#","#","w"][z];
  }
  function view(w,p){
    const out=[];for(let dy=-3;dy<=3;dy++){
      let s="";for(let dx=-3;dx<=3;dx++)s+=material(w,[p[0]+dx,p[1]+dy]);out.push(s);
    }return out;
  }
  function legal(w,p,closed=new Set()){
    return dirs.filter(([_,dx,dy])=>w.open.has(key([p[0]+dx,p[1]+dy]))&&!closed.has(key([p[0]+dx,p[1]+dy]))).map(([d])=>actions[d]);
  }
  function checkTrace(w,path,closedIndex=-1,closedGate=null){
    if(!path.length)throw Error("empty trajectory");
    for(let i=0;i<path.length;i++){
      if(!w.open.has(key(path[i]))||w.protectedCells.has(key(path[i])))throw Error("nonopen/protected step");
      if(i>0&&Math.abs(path[i][0]-path[i-1][0])+Math.abs(path[i][1]-path[i-1][1])!==1)throw Error("noncardinal step");
      if(closedGate&&i>=closedIndex&&key(path[i])===key(closedGate))throw Error("entered closed gate");
    }return path.length-1;
  }
  function commonTail(w){
    const out=[];for(let repeat=0;repeat<2;repeat++){
      for(let d=1;d<=12;d++)out.push([w.cx,w.hy-d]);
      for(let d=11;d>=0;d--)out.push([w.cx,w.hy-d]);
    }return out;
  }
  function baseTrace(w){
    const path=[];for(let y=0;y<=w.hy;y++)path.push([w.cx,y]);
    return path.concat(commonTail(w));
  }
  function probeTrace(w,side){
    const s=side==="WEST"?-1:1,mid=Math.floor(w.height/2),x=w.cx+s*w.span;
    const path=[[w.cx,w.hy]];
    for(let d=1;d<=w.span;d++)path.push([w.cx+s*d,w.hy]);
    for(let d=1;d<mid;d++)path.push([x,w.hy+d]);
    const failedAt=path.length-1,gate=[x,w.hy+mid];
    path.push(...path.slice(0,-1).reverse());
    path.push(...commonTail(w));
    checkTrace(w,path,failedAt+1,gate);
    if(key(path.at(-1))!==key(w.hub))throw Error("probe did not return to hub");
    return {path,failedAt,gate,side};
  }
  function route(w,goal,closed=new Set(),law=null){
    const frontier=[{p:w.hub,c:0}],distance=new Map([[key(w.hub),0]]),parents=new Map([[key(w.hub),null]]),
      coords=new Map([[key(w.hub),w.hub]]);
    while(frontier.length){
      frontier.sort((a,b)=>a.c-b.c||key(a.p).localeCompare(key(b.p)));
      const {p,c}=frontier.shift(),pk=key(p);
      if(c!==distance.get(pk))continue;
      if(pk===key(goal)){
        const path=[];for(let t=pk;t!==null;t=parents.get(t))path.push(coords.get(t));
        return {path:path.reverse(),cost:c};
      }
      for(const [_,dx,dy] of dirs){
        const q=[p[0]+dx,p[1]+dy],qk=key(q);
        if(!w.open.has(qk)||closed.has(qk)||w.protectedCells.has(qk))continue;
        let cost=1;
        if(law&&q[0]!==w.cx)cost=(q[0]<w.cx)===(law==="DRY")?1:7;
        const next=c+cost;
        if(next<(distance.get(qk)??Infinity)){
          distance.set(qk,next);parents.set(qk,pk);coords.set(qk,q);frontier.push({p:q,c:next});
        }
      }
    }return null;
  }
  function nextAction(path){
    if(!path||path.length<2)throw Error("no oracle route");
    const a=path[0],b=path[1],d=dirs.find(([_,dx,dy])=>dx===b[0]-a[0]&&dy===b[1]-a[1]);
    if(!d)throw Error("oracle noncardinal");return actions[d[0]];
  }
  function motifBody(w){
    return Array.from({length:36},(_,i)=>h32("v11-motif/"+w.style+"/"+Math.floor(i/3))%3===0?"SAND":"METAL");
  }
  function makeRow(w,split,task,variant){
    const shutter=task==="delayed_shutter",lawTask=task==="delayed_material_law",motif=task==="periodic_motif";
    const side=variant==="A"?"WEST":"EAST",trace=shutter?probeTrace(w,side):{path:baseTrace(w)};
    const path=trace.path,steps=checkTrace(w,path,shutter?trace.failedAt+1:-1,shutter?trace.gate:null);
    const common={position:w.hub,local_material_7x7:view(w,w.hub),last32_positions:path.slice(-32),
      observation_radius:3,protected_cells_immutable:true};
    let menu,target,context,controlContext,oracle,goal=w.goal;
    if(shutter){
      menu=legal(w,w.hub);
      const closed=new Set([key(trace.gate)]),best=route(w,goal,closed);
      target=nextAction(best.path);
      context={objective:"reach merge without entering the remembered closed shutter",goal,
        executed_positions:path,history_events:[
          {step:trace.failedAt,kind:"probe_denied_by_closed_shutter",cell:trace.gate},
          {step:path.length-19,kind:"ambient_status",change:"none"},
          {step:path.length-3,kind:"ambient_status",change:"none"}]};
      controlContext={objective:context.objective,goal,executed_positions:path.slice(-32),
        history_events:context.history_events.filter(e=>e.step>=path.length-32)};
      oracle={closed_gate:trace.gate,path:best.path,cost:best.cost,proof:"synthetic graph shortest safe route"};
      if(target!==(side==="WEST"?"MOVE_EAST":"MOVE_WEST"))throw Error("shutter oracle not opposite");
    }else if(lawTask){
      menu=legal(w,w.hub);
      const law=variant==="A"?"DRY":"WET",best=route(w,goal,new Set(),law);
      target=nextAction(best.path);
      context={objective:"minimise traversal cost under most recent remote material-law receipt",
        goal,executed_positions:path,history_events:[
          {step:0,kind:"material_law_changed",law,west_material:"SAND",east_material:"METAL",traversal_costs:law==="DRY"?{SAND:1,METAL:7}:{SAND:7,METAL:1}},
          {step:path.length-18,kind:"ambient_status",change:"none"},
          {step:path.length-4,kind:"ambient_status",change:"none"}]};
      controlContext={objective:context.objective,goal,executed_positions:path.slice(-32),
        history_events:context.history_events.filter(e=>e.step>=path.length-32)};
      oracle={law,path:best.path,cost:best.cost,proof:"weighted synthetic shortest route"};
      if(target!==(law==="DRY"?"MOVE_WEST":"MOVE_EAST"))throw Error("material-law oracle mismatch");
    }else if(motif){
      menu=["PAINT_SAND","PAINT_METAL"];
      const body=motifBody(w),keyMaterial=variant==="A"?"SAND":"METAL";
      const sequence=[...body,keyMaterial,...body];
      const receipts=sequence.map((material,index)=>({index,material,accepted:true,protected:false}));
      target="PAINT_"+keyMaterial;
      context={objective:"continue the remote periodic 37-cell painting motif at index 73",
        cursor:w.hub,executed_positions:path,pattern_period:37,paint_receipts:receipts};
      controlContext={objective:context.objective,cursor:w.hub,executed_positions:path.slice(-32),
        pattern_period:37,paint_receipts:receipts.slice(-32)};
      oracle={period:37,preceding_same_phase_index:36,material:sequence[36],
        proof:"repeat observed symbol at offset one period"};
      if(target!=="PAINT_"+sequence[36])throw Error("motif oracle mismatch");
    }else throw Error("unrecognized task");
    if(!menu.includes(target)||w.protectedCells.has(key(w.hub)))throw Error("illegal target/cursor");
    const current={...common,legal_actions:menu,objective:context.objective,
      ...(motif?{cursor:w.hub,pattern_period:37}:{goal})};
    const modelInput={...current,history:context};
    const shortInput={...current,history:controlContext};
    const pairId=w.id+"/"+task;
    return {schema:"driv3r/corpus-factory-v11",split,task,
      scenario_id:pairId+"/"+variant,group_id:w.id,topology_group:w.topology,
      pair_id:pairId,source:"external-deterministic-synthetic-host-v11",native_receipt:false,
      difficulty:shutter?"D6-silent-delayed-gate":lawTask?"D5-delayed-material-law":"D5-long-period-motif",
      provenance:{generator:"stateful_challenge_v11.js",grammar:w.grammar,span:w.span,height:w.height,
        style_seed:w.seed,style_hash32:w.style.toString(16),host_applied_movement_steps:steps,
        ...(motif?{host_applied_paint_receipts:73}:{})},
      model_input:modelInput,target,
      controls:{last32_only:shortInput,zero_history:{...current,history:{objective:context.objective}}},
      hard_negatives:{legal_but_wrong:menu.filter(x=>x!==target),
        host_illegal:motif?["PAINT_PROTECTED","PAINT_UNKNOWN"]:["MOVE_SOUTH","MOVE_DIAGONAL","PAINT_PROTECTED"]},
      evidence:{oracle,executed_positions:path,protected_entries:0,
        ...(shutter?{attempted_probe_cell:trace.gate,probe_denied_step:trace.failedAt}:{})},
      excluded_model_fields:["scenario_id","group_id","topology_group","pair_id","split",
        "target","provenance","controls","hard_negatives","evidence","difficulty","source"]};
  }
  const shards={TRAIN:[],DEV:[],HELDOUT:[]},counts={},shapes=new Map(),worlds=new Set(),seenViews=[];
  let steps=0,paintReceipts=0,nearDuplicatesRejected=0;
  for(const [split,spec] of Object.entries(specs)){
    for(const span of spec.spans)for(const height of spec.heights){
      const topology=spec.grammar+"/"+span+"x"+height;
      if(shapes.has(topology))throw Error("cross-split topology");shapes.set(topology,split);
      for(let index=0;index<styleSeeds;index++){
        const seed=h32("v11-seed/"+topology+"/"+index),w=world(spec.grammar,span,height,seed);
        const signature=view(w,w.hub).join("");
        if(seenViews.some(old=>{let d=0;for(let j=0;j<signature.length;j++){
          if(signature[j]!==old[j]&&++d>4)return false;
        }return d<=4;})){nearDuplicatesRejected++;continue}
        seenViews.push(signature);
        if(worlds.has(w.id))throw Error("duplicate world ID");worlds.add(w.id);
        const rows=["delayed_shutter","delayed_material_law","periodic_motif"].flatMap(task=>
          [makeRow(w,split,task,"A"),makeRow(w,split,task,"B")]);
        for(let i=0;i<rows.length;i+=2){
          const a=rows[i],b=rows[i+1];
          if(a.target===b.target)throw Error("paired labels not opposed");
          if(JSON.stringify(a.controls.last32_only)!==JSON.stringify(b.controls.last32_only))
            throw Error("last32 control leaked variant for "+a.pair_id);
          if(JSON.stringify({...a.model_input,history:null})!==JSON.stringify({...b.model_input,history:null}))
            throw Error("non-history observable differs");
          if(!a.model_input.legal_actions.includes(a.target)||!b.model_input.legal_actions.includes(b.target))
            throw Error("oracle not in legal menu");
        }
        for(const row of rows){
          steps+=row.provenance.host_applied_movement_steps;
          paintReceipts+=row.provenance.host_applied_paint_receipts||0;
          shards[split].push(row);
        }
      }
    }
  }
  for(const [split,rows] of Object.entries(shards)){
    const ids=rows.map(r=>r.scenario_id);
    if(new Set(ids).size!==ids.length)throw Error("duplicate scenario");
    counts[split]={topologies:specs[split].spans.length*specs[split].heights.length,
      grammars:[specs[split].grammar],world_groups:rows.length/6,pairs:rows.length/2,rows:rows.length,
      tasks:Object.fromEntries(["delayed_shutter","delayed_material_law","periodic_motif"]
        .map(task=>[task,rows.filter(r=>r.task===task).length]))};
  }
  return {shards,summary:{
    schema:"driv3r/corpus-factory-v11",source:"synthetic host; NOT native CyberSand",
    native_receipt:false,style_seeds_per_topology:styleSeeds,
    split_design:"TRAIN plain; DEV single-spur; HELDOUT double-spur topology grammars, plus disjoint dimensions/seed families",
    pairing:"both siblings share current observation and last 32 positions/paint receipts; full older history changes opposite oracle target",
    checks:"all synthetic movements cardinal/open/nonprotected; protected cell entries 0; oracle graph routing and paired inversion validated",
    controls:"last32_only and zero_history are evaluation ablations, not trainer inputs",
    dedupe:"reject non-sibling world groups with hub-local material views at Hamming distance <=4 across all splits",
    near_duplicate_world_groups_rejected:nearDuplicatesRejected,
    counts,host_applied_movement_steps:steps,synthetic_paint_receipts:paintReceipts,
    limits:["Host-scripted legal synthetic preludes, oracle-labeled decision only; no model closed-loop competence inferred",
      "Synthetic host action names/material law/paint receipts not checked against CyberSand canonical native ABI",
      "HELDOUT must not be used for training or model selection; no actual primary SLM trained",
      "DEV/HELDOUT introduce grammar shifts but primary twin-loop geometry remains shared"]}};
}
function main(){
  const dest=process.argv[2],n=process.argv[3]===undefined?16:Number(process.argv[3]);
  if(!dest||!Number.isInteger(n)||n<1||n>32)throw Error("Usage: node stateful_challenge_v11.js NEW_OUTPUT_DIR [styleSeeds 1..32]");
  if(fs.existsSync(dest))throw Error("refuse to overwrite destination");
  const output=generateDataset(n);
  fs.mkdirSync(dest,{recursive:true});
  const shards={};
  for(const [split,rows] of Object.entries(output.shards)){
    const file=split+".jsonl",content=rows.map(r=>JSON.stringify(r)).join("\n")+"\n";
    fs.writeFileSync(path.join(dest,file),content,{flag:"wx"});
    shards[split]={file,rows:rows.length,bytes:Buffer.byteLength(content),
      sha256:sha256(content),usage:split==="TRAIN"?"optimizer-permitted":"evaluation-only; do not train"};
  }
  const manifest={...output.summary,version:"stateful-v11",generator:"stateful_challenge_v11.js",shards,
    model_input_contract:"ONLY model_input; do not expose target/evidence/controls/metadata to model",
    provenance_note:"FNV32 deterministic generator, SHA256 digest of JSONL bytes, no native receipts"};
  fs.writeFileSync(path.join(dest,"manifest.json"),JSON.stringify(manifest,null,2)+"\n",{flag:"wx"});
  process.stdout.write(JSON.stringify({output:dest,shards,counts:output.summary.counts})+"\n");
}
if(require.main===module)main();
module.exports={generateDataset};
