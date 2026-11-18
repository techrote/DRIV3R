/* DRIV3R Corpus Factory v10 — external SYNTHETIC host, not CyberSand/native ABI.
   Data generation only: no policy trainer, no production changes.
   Node 18+: node dynamic_challenge_v10.js NEW_DIR [style_seeds_per_topology=12]
   Emits TRAIN/DEV/HELDOUT JSONL + sha256 manifest, refuses overwrite. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const H=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const hash32=s=>parseInt(H(s).slice(0,8),16)>>>0;
const k=p=>p[0]+','+p[1];
const DIR=[['N',0,-1],['E',1,0],['S',0,1],['W',-1,0]];
const ACTION={N:'MOVE_NORTH',E:'MOVE_EAST',S:'MOVE_SOUTH',W:'MOVE_WEST'};
function layout(span,height,seed){
 const cx=span+9,hy=19,by=hy+height,open=new Set(),protectedCells=new Set();
 const add=(x,y)=>open.add(x+','+y);
 for(let y=0;y<=hy;y++)add(cx,y); // long common return stem
 for(const s of [-1,1]){
  for(let i=0;i<=span;i++)add(cx+s*i,hy);
  for(let y=hy;y<=by;y++)add(cx+s*span,y);
  for(let i=0;i<=span;i++)add(cx+s*i,by);
 }
 for(let y=by;y<=by+5;y++)add(cx,y);
 const w=2*span+21,h=by+12,style=hash32('v10-style/'+span+'/'+height+'/'+seed);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  if(open.has(x+','+y))continue;
  if(hash32('v10-protect/'+style+'/'+x+','+y)%11===0)protectedCells.add(x+','+y);
 }
 return {span,height,seed,style,cx,hy,by,w,h,open,protectedCells,
  hub:[cx,hy],merge:[cx,by],group:'v10-'+span+'x'+height+'-s'+seed,
  shape:'v10-'+span+'x'+height};
}
function material(w,p){
 if(p[0]<0||p[1]<0||p[0]>=w.w||p[1]>=w.h)return '#';
 if(w.protectedCells.has(k(p)))return 'P';
 const r=hash32('v10-material/'+w.style+'/'+k(p))%5;
 return w.open.has(k(p))?['.','.','s','m','r'][r]:['#','#','s','m','r'][r];
}
function grid(w,p){let rows=[];for(let dy=-3;dy<=3;dy++){
 let line='';for(let dx=-3;dx<=3;dx++)line+=material(w,[p[0]+dx,p[1]+dy]);rows.push(line);
 }return rows;}
function legal(w,p,closed=new Set()){
 return DIR.filter(([name,dx,dy])=>w.open.has(k([p[0]+dx,p[1]+dy]))&&!closed.has(k([p[0]+dx,p[1]+dy])))
  .map(([name])=>ACTION[name]);
}
function assertTrace(w,steps,gateStep=-1,gate=null){
 if(!steps.length)throw Error('empty trace');
 for(let i=0;i<steps.length;i++){
  if(!w.open.has(k(steps[i]))||w.protectedCells.has(k(steps[i])))throw Error('protected/non-open trace');
  if(i&&Math.abs(steps[i][0]-steps[i-1][0])+Math.abs(steps[i][1]-steps[i-1][1])!==1)throw Error('non-cardinal move');
  if(gate&&i>=gateStep&&k(steps[i])===k(gate))throw Error('entered closed gate');
 }
 return steps.length-1;
}
function shortest(w,start,goal,closed=new Set()){
 const queue=[start],prev=new Map([[k(start),null]]),positions=new Map([[k(start),start]]);
 for(let i=0;i<queue.length;i++){
  const p=queue[i];if(k(p)===k(goal)){
   const result=[];for(let cur=k(p);cur!==null;cur=prev.get(cur))result.push(positions.get(cur));
   return result.reverse();
  }
  for(const [_,dx,dy] of DIR){const next=[p[0]+dx,p[1]+dy],key=k(next);
   if(!w.open.has(key)||closed.has(key)||prev.has(key))continue;
   prev.set(key,k(p));positions.set(key,next);queue.push(next);
  }
 }
 return null;
}
function actionBetween(a,b){const dx=b[0]-a[0],dy=b[1]-a[1];
 const t=DIR.find(([_,x,y])=>x===dx&&y===dy);if(!t)throw Error('bad oracle edge');return ACTION[t[0]];}
function inputBase(w,positions){return {local_material_7x7:grid(w,w.hub),position:w.hub,
 last12_positions:positions.slice(-12),legal_actions:legal(w,w.hub),
 protected_cells_are_immutable:true,observation_radius:3};}
function approach(w){let out=[];for(let y=0;y<=w.hy;y++)out.push([w.cx,y]);return out;}
function dynamicPrelude(w,side){
 const x=w.cx+(side==='WEST'?-w.span:w.span),gate=[x,w.hy+Math.floor(w.height/2)];
 let p=[[w.cx,w.hy]];for(let t=1;t<=w.span;t++)p.push([w.cx+(side==='WEST'?-t:t),w.hy]);
 for(let y=w.hy+1;y<gate[1];y++)p.push([x,y]);
 const failedAt=p.length-1; // synthetic host closes gate *ahead* of agent
 p=p.concat(p.slice(0,-1).reverse());
 for(let d=1;d<=16;d++)p.push([w.cx,w.hy-d]);
 for(let d=15;d>=0;d--)p.push([w.cx,w.hy-d]);
 assertTrace(w,p,failedAt+1,gate);if(k(p.at(-1))!==k(w.hub))throw Error('not back at hub');
 return {path:p,gate,failedAt};
}
function makeRow(w,task,variant,split){
 const side=variant==='WEST'?'WEST':'EAST',other=side==='WEST'?'EAST':'WEST';
 const recovery=task==='shutter_recovery',retarget=task==='goal_retarget',motif=task==='motif_echo';
 if(!recovery&&!retarget&&!motif)throw Error('unknown task');
 let prelude=recovery?dynamicPrelude(w,side).path:approach(w);
 const b=inputBase(w,prelude),group_id=w.group,tag=w.group+'/'+task+'/'+variant;
 const menu=motif?['PAINT_SAND','PAINT_METAL']:b.legal_actions;
 let context,target,goal=null,event=null,paintHistory=null;
 if(recovery){
  const d=dynamicPrelude(w,side),closed=new Set([k(d.gate)]);
  goal=w.merge;const route=shortest(w,w.hub,goal,closed);
  target=actionBetween(route[0],route[1]);
  if(target!==ACTION[other==='WEST'?'W':'E'])throw Error('shutter oracle mismatch');
  event={index:d.failedAt,kind:'remote_shutter_closed',cell:d.gate,reported_from:d.path[d.failedAt]};
  context={goal,executed_positions:d.path,host_event_receipts:[event],objective:'reach goal without violating closed shutters'};
 }else if(retarget){
  goal=[w.cx+(side==='WEST'?-w.span:w.span),w.by-2];
  const route=shortest(w,w.hub,goal);
  target=actionBetween(route[0],route[1]);
  if(target!==ACTION[side==='WEST'?'W':'E'])throw Error('goal oracle mismatch');
  context={current_goal:goal,goal_revision:2,goal_event:{kind:'beacon_retarget',position:goal},
   objective:'reach newly selected goal via shortest legal path'};
 }else{
  // Authoring actions on one unprotected cursor cell; paint receipts are remote
  // contextual exemplars, not mutations of the local 7x7 observation.
  const paintable=[w.cx,w.hy];if(w.protectedCells.has(k(paintable)))throw Error('protected cursor');
  const seq=Array.from({length:24},(_,i)=>(i+(side==='WEST'?0:1))%2===0?'SAND':'METAL');
  paintHistory=seq.map((material,i)=>({step:i,material,accepted:true,protected:false}));
  target='PAINT_'+(seq.at(-1)==='SAND'?'METAL':'SAND');
  context={remote_paint_receipts:paintHistory,pattern_contract:'continue the observed alternating material motif',
   cursor:paintable};
 }
 if(!menu.includes(target))throw Error('oracle target outside host legal menu');
 const fullInput={...b,legal_actions:menu,context};
 const shortInput={...b,legal_actions:menu,context:recovery?{goal}:retarget?context:{remote_paint_receipts:paintHistory.slice(-1)}};
 const legalWrong=menu.filter(a=>a!==target);
 if(!legalWrong.length)throw Error('missing hard negative');
 return {schema:'driv3r/corpus-v10',split,task,scenario_id:tag,group_id,
  topology_group:w.shape,pair_id:w.group+'/'+task,source:'external-synthetic-host-v10',
  difficulty:recovery?'D5-long-recovery-alias':retarget?'D4-moving-goal':'D3-motif-continuation',
  native_receipt:false,provenance:{generator:'dynamic_challenge_v10.js',span:w.span,height:w.height,
   style_seed:w.seed,style_hash:H(w.style.toString()),host_applied_moves:assertTrace(w,prelude,recovery?dynamicPrelude(w,side).failedAt+1:-1,recovery?dynamicPrelude(w,side).gate:null),
   kind:recovery?'physical-movement-and-obstruction-event':retarget?'physical-approach-and-goal-update':'physical-approach-and-synthetic-paint-receipts'},
  model_input:fullInput,target,legal_actions:menu,
  controls:{last12_only:shortInput,zero_history:{...b,legal_actions:menu,context:retarget?context:{}}},
  hard_negatives:{legal_but_wrong:legalWrong,host_illegal:motif?['PAINT_PROTECTED','PAINT_UNKNOWN']:['MOVE_SOUTH','MOVE_DIAGONAL','PAINT_PROTECTED']},
  evidence:{prelude_positions:prelude,shutter_event:event,paint_receipts:paintHistory,
   oracle_goal:goal,oracle_route:motif?null:shortest(w,w.hub,goal,recovery?new Set([k(event.cell)]):new Set()),
   protected_entries:0},
  excluded_model_fields:['scenario_id','group_id','topology_group','pair_id','split','target','provenance','evidence','hard_negatives','controls']};
}
function templates(){const out={TRAIN:[],DEV:[],HELDOUT:[]};
 for(const span of [11,14,17])for(const height of [18,22,26,30,34,38])out.TRAIN.push([span,height]);
 for(const span of [20,23])for(const height of [21,27,33])out.DEV.push([span,height]);
 for(const span of [26,29])for(const height of [23,31,39])out.HELDOUT.push([span,height]);
 return out;
}
function produce(n=12){if(!Number.isInteger(n)||n<1||n>64)throw Error('seeds must be 1..64');
 const shards={TRAIN:[],DEV:[],HELDOUT:[]},seenShapes=new Map(),seenLocal=[];
 const skipped={},counts={},pairs={};let steps=0;
 for(const [split,combos] of Object.entries(templates())){
  skipped[split]=0;pairs[split]=0;
  for(const [span,height] of combos){const shape=span+':'+height;
   if(seenShapes.has(shape))throw Error('topology crosses split');seenShapes.set(shape,split);
   for(let i=0;i<n;i++){
    const seed=hash32('v10-world/'+shape+'/'+i),w=layout(span,height,seed);
    const view=grid(w,w.hub).join('');
    if(seenLocal.some(old=>{let d=0;for(let j=0;j<view.length;j++)if(view[j]!==old[j]&&++d>4)return false;return d<=4;})){
     skipped[split]++;continue;
    }
    seenLocal.push(view);
    const rows=['shutter_recovery','goal_retarget','motif_echo'].flatMap(task=>[
      makeRow(w,task,'WEST',split),makeRow(w,task,'EAST',split)]);
    for(let q=0;q<rows.length;q+=2){const a=rows[q],b=rows[q+1];
     if(a.target===b.target)throw Error('no counterfactual flip');
     if(JSON.stringify({...a.model_input,context:undefined})!==JSON.stringify({...b.model_input,context:undefined}))throw Error('observable drift');
     if(a.task==='shutter_recovery'&&JSON.stringify(a.controls.last12_only)!==JSON.stringify(b.controls.last12_only))throw Error('short-history leakage');
     pairs[split]++;
    }
    for(const row of rows){steps+=row.provenance.host_applied_moves;shards[split].push(row);}
   }
  }
  const rows=shards[split];if(new Set(rows.map(x=>x.scenario_id)).size!==rows.length)throw Error('duplicate scenario');
  counts[split]={templates:combos.length,world_groups:rows.length/6,pairs:pairs[split],rows:rows.length,
    by_task:Object.fromEntries(['shutter_recovery','goal_retarget','motif_echo'].map(t=>[t,rows.filter(x=>x.task===t).length])),
    near_duplicate_world_groups_rejected:skipped[split]};
 }
 return {shards,summary:{schema:'driv3r/corpus-v10',generator:'dynamic_challenge_v10.js',
  source:'external deterministic synthetic host, NOT native CyberSand',native_receipt:false,
  group_policy:'shape disjoint across splits; all counterfactual sibling outcomes kept in same split',
  dedupe_policy:'reject world groups whose paired local 7x7 token view is within Hamming distance 4 of any accepted world, across all splits',
  counts,host_applied_movement_steps:steps,known_limits:[
   'Host-only synthetic movement, painting and gate events; no canonical native ABI acceptance',
   'Oracle-labeled first decision after host-executed preludes, not learned-policy full episodes',
   'Goal retargeting and motif continuation need not require long-term memory; shutter recovery uses aliased final 12 positions',
   'Protect DEV and HELDOUT from all optimizer/data selection leakage']}};
}
function main(){const dest=process.argv[2],n=process.argv[3]===undefined?12:Number(process.argv[3]);
 if(!dest||!Number.isInteger(n)||n<1||n>64)throw Error('Usage: node dynamic_challenge_v10.js NEW_OUTPUT_DIR [seeds 1..64]');
 if(fs.existsSync(dest))throw Error('refuse overwrite');
 const out=produce(n);fs.mkdirSync(dest,{recursive:true});const shardInfo={};
 for(const [split,rows] of Object.entries(out.shards)){
  const content=rows.map(x=>JSON.stringify(x)).join('\n')+'\n',filename=split+'.jsonl';
  fs.writeFileSync(path.join(dest,filename),content,{flag:'wx'});
  shardInfo[split]={path:filename,sha256:H(content),bytes:Buffer.byteLength(content),records:rows.length,
   usage:split==='TRAIN'?'optimizer_allowed':'evaluation_only_NO_TRAIN'};
 }
 const manifest={...out.summary,style_seeds_per_topology:n,shards:shardInfo,
  input_contract:'Only model_input is input; target and all metadata, hard negatives, controls and evidence are not features.'};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 process.stdout.write(JSON.stringify({dir:dest,counts:manifest.counts,shards:shardInfo},null,2)+'\n');
}
if(require.main===module)main();
module.exports={layout,grid,legal,shortest,assertTrace,dynamicPrelude,makeRow,templates,produce};
