"use strict";
// FORECAST-UTILITY-002 — independently executed synthetic research. Never a host ABI.
const forecastStudy = function forecastStudy(){
"use strict";
const VERSION="driv3r-forecast-utility/1";
const H=[36,72,128],TRAIN=Array.from({length:48},(_,i)=>1000+i),DEV=Array.from({length:16},(_,i)=>3000+i);
const MODES=["early","late","irrelevant"],ARMS=["lazy","overlap","epoch","privileged_oracle","no_cue","cue_classifier","cue_utility"];
const DIR=[["MOVE_E",1,0],["MOVE_N",0,-1],["MOVE_S",0,1],["MOVE_W",-1,0]];
const lookup=Object.fromEntries(DIR.map(([a,x,y])=>[a,[x,y]]));
const started=Date.now(),key=p=>p.join(",");
const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
const digest=s=>{let a=0x811c9dc5,b=0x9e3779b9;for(let i=0;i<s.length;i++){const x=s.charCodeAt(i);a=Math.imul(a^x,16777619)>>>0;b=Math.imul(b+x,2246822519)>>>0}return a.toString(16).padStart(8,"0")+b.toString(16).padStart(8,"0")};
const stateHash=w=>digest(JSON.stringify({pos:w.cursor,trail:w.trail,ink:[...w.ink].sort(),materials:[...w.materials].sort(),epoch:w.epoch,steps:w.steps,paints:w.paints,stopped:w.stopped}));
function spec(split,seed,horizon,mode){
 assert((split==="train"?TRAIN:split==="development"?DEV:[]).includes(seed)&&H.includes(horizon)&&MODES.includes(mode),"split/seed invalid; structural HELDOUT sealed");
 return{split,seed,horizon,mode,id:`${split}:${seed}:${horizon}:${mode}`};
}
function world(sp){
 const gx=({36:24,72:50,128:88})[sp.horizon]+sp.seed%3-1;
 const gatex=({36:12,72:24,128:43})[sp.horizon]+Math.floor(sp.seed/3)%3-1;
 const gap=[3,4,12,13][sp.seed%4],materials=new Set();
 for(let y=0;y<17;y++)if(y!==8&&y!==gap)materials.add(key([gatex,y]));
 for(let x=7;x<gatex;x++){materials.add(key([x,7]));materials.add(key([x,9]))}
 const start=[2,8],w={sp,gx,gate:[gatex,8],goal:[gx,8],offroute:[gx-2,15],width:gx+3,
  cursor:start,trail:[start],ink:new Set([key(start)]),materials,epoch:0,steps:0,paints:0,stopped:false};
 w.occupied=p=>p[0]<0||p[0]>=w.width||p[1]<0||p[1]>=17||w.ink.has(key(p))||w.materials.has(key(p));
 w.menu=()=>w.stopped?[]:[...DIR.filter(([,dx,dy])=>!w.occupied([w.cursor[0]+dx,w.cursor[1]+dy])).map(x=>x[0]),...(w.trail.length>1?["BACKTRACK"]:[]),"STOP"];
 w.step=action=>{
  if(!w.menu().includes(action))throw Error("ILLEGAL_SYNTHETIC_ACTION:"+action);
  if(lookup[action]){const[dx,dy]=lookup[action];w.cursor=[w.cursor[0]+dx,w.cursor[1]+dy];w.trail.push(w.cursor);w.ink.add(key(w.cursor));w.paints++}
  else if(action==="BACKTRACK"){w.trail.pop();w.cursor=w.trail[w.trail.length-1]}else w.stopped=true;
  w.steps++;
 };
 w.event=(tick,kind)=>{
  const cell=sp.mode==="irrelevant"?w.offroute:w.gate,p=key(cell),pre=stateHash(w);
  if(kind==="close"){assert(tick===5&&!w.materials.has(p)&&!w.ink.has(p),"invalid close");w.materials.add(p)}
  else{assert(kind==="open"&&w.materials.has(p),"invalid open");w.materials.delete(p)}
  w.epoch++;return{tick,kind,cell,epoch:w.epoch,pre,post:stateHash(w)};
 };
 return w;
}
function bfs(w){
 if(key(w.cursor)===key(w.goal))return[];
 const dist=new Map([[key(w.goal),0]]),q=[w.goal];
 for(let i=0;i<q.length;i++){
  const p=q[i],d=dist.get(key(p));
  for(const[,dx,dy]of DIR){const n=[p[0]+dx,p[1]+dy],id=key(n);if(!dist.has(id)&&!w.occupied(n)){dist.set(id,d+1);q.push(n)}}
 }
 let best=null;
 for(let k=0;k<w.trail.length;k++){const p=w.trail[w.trail.length-1-k];
  for(const[action,dx,dy]of DIR){const n=[p[0]+dx,p[1]+dy],id=key(n);
   if(!dist.has(id))continue;
   const c=[k+1+dist.get(id),k,action,n];
   if(!best||c[0]<best[0]||(c[0]===best[0]&&c[1]<best[1]))best=c;
  }
 }
 if(!best)return null;
 const[,k,action,n]=best,path=Array(k).fill("BACKTRACK");path.push(action);let p=n;
 while(key(p)!==key(w.goal)){
  const d=dist.get(key(p));let next=null;
  for(const[a,dx,dy]of DIR){const r=[p[0]+dx,p[1]+dy];if(dist.get(key(r))===d-1){next=[a,r];break}}
  assert(next!==null,"BFS reconstruction");path.push(next[0]);p=next[1];
 }
 return path;
}
function cacheTargets(w,cache){
 let pos=w.cursor,trail=w.trail.slice(),targets=[];
 for(const act of cache){
  if(act==="BACKTRACK"){assert(trail.length>1,"cache history");trail.pop();pos=trail[trail.length-1]}
  else{const[dx,dy]=lookup[act];pos=[pos[0]+dx,pos[1]+dy];trail.push(pos)}
  targets.push(key(pos));
 }
 return targets;
}
function forecast(sp){
 // Exogenous, deliberately future-conditioned experimental sensor; NOT in CyberSand's ABI.
 if(sp.mode==="irrelevant")return 0;
 const wrong=sp.seed%4===0;
 return ((sp.mode==="late")!==wrong)?1:-1;
}
function features(sp){return[1,sp.mode==="irrelevant"?0:1,forecast(sp),sp.horizon/128]}
function logistic(x,w){let z=0;for(let i=0;i<w.length;i++)z+=x[i]*w[i];return 1/(1+Math.exp(-Math.max(-35,Math.min(35,z))))}
function train(rows,cue){
 const weights=[0,0,0,0],m=[0,0,0,0],v=[0,0,0,0],losses=[];
 let rand=214013,step=0;
 const sample=n=>{rand=(Math.imul(1664525,rand)+1013904223)>>>0;return rand%n};
 for(const horizon of H){
  const batch=rows.filter(r=>r.sp.horizon===horizon);
  for(let j=0;j<120;j++){
   const g=[0,0,0,0];
   for(let i=0;i<40;i++){const r=batch[sample(batch.length)],x=r.x.slice();if(!cue)x[2]=0;
    const err=(logistic(x,weights)-r.y)/40;for(let k=0;k<4;k++)g[k]+=err*x[k]}
   step++;
   for(let k=0;k<4;k++){
    m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]*g[k];
    weights[k]-=.02*(m[k]/(1-Math.pow(.9,step)))/(Math.sqrt(v[k]/(1-Math.pow(.999,step)))+1e-8);
   }
  }
  let loss=0,correct=0;
  for(const r of batch){const x=r.x.slice();if(!cue)x[2]=0;
   const p=Math.max(1e-9,Math.min(1-1e-9,logistic(x,weights)));
   loss+=-r.y*Math.log(p)-(1-r.y)*Math.log(1-p);correct+=Number((p>=.5)===Boolean(r.y))}
  losses.push({horizon,updates:step,cross_entropy:+(loss/batch.length).toFixed(6),accuracy:+(correct/batch.length).toFixed(6)});
 }
 return{weights,losses,updates:step,weights_fnv64:digest(JSON.stringify(weights))};
}
function episode(sp,arm,models,opts={}){
 const cap=opts.cap||64,record=Boolean(opts.record),w=world(sp);
 assert(cap===64||(cap===128&&sp.horizon===128),"128 cap only for research shadow128");
 assert(ARMS.includes(arm),"invalid arm");
 let cache=bfs(w);assert(cache&&cache[0].startsWith("MOVE_"),"initial route failed");
 let calls=1,invalidations=0,illegal=0,steps=[],counts={},prediction=null,overlap=null;
 for(let tick=0;tick<sp.horizon;tick++){
  if(key(w.cursor)===key(w.goal)||w.stopped||w.paints>=cap)break;
  let event=null;
  if(tick===5){
   event=w.event(tick,"close");
   const cell=sp.mode==="irrelevant"?w.offroute:w.gate;
   overlap=cacheTargets(w,cache).includes(key(cell));
   assert(overlap===(sp.mode!=="irrelevant"),"route intersection invariant");
   let invalidate=false;
   if(arm==="epoch")invalidate=true;
   if(arm==="overlap")invalidate=overlap;
   if(arm==="privileged_oracle")invalidate=overlap&&sp.mode==="late";
   if(arm==="no_cue"||arm==="cue_classifier"){
    const x=features(sp).slice();if(arm==="no_cue")x[2]=0;
    prediction=logistic(x,arm==="no_cue"?models.nocue.weights:models.cue.weights);
    invalidate=prediction>=.5;
   }
   if(arm==="cue_utility")invalidate=Boolean(models.utilityRules[`${sp.horizon}:${forecast(sp)}`]);
   if(invalidate){cache=[];invalidations++}
  }
  if(tick===8&&(sp.mode==="early"||sp.mode==="irrelevant"))event=w.event(tick,"open");
  if(tick===sp.horizon-1&&sp.mode==="late"&&w.materials.has(key(w.gate)))event=w.event(tick,"open");
  if(!cache.length){cache=bfs(w)||[];calls++}
  let action=cache.length?cache[0]:"STOP";
  if(!w.menu().includes(action)){
   illegal++;cache=bfs(w)||[];calls++;action=cache.length?cache[0]:"STOP";
  }
  if(cache[0]===action)cache=cache.slice(1);
  const pre=record?stateHash(w):null,menu=record?w.menu():null;
  w.step(action);counts[action]=(counts[action]||0)+1;
  if(record)steps.push({tick,event,pre,menu,action,post:stateHash(w),cursor:w.cursor.slice()});
 }
 const arrived=key(w.cursor)===key(w.goal),end=stateHash(w);
 return{spec:sp.id,arm,cap,cohort:cap===64?"native64_proxy":"shadow128",arrived,steps:w.steps,
  paint_moves:w.paints,backtracks:counts.BACKTRACK||0,planner_calls:calls,invalidations,illegal_prevented:illegal,
  reason:arrived?"goal_arrived":w.paints>=cap?"paint_censor":w.stopped?"policy_stop":"decision_budget",
  action_counts:counts,state_digest:record?end:null,prediction,overlap,
  ...(record?{trace:{spec:sp.id,arm,cap,initial:stateHash(world(sp)),steps,final:end}}:{})};
}
function replay(row){
 const tr=row.trace,[split,id,h,mode]=tr.spec.split(":"),w=world(spec(split,+id,+h,mode));
 assert(stateHash(w)===tr.initial,"initial hash");
 for(const item of tr.steps){
  if(item.event){const ev=w.event(item.tick,item.event.kind);assert(JSON.stringify(ev)===JSON.stringify(item.event),"event receipt tampered")}
  assert(stateHash(w)===item.pre,"pre state tampered");
  assert(JSON.stringify(w.menu())===JSON.stringify(item.menu),"legal menu tampered");
  w.step(item.action);
  assert(stateHash(w)===item.post&&JSON.stringify(w.cursor)===JSON.stringify(item.cursor),"action receipt tampered");
 }
 assert(stateHash(w)===tr.final&&stateHash(w)===row.state_digest,"final hash");
 return tr.steps.length;
}
const utility=r=>(r.arrived?1000:0)-2*r.steps-3*r.planner_calls-r.backtracks;
const empty={cue:{weights:[0,0,0,0]},nocue:{weights:[0,0,0,0]},utilityRules:{}};
const rows=[],deltas={};
for(const h of H)for(const seed of TRAIN)for(const mode of MODES){
 const sp=spec("train",seed,h,mode),cap=h===128?128:64;
 const lazy=episode(sp,"lazy",empty,{cap}),overlap=episode(sp,"overlap",empty,{cap});
 const diff=utility(overlap)-utility(lazy),x=features(sp),y=Number(diff>0),k=`${h}:${forecast(sp)}`;
 rows.push({sp,x,y,diff});
 if(!(k in deltas))deltas[k]=[];
 deltas[k].push(diff);
}
const cue=train(rows,true),nocue=train(rows,false),utilityRules={};
for(const[k,v]of Object.entries(deltas))utilityRules[k]=v.reduce((a,b)=>a+b,0)/v.length>0;
const models={cue,nocue,utilityRules};
let ntests=0,receipts=0,episodes=0;
function check(x,m){assert(x,"test failed: "+m);ntests++}
const early=spec("development",3001,36,"early"),late=spec("development",3001,36,"late");
check(JSON.stringify(features(early).map((v,i)=>i===2?0:v))===JSON.stringify(features(late).map((v,i)=>i===2?0:v)),"matched hidden future");
check(forecast(early)===-forecast(late),"forecast differentiates opposing pairs");
check(!world(early).menu().includes("BACKTRACK"),"host history is not initially available");
{const w=world(early);w.step("MOVE_E");check(w.menu().includes("BACKTRACK")&&!w.menu().includes("MOVE_W"),"host ink and BACKTRACK");
 try{w.step("MOVE_W");throw Error("expected illegal action")}catch(e){check(String(e.message).startsWith("ILLEGAL_SYNTHETIC_ACTION"),"illegal action refused")}}
check(cue.weights_fnv64===train(rows,true).weights_fnv64,"cue training determinism");
check(nocue.weights_fnv64===train(rows,false).weights_fnv64,"no-cue training determinism");
check(cue.updates===360&&nocue.updates===360,"three-stage bounded optimizer");
const results=[],examples=[];
for(const h of H)for(const seed of DEV)for(const mode of MODES)for(const cap of (h===128?[64,128]:[64])){
 const sp=spec("development",seed,h,mode);
 for(const arm of ARMS){
  const row=episode(sp,arm,models,{cap,record:true});
  receipts+=replay(row);episodes++;
  if(seed===3001&&h===72&&(mode==="early"||mode==="late")&&(arm==="lazy"||arm==="overlap"||arm==="cue_classifier")){
   examples.push({spec:sp.id,arm,cap,arrived:row.arrived,paint_moves:row.paint_moves,planner_calls:row.planner_calls,
    receipt_fnv64:digest(JSON.stringify(row.trace)),actions:row.trace.steps.map(s=>s.action),trace:row.trace});
  }
  delete row.trace;results.push(row);
 }
}
check(episodes===1344&&results.length===1344,"full development factorial");
check(receipts>80000,"all receipts independently replayed");
{const row=episode(late,"overlap",models,{record:true});check(replay(row)===row.trace.steps.length,"valid replay");
 const changed=JSON.parse(JSON.stringify(row));changed.trace.steps[0].action="STOP";
 let rejected=false;try{replay(changed)}catch(_){rejected=true}check(rejected,"tampered receipt rejected");}
{const sp=spec("development",3000,128,"early"),native=episode(sp,"lazy",models),shadow=episode(sp,"lazy",models,{cap:128});
 check(!native.arrived&&native.paint_moves===64&&shadow.arrived&&shadow.paint_moves>64,"native 64 censor vs shadow 128");}
let channelCorrect=0;
for(const h of H)for(const seed of DEV)for(const mode of ["early","late"])
 channelCorrect+=Number((forecast(spec("development",seed,h,mode))===1)===(mode==="late"));
check(channelCorrect===72,"controlled cue exactly 75% correct on gate development cases");
const aggmap=new Map();
for(const row of results){const[,seed,h,mode]=row.spec.split(":"),k=[row.cohort,h,mode,row.arm].join("|");
 if(!aggmap.has(k))aggmap.set(k,[]);aggmap.get(k).push(row)}
const aggregates=[...aggmap].sort((a,b)=>a[0].localeCompare(b[0])).map(([k,rs])=>{
 const[cohort,h,mode,arm]=k.split("|"),n=rs.length,c={};
 for(const r of rs)for(const[a,count]of Object.entries(r.action_counts))c[a]=(c[a]||0)+count;
 return{cohort,horizon:+h,mode,arm,n,goals:rs.filter(r=>r.arrived).length,
  mean_decisions:+(rs.reduce((s,r)=>s+r.steps,0)/n).toFixed(3),
  mean_paints:+(rs.reduce((s,r)=>s+r.paint_moves,0)/n).toFixed(3),
  mean_planner_calls:+(rs.reduce((s,r)=>s+r.planner_calls,0)/n).toFixed(3),
  mean_utility:+(rs.reduce((s,r)=>s+utility(r),0)/n).toFixed(3),action_counts:c};
});
function accuracy(arm){
 let correct=0,total=0;
 for(const h of H)for(const seed of DEV)for(const mode of MODES){
  const sp=spec("development",seed,h,mode),x=features(sp);let yes=false;
  if(arm==="cue_classifier")yes=logistic(x,cue.weights)>=.5;
  if(arm==="no_cue"){x[2]=0;yes=logistic(x,nocue.weights)>=.5}
  if(arm==="cue_utility")yes=utilityRules[`${h}:${forecast(sp)}`];
  if(arm==="overlap")yes=mode!=="irrelevant";
  if(arm==="privileged_oracle")yes=mode==="late";
  correct+=Number(yes===(mode==="late"));total++;
 }
 return{correct,total,accuracy:+(correct/total).toFixed(6)};
}
const handoffs=Object.fromEntries(["overlap","no_cue","cue_classifier","cue_utility","privileged_oracle"].map(a=>[a,accuracy(a)]));
const utilityCalibration=Object.fromEntries(Object.entries(deltas).map(([k,v])=>[k,{n:v.length,mean_overlap_minus_lazy:+(v.reduce((a,b)=>a+b,0)/v.length).toFixed(3),choose_overlap:utilityRules[k]}]));
const summary=Object.fromEntries(ARMS.map(arm=>{
 const rs=results.filter(r=>r.arm===arm);
 return[arm,{episodes:rs.length,goals:rs.filter(r=>r.arrived).length,
  mean_utility:+(rs.reduce((s,r)=>s+utility(r),0)/rs.length).toFixed(3),
  total_planner_calls:rs.reduce((s,r)=>s+r.planner_calls,0),
  total_backtracks:rs.reduce((s,r)=>s+r.backtracks,0)}];
}));
check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===36&&x.mode==="late"&&x.arm==="lazy").goals===0,"36 late lazy failure control");
check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===72&&x.mode==="early"&&x.arm==="lazy").goals===16,"72 early lazy success control");
function gallery(examples){
 let s='<svg xmlns="http://www.w3.org/2000/svg" width="960" height="402" viewBox="0 0 960 402"><rect width="960" height="402" fill="#121b27"/><text x="15" y="23" fill="#eaf3ff" font-family="sans-serif" font-size="16">FORECAST-UTILITY-002: synthetic 72-decision gate trajectories (not CyberSand gameplay)</text>';
 for(let i=0;i<examples.length;i++){
  const e=examples[i],[,seed,h,mode]=e.spec.split(":"),w=world(spec("development",+seed,+h,mode));
  const ox=9+(i%3)*318,oy=34+Math.floor(i/3)*178,sx=270/w.width,sy=114/17;
  let pos=[2,8],stack=[pos],path=[pos];
  for(const t of e.trace.steps){
   if(lookup[t.action]){const[dx,dy]=lookup[t.action];pos=[pos[0]+dx,pos[1]+dy];stack.push(pos);path.push(pos)}
   else if(t.action==="BACKTRACK"){stack.pop();pos=stack[stack.length-1];path.push(pos)}
  }
  s+=`<g transform="translate(${ox},${oy})"><rect width="310" height="172" rx="5" fill="#233345"/><text x="9" y="17" font-family="sans-serif" font-size="13" fill="#edf5ff">${mode} / ${e.arm} / ${e.arrived?"arrived":"censored"}</text><g transform="translate(10,27)"><rect width="285" height="114" fill="#172838"/><line x1="${w.gate[0]*sx}" x2="${w.gate[0]*sx}" y1="0" y2="114" stroke="#df9968" stroke-dasharray="3 2" stroke-width="3"/><polyline fill="none" stroke="${e.arrived?"#66e0b7":"#f0ac78"}" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.4" points="${path.map(p=>`${(p[0]*sx).toFixed(1)},${(p[1]*sy).toFixed(1)}`).join(" ")}"/><circle cx="${w.goal[0]*sx}" cy="${8*sy}" r="3.5" fill="#71d6ee"/></g><text x="9" y="158" font-family="monospace" font-size="11" fill="#cbd4e4">paint ${e.paint_moves}; BFS ${e.planner_calls}; trace ${e.receipt_fnv64.slice(0,8)}</text></g>`;
 }
 return s+"</svg>";
}
const svg=gallery(examples);
return{svg,evidence:{
 study:"FORECAST-UTILITY-002",version:VERSION,base_main:"85df84666a8bad18f0092b930c08557b8010687f",
 predecessor:"FUTURE-GATE-ALIAS-001",scope:"DEVELOPMENT synthetic gated corridors, controlled future-conditioned experimental cue, not CyberSand ABI",
 train_seeds:[1000,1047],development_seeds:[3000,3015],
 train_examples:rows.length,updates:cue.updates+nocue.updates,
 real_optimizer:"Independent seeded minibatch Adam logistic training, 120 per 36/72/128 stage per cue and no-cue policy",
 policies:{cue:{weight_fingerprint_fnv64:cue.weights_fnv64,losses:cue.losses},no_cue:{weight_fingerprint_fnv64:nocue.weights_fnv64,losses:nocue.losses}},
 cue_channel:{correct:channelCorrect,total:96,reliability:0.75,
  qualifier:"Exogenous 75%-correct future-conditioned forecast, seeded deterministic error; NOT an observed CyberSand field"},
 handoffs,utility_calibration:utilityCalibration,
 development_episodes:episodes,independent_replay_episodes:episodes,action_receipts_replayed:receipts,
 tests_passed:ntests,summary,aggregates,
 examples:examples.map(({trace,...other})=>other),gallery_fnv64:digest(svg),
 wall_ms:Date.now()-started,native_receipts:0,structural_heldout_opened:false,
 binary_checkpoint_committed:false,
 limitations:["JS replica uses parent-inspired layout/semantics but is not a byte-identical Python replay","No deployed forecast input exists in CyberSand host action/observation ABI","Cost-sensitive model chooses overlap for both cue signs: classifier accuracy alone is not gameplay utility","Weights are reproducibly trained but not retained as immutable checkpoint binaries","Action receipt fingerprints are FNV64 diagnostic hashes, not cryptographic signatures","Per-inference latency percentiles and process RSS require a separate instrumented Node run","shadow128 is synthetic research only, native64 proxy stays capped at 64 logical paint moves"]
}};
};
if(typeof module!=="undefined")module.exports={forecastStudy};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("node:fs"),path=require("node:path");
 const result=forecastStudy(),idx=process.argv.indexOf("--out");
 if(idx>=0){if(!process.argv[idx+1])throw Error("--out requires a directory");const dir=path.resolve(process.argv[idx+1]);fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,"FORECAST_UTILITY_EVIDENCE_2026-11-19.json"),JSON.stringify(result.evidence,null,2)+"\n");fs.writeFileSync(path.join(dir,"FORECAST_UTILITY_PATHS_2026-11-19.svg"),result.svg+"\n");}
 process.stdout.write(JSON.stringify({study:result.evidence.study,episodes:result.evidence.development_episodes,action_receipts:result.evidence.action_receipts_replayed,tests:result.evidence.tests_passed,summary:result.evidence.summary},null,2)+"\n");
}
