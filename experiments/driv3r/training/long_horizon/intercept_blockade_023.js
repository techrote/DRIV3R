"use strict";
// INTERCEPT-BLOCKADE-023 — manually opted-in 021 policy stress experiment.
// Research-only dynamic walls and STOP receipt; NO claimed native CyberSand ABI.
// Never start optimizer on import, PR or push. Structural HELDOUT is sealed.
const FILES=["physical_inspection_016.js","multipulse_physical_018.js","stop_reservation_020.js","twin_supervision_021.js"];
const PINS=["45a426f893203f93fa9a3972d63bd4a110ceaa3d","4e7f4b4b20d4db42d601961a7689f3729477aaf6","51b43ff4f31299f9ed0b0ff8a9a559b1e9a0a79d","126051579797450227700b381f5f3524f29d2eb5"];
const CAPS=[[36,36],[72,64],[128,64],[128,128]],EXTRA=["5,4","9,4","6,6"];
const MODES=["feasible","early","late","both"],ARMS=["guard","student021","early","late","both","hindsight"];
const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const forks=h=>h===128?[[0,0],[0,1],[1,0],[1,1]]:[[0,0],[1,0]];
function eventSites(c,e,t,root,cap){
 const log=t.preempt(c,e.variant(root,0),cap,"feasible",0).host.log.slice(root.tick+3);
 const sites=[],wall=new Set(root.walls);
 for(let i=0;i<log.length&&sites.length<2;i++){
  const p=log[i].after,k=p.join(",");
  if(wall.has(k)||[4,8].includes(p[0])||c.TARGETS.some(q=>c.eq(q,p))||
     c.eq(p,root.pos)||sites.length&&i+3<sites[0].pathIndex+5)continue;
  const check=new Set(wall);check.add(k);
  if(!c.TARGETS.every(q=>c.eq(root.pos,q)||c.bfs(root.pos,q,check).length>0))continue;
  sites.push({cell:k,atTick:root.tick+(sites.length?5:1),pathIndex:i+3});wall.add(k);
 }
 return sites;
}
function withEvents(root,events){
 const clone=root.clone(),fn=clone.clone.bind(clone);
 clone.clone=function(){
  const z=fn(),step=z.step.bind(z);
  z.step=function(a){
   const r=step(a);
   for(const ev of events)if(z.tick===ev.atTick&&ev.cell!==z.pos.join(",")&&!z.walls.has(ev.cell))
    z.walls.add(ev.cell);
   return r;
  };return z;
 };return clone;
}
function replay(c,rr,seed,hor,cap,pre,first,second,events){
 const z=new c.Host(seed,hor,"dev"),k="2,1";
 for(const p of EXTRA)z.walls.add(p);
 if(rr.host.log.length>cap)throw Error("censor failure");
 let installed=0;
 for(let i=0;i<rr.host.log.length;i++){
  if(i===pre){
   if(!z.hazard||!z.paintedA)throw Error("prefix mismatch");
   z.ink[k]=1;z.damage=!!first;if(first)delete z.ink[k];
  }
  const old=z.ink[k]??null,receipt=z.step(rr.host.log[i].action);
  if(!equal(receipt,rr.host.log[i]))throw Error("receipt replay mismatch "+seed+"/"+i);
  if(hor===128&&receipt.action==="PAINT"&&c.eq(receipt.after,c.TARGETS[4])&&z.secondHazard){
   if(second)delete z.ink[k];else if(old===null)delete z.ink[k];else z.ink[k]=old;
   z.secondDamage=!!second;
  }
  for(const ev of events)if(z.tick===ev.atTick&&ev.cell!==z.pos.join(",")&&!z.walls.has(ev.cell)){
   z.walls.add(ev.cell);installed++;
  }
 }
 if(!equal(z.pos,rr.host.pos)||!equal(z.ink,rr.host.ink)||
    !equal([...z.walls].sort(),[...rr.host.walls].sort()))throw Error("replayed final state mismatch");
 if(z.log.at(-1)?.action!=="STOP")throw Error("missing synthetic STOP");
 if(c.assertReceipts(rr.host.log)!==rr.host.log.length)throw Error("receipt audit incomplete");
 return{count:z.log.length,installed};
}
function study(c,e,t,z,N=30){
 const begin=Date.now(),model=z.fit(c,e,t),summary={},cohorts={},rows=[],negative=[],galleryRows=[];
 let cases=0,rollouts=0,receipts=0,installations=0,eligibleSingle=0,eligibleDouble=0,rejected=0;
 for(const condition of ["control","one","two"]){
  summary[condition]=Object.fromEntries(ARMS.map(a=>[a,{n:0,goals:0,material:0,utility:0,calls:0,checks:0,
   actions:Object.fromEntries(c.ACT.map(k=>[k,0]))}]));
  for(const [h,cap] of CAPS){
   const cohort=condition+"/"+h+"/"+cap;
   cohorts[cohort]=Object.fromEntries(ARMS.map(a=>[a,0]));
   for(let seed=160100;seed<160100+N;seed++){
    const root=t.prefix(c,seed,h,cap,true);
    if(!root.hazard||!root.paintedA)throw Error("invalid prefix");
    const possible=eventSites(c,e,t,root,cap);
    if(condition==="two"){eligibleSingle+=+(possible.length>=1);eligibleDouble+=+(possible.length>=2)}
    const events=condition==="control"?[]:condition==="one"?possible.slice(0,1):possible;
    const features=e.publicFeatures(c,root,cap),scores=e.probs(model.weights,features);
    const student=MODES[scores.indexOf(Math.max(...scores))];
    for(const [first,second] of forks(h)){
     const world=e.variant(root,first);
     if(!equal(world.observation(),e.variant(root,1-first).observation())||
        !equal(features,e.publicFeatures(c,world,cap)))throw Error("hidden material leaked");
     const tries={};
     for(const mode of MODES){
      const rr=t.preempt(c,withEvents(world,events),cap,mode,second);
      const audit=replay(c,rr,seed,h,cap,root.tick,first,second,events);
      receipts+=audit.count;installations+=audit.installed;rollouts++;
      if(rr.host.tick>cap||rr.out.stopped!==1)throw Error("cap/STOP");
      tries[mode]=rr.out;
     }
     const hindsight=MODES.slice().sort((a,b)=>tries[b].terminalGoal-tries[a].terminalGoal||
      tries[b].utility-tries[a].utility||MODES.indexOf(a)-MODES.indexOf(b))[0];
     const selected={guard:"feasible",student021:student,early:"early",late:"late",both:"both",hindsight};
     const outcome={};
     for(const arm of ARMS){
      const mode=selected[arm],r=tries[mode],agg=summary[condition][arm];
      agg.n++;agg.goals+=r.terminalGoal;agg.material+=r.goal;agg.utility+=r.utility;
      agg.calls+=r.calls;agg.checks+=r.checks;cohorts[cohort][arm]+=r.terminalGoal;
      for(const action of c.ACT)agg.actions[action]+=r.counts[action];
      outcome[arm]={mode,goal:r.terminalGoal,calls:r.calls,hash:r.hash};
     }
     const row={condition,seed,h,cap,first,second,events,student,outcome};rows.push(row);cases++;
     if(outcome.student021.goal<outcome.hindsight.goal||
        outcome.student021.goal<outcome.guard.goal||
        outcome.student021.calls>outcome.guard.calls){
      negative.push(row);
      if(condition!=="control"&&galleryRows.length<24&&
       (outcome.student021.goal<outcome.hindsight.goal||outcome.student021.goal<outcome.guard.goal)){
       for(const mode of new Set(["feasible",student,hindsight]))
        if(galleryRows.length<24)galleryRows.push({condition,seed,h,cap,first,second,mode,
         goal:tries[mode].terminalGoal,positions:tries[mode].positions});
      }
     }
    }
   }
  }
 }
 const prior=new Map(rows.filter(x=>x.condition==="control").map(x=>[[x.seed,x.h,x.cap,x.first,x.second].join("/"),x]));
 const deltas=Object.fromEntries(["one","two"].map(k=>[k,{goals:Object.fromEntries(ARMS.map(a=>[a,0])),
  changedHashes:0,studentBeatsGuard:0,studentLosesGuard:0}]));
 for(const row of rows.filter(x=>x.condition!=="control")){
  const orig=prior.get([row.seed,row.h,row.cap,row.first,row.second].join("/"));
  if(!orig)throw Error("unpaired control");
  const d=deltas[row.condition];
  for(const arm of ARMS){d.goals[arm]+=row.outcome[arm].goal-orig.outcome[arm].goal;
   d.changedHashes+=+(row.outcome[arm].hash!==orig.outcome[arm].hash)}
  d.studentBeatsGuard+=+(row.outcome.student021.goal>row.outcome.guard.goal);
  d.studentLosesGuard+=+(row.outcome.student021.goal<row.outcome.guard.goal);
 }
 const good={tick:0,action:"E",menu:["E"],before:[1,1],after:[2,1],before_material:null,after_material:null,result:"ok"};
 for(const bad of [{...good,action:"TELEPORT"},{...good,after:[99,99]},
  {...good,tick:7},{...good,menu:["STOP"]}])try{z.audit([bad])}catch{rejected++}
 if(cases!==36*N||rollouts!==cases*MODES.length||receipts===0||rejected!==4)throw Error("campaign incomplete");
 for(const modes of Object.values(summary))for(const agg of Object.values(modes))
  agg.meanUtility=+(agg.utility/agg.n).toFixed(3);
 return{study:"INTERCEPT_BLOCKADE_023",provenance:{gitBlobs:PINS,train:"021 TRAIN 120100-120169",
  development:"fresh unsealed 160100-"+(160099+N),structuralHeldout:"UNOPENED"},
  limitations:"Same-source synthetic host replay with separately checked receipts, NOT independent native CyberSand. Dynamic wall/STOP semantics synthetic. Closure targets selected from nominal future guard path as stress design, not legal agent observations.",
  training:{updates:model.updates,labels:model.labels,trainRollouts:model.rollouts,weights:model.weights},
  run:{seedClusters:N,conditions:cases,rollouts,sourceReplayedReceipts:receipts,
   closureInstallations:installations,eligibleOne:eligibleSingle,eligibleTwo:eligibleDouble,
   corruptRejections:rejected,negativeCases:negative.length,v8WallMs:Date.now()-begin,
   nodeRSS:"unmeasured in V8",predictorP95:"unmeasured in this run"},
  summary,cohorts,deltas,negative,rows,galleryRows};
}
function svgGallery(rows){
 const w=1200,h=Math.max(180,Math.ceil(rows.length/3)*175);
 const out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'"><rect width="100%" height="100%" fill="#152732"/>'];
 for(let i=0;i<rows.length;i++){
  const r=rows[i],x=i%3*400,y=Math.floor(i/3)*175;
  out.push('<rect x="'+(x+3)+'" y="'+(y+3)+'" width="394" height="169" fill="#2a3f4c" stroke="#718b99"/>');
  out.push('<polyline fill="none" stroke-width="2" stroke="'+(r.goal?"#8bd9b7":"#e9a399")+'" points="'+r.positions.map(p=>(x+12+p[0]*25)+","+(y+14+p[1]*14)).join(" ")+'"/>');
  out.push('<text x="'+(x+8)+'" y="'+(y+164)+'" fill="white" font-size="11" font-family="monospace">'+r.condition+" "+r.seed+" "+r.h+"/"+r.cap+" "+r.mode+" goal="+r.goal+"</text>");
 }
 return out.concat("</svg>").join("\n")+"\n";
}
if(typeof module!=="undefined")module.exports={study,eventSites,replay,svgGallery};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),vm=require("vm"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))throw Error("Explicit: node intercept_blockade_023.js /absolute/output");
 const exports=["{Host,prefix,move,correct,bfs,TARGETS,ACT,DIR,eq,assertReceipts}",
 "{variant,publicFeatures,train,probs,CANDIDATES}","{preempt,prefix}","{fit,audit}"];
 const modules=FILES.map((file,i)=>{
  const data=fs.readFileSync(path.join(__dirname,file));
  const sha=crypto.createHash("sha1").update("blob "+data.length+"\0").update(data).digest("hex");
  if(sha!==PINS[i])throw Error("dependency drift "+file);
  return vm.runInNewContext(data.toString("utf8")+"\n;("+exports[i]+")",{});
 });
 const r=study(...modules,30);
 r.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 r.provenance.scriptSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 fs.mkdirSync(process.argv[2],{recursive:true});
 fs.writeFileSync(path.join(process.argv[2],"INTERCEPT_BLOCKADE_023_EVIDENCE.json"),JSON.stringify(r,null,2)+"\n");
 fs.writeFileSync(path.join(process.argv[2],"INTERCEPT_BLOCKADE_023_PATHS.svg"),svgGallery(r.galleryRows));
 console.log(JSON.stringify({run:r.run,summary:r.summary,cohorts:r.cohorts,deltas:r.deltas},null,2));
}
