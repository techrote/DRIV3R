"use strict";
// ROUTE-INVALIDATION-022. Opt-in bounded game-policy research; never CI-triggered.
// 64 logical moves are censored; 128/128 remains a separate synthetic shadow.
// Route closures and terminal STOP accounting are experimental, not native ABI.
const FILES=["physical_inspection_016.js","multipulse_physical_018.js","stop_reservation_020.js","twin_supervision_021.js"];
const BLOBS=["45a426f893203f93fa9a3972d63bd4a110ceaa3d","4e7f4b4b20d4db42d601961a7689f3729477aaf6","51b43ff4f31299f9ed0b0ff8a9a559b1e9a0a79d","126051579797450227700b381f5f3524f29d2eb5"];
const EXTRA=["5,4","9,4","6,6"],CAPS=[[36,36],[72,64],[128,64],[128,128]],MODES=["feasible","early","late","both","never"];
const ARMS=["guard","student021","frozen018","hindsight"],START=150100,N=50;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),key=p=>p.join(",");
const forks=h=>h===128?[[0,0],[0,1],[1,0],[1,1]]:[[0,0],[1,0]];
function chooseClosure(c,root){
 const target=root.horizon===128?c.TARGETS[3]:c.TARGETS[0];
 const route=c.bfs(root.pos,target,root.walls),pos=[...root.pos],sites=[];
 for(const a of route){pos[0]+=c.DIR[a][0];pos[1]+=c.DIR[a][1];sites.push([...pos])}
 for(let i=2;i<Math.min(7,sites.length);i++){
  const cell=sites[i],k=key(cell),blocked=new Set(root.walls);
  if([4,8,9].includes(cell[0])||c.eq(cell,target)||blocked.has(k))continue;
  blocked.add(k);
  const alternate=c.bfs(root.pos,target,blocked);
  if(alternate.length&&alternate.length<60)
   return{cell:k,atTick:root.tick+i,plannedIndex:i,routeLength:route.length,alternateLength:alternate.length};
 }
 return null;
}
function inject(root,event){
 if(!event)return root;
 const adapted=root.clone(),copy=adapted.clone.bind(adapted);
 adapted.clone=function(){
  const h=copy(),advance=h.step.bind(h);
  h.step=function(action){
   const receipt=advance(action);
   if(h.tick===event.atTick)h.walls.add(event.cell);
   return receipt;
  };
  return h;
 };
 return adapted;
}
function audit(log){
 const deltas={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]};
 for(let i=0;i<log.length;i++){
  const r=log[i];
  if(i!==r.tick||!r.menu.includes(r.action)||(i&&!same(log[i-1].after,r.before)))throw Error("receipt sequence/menu");
  const d=deltas[r.action],expected=d?[r.before[0]+d[0],r.before[1]+d[1]]:r.before;
  if(!same(r.after,expected))throw Error("receipt displacement");
  if(r.action==="STOP"&&r.result!=="stopped"||r.action==="ERASE"&&r.after_material!==null||
     r.action==="PAINT"&&![1,2].includes(r.after_material))throw Error("receipt material");
 }
 return log.length;
}
function replay(c,run,seed,hor,cap,pre,first,second,event){
 const z=new c.Host(seed,hor,"dev"),k="2,1",logs=run.host.log;
 for(const cell of EXTRA)z.walls.add(cell);
 if(logs.length>cap)throw Error("native censor exceeded");
 for(let i=0;i<logs.length;i++){
  if(i===pre){
   if(!z.paintedA||!z.hazard)throw Error("invalid intervention prefix");
   z.ink[k]=1;z.damage=!!first;if(first)delete z.ink[k];
  }
  const old=z.ink[k]??null,receipt=z.step(logs[i].action);
  if(!same(receipt,logs[i]))throw Error("receipt replay diverged "+seed+"/"+i);
  if(hor===128&&receipt.action==="PAINT"&&c.eq(receipt.after,c.TARGETS[4])&&z.secondHazard){
   if(second)delete z.ink[k];else if(old===null)delete z.ink[k];else z.ink[k]=old;
   z.secondDamage=!!second;
  }
  if(event&&z.tick===event.atTick)z.walls.add(event.cell);
 }
 if(z.tick!==run.host.tick||!c.eq(z.pos,run.host.pos)||!same(z.ink,run.host.ink)||
    !same([...z.walls].sort(),[...run.host.walls].sort()))throw Error("final replay diverged");
 return audit(logs);
}
function winning(rows){return rows.slice().sort((a,b)=>b.goal-a.goal||b.utility-a.utility||
 MODES.indexOf(a.mode)-MODES.indexOf(b.mode))[0].mode}
function study(c,e,t,z,limit=N){
 const began=Date.now(),model=z.fit(c,e,t),prior=t.freeze(c,e);
 const results={},cohorts={},rows=[],negative=[],gallery=[],paired=[];
 let conditions=0,rollouts=0,receipts=0,rejected=0,actualClosures=0,unavailable=0;
 for(const condition of ["control","closure"]){
  results[condition]=Object.fromEntries(ARMS.map(a=>[a,{n:0,goals:0,material:0,utility:0,calls:0,
   checks:0,actions:Object.fromEntries(c.ACT.map(x=>[x,0]))}]));
  for(const [hor,cap] of CAPS){
   const ck=condition+"/"+hor+"/"+cap;
   cohorts[ck]=Object.fromEntries(ARMS.map(a=>[a,0]));
   for(let seed=START;seed<START+limit;seed++){
    const original=t.prefix(c,seed,hor,cap,true);
    if(!original.hazard||!original.paintedA)throw Error("invalid unsealed prefix");
    const event=condition==="closure"?chooseClosure(c,original):null;
    if(condition==="closure"&&!event)unavailable+=forks(hor).length;
    const features=e.publicFeatures(c,original,cap);
    const x=e.probs(model.weights,features),trained=["feasible","late","early","both"][x.indexOf(Math.max(...x))];
    const y=e.probs(prior.weights,features),frozen=e.CANDIDATES[y.indexOf(Math.max(...y))];
    for(const [first,second] of forks(hor)){
     const root=e.variant(original,first),twin=e.variant(original,1-first);
     if(!same(root.observation(),twin.observation())||
        !same(features,e.publicFeatures(c,twin,cap)))throw Error("hidden erasure leaked");
     const attempts={};
     for(const mode of MODES){
      const run=t.preempt(c,inject(root,event),cap,mode,second);
      receipts+=replay(c,run,seed,hor,cap,original.tick,first,second,event);
      rollouts++;
      if(!run.out.stopped||run.host.log.at(-1).action!=="STOP"||run.host.tick>cap)
       throw Error("terminal reservation/censor failed");
      attempts[mode]=run;
     }
     conditions++;
     if(event&&attempts.feasible.host.walls.has(event.cell))actualClosures++;
     const hindsight=winning(MODES.map(mode=>({mode,goal:attempts[mode].out.terminalGoal,
      utility:attempts[mode].out.utility})));
     const selected={guard:"feasible",student021:trained,frozen018:frozen,hindsight};
     const outcomes={};
     for(const arm of ARMS){
      const mode=selected[arm],out=attempts[mode].out,s=results[condition][arm];
      s.n++;s.goals+=out.terminalGoal;s.material+=out.goal;s.utility+=out.utility;
      s.calls+=out.calls;s.checks+=out.checks;cohorts[ck][arm]+=out.terminalGoal;
      for(const action of c.ACT)s.actions[action]+=out.counts[action];
      outcomes[arm]={mode,goal:out.terminalGoal,utility:out.utility,
       calls:out.calls,receiptHash:out.hash};
     }
     const row={condition,seed,hor,cap,first,second,preTick:original.tick,event,
      trained,frozen,hindsight,outcomes};
     rows.push(row);
     if(outcomes.student021.goal<outcomes.guard.goal||
        outcomes.student021.goal<outcomes.hindsight.goal||
        outcomes.student021.utility<outcomes.guard.utility){
      negative.push(row);
      if(gallery.length<18&&outcomes.student021.goal<outcomes.hindsight.goal){
       for(const mode of new Set(["feasible",trained,hindsight]))
        if(gallery.length<18)gallery.push({condition,seed,hor,cap,first,second,
         mode,goal:attempts[mode].out.terminalGoal,positions:attempts[mode].out.positions});
      }
     }
    }
   }
  }
 }
 const base=new Map(rows.filter(r=>r.condition==="control").map(r=>
  [[r.seed,r.hor,r.cap,r.first,r.second].join("/"),r]));
 let changedReceipts=0;
 for(const row of rows.filter(r=>r.condition==="closure")){
  const match=base.get([row.seed,row.hor,row.cap,row.first,row.second].join("/"));
  if(!match)throw Error("unpaired closure case");
  for(const arm of ARMS)changedReceipts+=+(match.outcomes[arm].receiptHash!==row.outcomes[arm].receiptHash);
  if(match.outcomes.guard.goal!==row.outcomes.guard.goal||
     match.outcomes.student021.goal!==row.outcomes.student021.goal)
   paired.push({seed:row.seed,hor:row.hor,cap:row.cap,first:row.first,second:row.second,
    event:row.event,control:{guard:match.outcomes.guard.goal,student:match.outcomes.student021.goal},
    closure:{guard:row.outcomes.guard.goal,student:row.outcomes.student021.goal}});
 }
 const good={tick:0,action:"E",menu:["E"],before:[1,1],after:[2,1],
  before_material:null,after_material:null,result:"ok"};
 for(const bad of [{...good,action:"TELEPORT"},{...good,after:[9,9]},
  {...good,tick:2},{...good,menu:["STOP"]}])try{audit([bad])}catch{rejected++}
 if(conditions!==24*limit||rollouts!==conditions*MODES.length||receipts===0||rejected!==4)
  throw Error("campaign incomplete "+JSON.stringify({conditions,rollouts,receipts,rejected}));
 for(const group of Object.values(results))for(const s of Object.values(group))
  s.meanUtility=+(s.utility/s.n).toFixed(3);
 return{experiment:"ROUTE_INVALIDATION_022",sources:{dependencyGitBlobs:BLOBS,
  TRAIN:"021 training seeds 120100-120169",DEV:"unsealed shifted 150100-150149",
  structuralHELDOUT:"unopened"},limitations:"Deterministic synthetic route closure is a research intervention, not the native CyberSand host ABI. Same-source host replay and separate receipt checks do not constitute native acceptance. STOP is synthetic terminal accounting.",
  training:{updates:model.updates,labelDistribution:model.labels,weights:model.weights,
   frozen018Weights:prior.weights,trainRollouts:model.rollouts},
  run:{uniqueSeedClusters:limit,horizonPrefixSituations:4*limit,pairedConditions:conditions,
   policyRollouts:rollouts,sourceReplayedReceipts:receipts,rejectedMalformedReceipts:rejected,
   actualClosures,unavailableClosureCases:unavailable,changedPairedArmReceipts:changedReceipts,
   changedPairedGoalCases:paired.length,v8WallMs:Date.now()-began},
  results,cohorts,pairedEffects:paired,negativeCases:negative,caseRows:rows,gallery};
}
function gallerySVG(rows){
 const w=1200,h=Math.max(164,Math.ceil(rows.length/3)*172),s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'"><rect width="100%" height="100%" fill="#172631"/>'];
 for(let i=0;i<rows.length;i++){
  const r=rows[i],x=i%3*400,y=Math.floor(i/3)*172;
  s.push('<rect x="'+(x+4)+'" y="'+(y+4)+'" width="392" height="164" fill="#293c49" stroke="#63818d"/>');
  s.push('<polyline fill="none" stroke-width="2" stroke="'+(r.goal?"#91e2c2":"#efaa96")+'" points="'+r.positions.map(p=>(x+17+p[0]*24)+","+(y+16+p[1]*13)).join(" ")+'"/>');
  s.push('<text x="'+(x+10)+'" y="'+(y+160)+'" fill="white" font-family="monospace" font-size="11">'+r.condition+" "+r.seed+" "+r.hor+"/"+r.cap+" "+r.mode+"="+r.goal+'</text>');
 }
 return s.concat("</svg>").join("\n")+"\n";
}
if(typeof module!=="undefined")module.exports={study,chooseClosure,replay,audit,gallerySVG};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),vm=require("vm"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))
  throw Error("Explicit opt-in: node route_invalidation_022.js /absolute/output_dir");
 function load(i,expr){
  const buf=fs.readFileSync(path.join(__dirname,FILES[i]));
  const sha=crypto.createHash("sha1").update("blob "+buf.length+"\0").update(buf).digest("hex");
  if(sha!==BLOBS[i])throw Error("dependency blob drift "+FILES[i]);
  return vm.runInNewContext(buf.toString("utf8")+"\n;("+expr+")",{});
 }
 const c=load(0,"{Host,prefix,move,correct,bfs,TARGETS,ACT,DIR,eq,assertReceipts}");
 const e=load(1,"{variant,runPolicy,publicFeatures,train,probs,CANDIDATES}");
 const t=load(2,"{preempt,replay,freeze,prefix}");
 const z=load(3,"{fit,audit}");
 const r=study(c,e,t,z);
 r.sources.scriptSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 r.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 fs.mkdirSync(process.argv[2],{recursive:true});
 fs.writeFileSync(path.join(process.argv[2],"ROUTE_INVALIDATION_022_EVIDENCE.json"),JSON.stringify(r,null,2)+"\n");
 fs.writeFileSync(path.join(process.argv[2],"ROUTE_INVALIDATION_022_PATHS.svg"),gallerySVG(r.gallery));
 console.log(JSON.stringify({run:r.run,results:r.results,cohorts:r.cohorts},null,2));
}
