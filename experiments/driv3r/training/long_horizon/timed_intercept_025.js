"use strict";
// TIMED-INTERCEPT-025: explicit opt-in, fresh unsealed synthetic world transfer.
// No game-side writes, no HELDOUT, no auto CPU training. Synthetic STOP != native ABI.
const FILES=["physical_inspection_016.js","multipulse_physical_018.js","stop_reservation_020.js","twin_supervision_021.js","intercept_blockade_023.js"];
const PINS=["45a426f893203f93fa9a3972d63bd4a110ceaa3d","4e7f4b4b20d4db42d601961a7689f3729477aaf6","51b43ff4f31299f9ed0b0ff8a9a559b1e9a0a79d","126051579797450227700b381f5f3524f29d2eb5","6c4964cd35f327370ae1775439dfc0f1a8bca916"];
const EXPRS=["({Host,prefix,move,correct,bfs,TARGETS,ACT,DIR,eq,assertReceipts})","({variant,publicFeatures,probs,train})","({preempt,prefix})","({fit,audit})","({eventSites,replay,svgGallery})"];
const CAPS=[[36,36],[72,64],[128,64],[128,128]],MODES=["feasible","early","late","both"];
const SCENARIOS=["control","early_one","late_one","staggered_two","late_two"];
const forks=h=>h===128?[[0,0],[0,1],[1,0],[1,1]]:[[0,0],[1,0]];
function intervene(root,events){
 const h=root.clone(),base=h.clone.bind(h);
 h.clone=function(){const z=base(),prior=z.step.bind(z);
  z.step=function(a){const r=prior(a);for(const ev of events)
   if(z.tick===ev.atTick&&ev.cell!==z.pos.join(",")&&!z.walls.has(ev.cell))
    z.walls.add(ev.cell);return r;};return z;};return h;
}
function eventPlan(sites,root,name){
 const at=(j,delay)=>sites[j]?{cell:sites[j].cell,atTick:root.tick+delay,pathIndex:sites[j].pathIndex}:null;
 return ({control:[],early_one:[at(0,1)],late_one:[at(0,7)],
  staggered_two:[at(0,1),at(1,7)],late_two:[at(0,7),at(1,12)]})[name].filter(Boolean);
}
function study(c,e,t,z,i,N=20){
 const started=Date.now(),model=z.fit(c,e,t),aggregates={},cohorts={},rows=[],negative=[],gallery=[];
 let attempts=0,receipts=0,independent=0,installed=0,rejections=0;
 for(const scenario of SCENARIOS){
  aggregates[scenario]=Object.fromEntries(["guard","student","early","late","both","oracle"].map(a=>
   [a,{n:0,terminal:0,material:0,planner:0,checks:0,utility:0,actions:Object.fromEntries(c.ACT.map(x=>[x,0]))}]));
  for(const [h,cap] of CAPS){
   const ck=scenario+"/"+h+"/"+cap;cohorts[ck]=Object.fromEntries(Object.keys(aggregates[scenario]).map(a=>[a,0]));
   for(let seed=180100;seed<180100+N;seed++){
    const root=t.prefix(c,seed,h,cap,true);
    if(!root.hazard||!root.paintedA)throw Error("unqualified prefix "+seed);
    const sites=i.eventSites(c,e,t,root,cap),events=eventPlan(sites,root,scenario);
    const feature=e.publicFeatures(c,root,cap),score=e.probs(model.weights,feature);
    const chosen=["feasible","late","early","both"][score.indexOf(Math.max(...score))];
    for(const [first,second] of forks(h)){
     const world=e.variant(root,first);
     if(JSON.stringify(world.observation())!==JSON.stringify(e.variant(root,1-first).observation()))
      throw Error("offscreen damage leaked");
     const outcomes={};
     for(const mode of MODES){
      const rr=t.preempt(c,intervene(world,events),cap,mode,second);
      const audited=i.replay(c,rr,seed,h,cap,root.tick,first,second,events);
      const checked=z.audit(rr.host.log);receipts+=audited.count;installed+=audited.installed;
      independent+=checked;attempts++;
      if(rr.host.tick>cap||rr.out.stopped!==1||rr.host.log.at(-1).action!=="STOP")
       throw Error("STOP/censor violation");
      outcomes[mode]=rr.out;
     }
     const oracle=MODES.slice().sort((a,b)=>outcomes[b].terminalGoal-outcomes[a].terminalGoal||
      outcomes[b].utility-outcomes[a].utility||MODES.indexOf(a)-MODES.indexOf(b))[0];
     const decisions={guard:"feasible",student:chosen,early:"early",late:"late",both:"both",oracle};
     const compact={};
     for(const [arm,mode] of Object.entries(decisions)){
      const o=outcomes[mode],a=aggregates[scenario][arm];
      a.n++;a.terminal+=o.terminalGoal;a.material+=o.goal;a.planner+=o.calls;
      a.checks+=o.checks;a.utility+=o.utility;cohorts[ck][arm]+=o.terminalGoal;
      for(const action of c.ACT)a.actions[action]+=o.counts[action];
      compact[arm]={mode,terminal:o.terminalGoal,material:o.goal,planner:o.calls,receipt:o.hash};
     }
     const row={scenario,seed,h,cap,first,second,events,student:chosen,outcomes:compact};rows.push(row);
     const noteworthy=compact.student.terminal<compact.oracle.terminal||
       compact.student.planner>compact.guard.planner||
       compact.guard.terminal<compact.oracle.terminal;
     if(noteworthy){
      if(negative.length<200)negative.push(row);
      if(scenario!=="control"&&gallery.length<24&&compact.student.terminal<compact.oracle.terminal)
       for(const mode of new Set(["feasible",chosen,oracle]))if(gallery.length<24)
        gallery.push({condition:scenario,seed,h,cap,first,second,mode,
         goal:outcomes[mode].terminalGoal,positions:outcomes[mode].positions});
     }
    }
   }
  }
 }
 const expected=N*12*SCENARIOS.length;
 if(rows.length!==expected||attempts!==expected*4||receipts!==independent)
  throw Error("incomplete campaign counts");
 const index=new Map(rows.filter(r=>r.scenario==="control").map(r=>
  [[r.seed,r.h,r.cap,r.first,r.second].join("/"),r]));
 const transfer={};
 for(const scenario of SCENARIOS.slice(1)){
  const s={matched:0,scheduled:0,receiptChangesGuard:0,receiptChangesStudent:0,
   terminalDeltaGuard:0,terminalDeltaStudent:0,studentBeatsGuard:0,studentLosesGuard:0};
  for(const r of rows.filter(r=>r.scenario===scenario)){
   const b=index.get([r.seed,r.h,r.cap,r.first,r.second].join("/"));
   if(!b)throw Error("missing paired control");
   s.matched++;s.scheduled+=r.events.length;
   s.receiptChangesGuard+=+(r.outcomes.guard.receipt!==b.outcomes.guard.receipt);
   s.receiptChangesStudent+=+(r.outcomes.student.receipt!==b.outcomes.student.receipt);
   s.terminalDeltaGuard+=r.outcomes.guard.terminal-b.outcomes.guard.terminal;
   s.terminalDeltaStudent+=r.outcomes.student.terminal-b.outcomes.student.terminal;
   s.studentBeatsGuard+=+(r.outcomes.student.terminal>r.outcomes.guard.terminal);
   s.studentLosesGuard+=+(r.outcomes.student.terminal<r.outcomes.guard.terminal);
  }transfer[scenario]=s;
 }
 const good={tick:0,action:"E",menu:["E"],before:[1,1],after:[2,1],
  before_material:null,after_material:null,result:"ok"};
 for(const bad of [{...good,action:"TELEPORT"},{...good,tick:5},
   {...good,after:[50,50]},{...good,menu:["STOP"]}])
  try{z.audit([bad]);}catch{rejections++;}
 if(rejections!==4)throw Error("corruption acceptance");
 for(const set of Object.values(aggregates))for(const a of Object.values(set))
  a.meanUtility=+(a.utility/a.n).toFixed(3);
 return{experiment:"TIMED_INTERCEPT_025",
  provenance:{dependencyGitBlobs:PINS,TRAIN:"frozen 021 TRAIN 120100..120169",
   DEV:"new unsealed 180100.."+(180099+N),structuralHELDOUT:"UNOPENED"},
  boundaries:"Synthetic future-path-guided intervention placement is stress design only, invisible to policy. Same-source synthetic host replay, independent receipt-field checks, not native acceptance; synthetic STOP hypothetical.",
  training:{model:"reconstructed 021 four-class softmax, NOT an SLM checkpoint",
   updates:model.updates,examples:model.samples,trainRollouts:model.rollouts,
   labels:model.labels,weights:model.weights},
  run:{seeds:N,conditions:rows.length,rollouts:attempts,sourceReplayedReceipts:receipts,
   independentlyCheckedReceiptFields:independent,installedWalls:installed,
   corruptionRejections:rejections,retainedNegativeCases:negative.length,
   elapsedV8Ms:Date.now()-started,nodeRSS:"unavailable in V8 study",
   fullPolicyLatency:"unavailable in V8 study"},aggregates,cohorts,transfer,negative,rows,gallery};
}
if(typeof module!=="undefined")module.exports={study,eventPlan,intervene};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),path=require("path"),crypto=require("crypto"),vm=require("vm");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))
  throw Error("Explicit opt-in: node timed_intercept_025.js /absolute/output_directory");
 const ds=FILES.map((file,j)=>{const bytes=fs.readFileSync(path.join(__dirname,file));
  const gitsha=crypto.createHash("sha1").update("blob "+bytes.length+"\0").update(bytes).digest("hex");
  if(gitsha!==PINS[j])throw Error("pinned dependency drift "+file);
  return vm.runInNewContext(bytes.toString("utf8")+"\n;"+EXPRS[j],{});
 });
 const r=study(...ds);r.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 r.run.nodeRSSBytes=process.memoryUsage().rss;
 r.provenance.sourceSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 fs.mkdirSync(process.argv[2],{recursive:true});
 fs.writeFileSync(path.join(process.argv[2],"TIMED_INTERCEPT_025_EVIDENCE.json"),JSON.stringify(r,null,2)+"\n");
 fs.writeFileSync(path.join(process.argv[2],"TIMED_INTERCEPT_025_PATHS.svg"),ds[4].svgGallery(r.gallery));
 console.log(JSON.stringify({run:r.run,aggregates:r.aggregates,transfer:r.transfer},null,2));
}