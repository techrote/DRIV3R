"use strict";
// TWIN-SUPERVISION-021: bounded opt-in CPU training on paired synthetic worlds.
// Uses existing physical host, 018 softmax and 020 STOP-reservation controls.
// HELDOUT untouched. Synthetic STOP is not a native CyberSand acceptance rule.
const FILES=["physical_inspection_016.js","multipulse_physical_018.js","stop_reservation_020.js"];
const PINS=["45a426f893203f93fa9a3972d63bd4a110ceaa3d","4e7f4b4b20d4db42d601961a7689f3729477aaf6","51b43ff4f31299f9ed0b0ff8a9a559b1e9a0a79d"];
const CAPS=[[36,36],[72,64],[128,64],[128,128]];
const MODES=["feasible","late","early","both"];
const ARMS=[...MODES,"frozen","trained","twin_oracle","hidden_oracle"];
const forks=h=>h===128?[[0,0],[0,1],[1,0],[1,1]]:[[0,0],[1,0]];
function winner(xs){return xs.slice().sort((a,b)=>b.goals-a.goals||b.utility-a.utility||MODES.indexOf(a.mode)-MODES.indexOf(b.mode))[0].mode}
function audit(log){
 const dir={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]},eq=(a,b)=>a[0]===b[0]&&a[1]===b[1];
 for(let i=0;i<log.length;i++){
  const r=log[i];
  if(r.tick!==i||!r.menu.includes(r.action)||i&&!eq(log[i-1].after,r.before))throw Error("receipt sequence/menu");
  if(r.action in dir){let d=dir[r.action];if(!eq(r.after,[r.before[0]+d[0],r.before[1]+d[1]]))throw Error("receipt motion")}
  else if(!eq(r.before,r.after))throw Error("receipt stationary action");
  if(r.action==="STOP"&&r.result!=="stopped"||r.action==="ERASE"&&r.after_material!==null||
     r.action==="PAINT"&&![1,2].includes(r.after_material))throw Error("receipt result");
 }
 return log.length;
}
function fit(c,e,t){
 let samples=[],labels=Object.fromEntries(MODES.map(k=>[k,0])),rollouts=0;
 for(let seed=120100;seed<120170;seed++)for(const [hor,cap] of CAPS){
  const root=c.prefix(seed,hor,"train",cap);
  if(!root.paintedA||!root.hazard)throw Error("invalid TRAIN prefix");
  const options=MODES.map(mode=>{let goals=0,utility=0;
   for(const [first,second] of forks(hor)){const r=t.preempt(c,e.variant(root,first),cap,mode,second).out;
    goals+=r.terminalGoal;utility+=r.utility;rollouts++}
   return{mode,goals,utility}});
  const label=winner(options);labels[label]++;
  samples.push({x:e.publicFeatures(c,root,cap),label:MODES.indexOf(label)});
 }
 if(samples.length!==280||rollouts!==3360)throw Error("TRAIN sample drift");
 return{weights:e.train(samples),labels,rollouts,samples:samples.length,updates:1100};
}
function study(c,e,t){
 const started=Date.now(),fitModel=fit(c,e,t),frozen=t.freeze(c,e);
 const summary={},cohorts={},negative=[],cases=[],paths=[];
 let conditions=0,rollouts=0,replayed=0,independent=0,corruptions=0;
 for(const [group,start,shift] of [["fresh",130100,false],["shift",140100,true]]){
  summary[group]=Object.fromEntries(ARMS.map(a=>[a,{count:0,goals:0,materials:0,utility:0,calls:0,checks:0,actions:Object.fromEntries(c.ACT.map(x=>[x,0]))}]));
  for(const [hor,cap] of CAPS){
   const cohort=group+"/"+hor+"/"+cap;cohorts[cohort]=Object.fromEntries(ARMS.map(a=>[a,0]));
   for(let seed=start;seed<start+50;seed++){
    const root=t.prefix(c,seed,hor,cap,shift);
    if(!root.paintedA||!root.hazard)throw Error("invalid DEVELOPMENT prefix");
    const features=e.publicFeatures(c,root,cap);
    const v=e.probs(fitModel.weights,features),trained=MODES[v.indexOf(Math.max(...v))];
    const v0=e.probs(frozen.weights,features),frozenMode=e.CANDIDATES[v0.indexOf(Math.max(...v0))];
    const score=Object.fromEntries(MODES.map(mode=>[mode,{mode,goals:0,utility:0}]));
    const tested=[];
    for(const [first,second] of forks(hor)){
     const rootA=e.variant(root,first),rootB=e.variant(root,1-first);
     if(JSON.stringify(rootA.observation())!==JSON.stringify(rootB.observation())||
        JSON.stringify(features)!==JSON.stringify(e.publicFeatures(c,rootA,cap)))throw Error("hidden damage leaked");
     const runs={};
     for(const mode of new Set([...MODES,frozenMode])){
      const rr=t.preempt(c,rootA,cap,mode,second);
      replayed+=t.replay(c,rr,seed,hor,cap,root.tick,first,second,shift);
      independent+=audit(rr.host.log);
      if(rr.host.tick>cap||rr.out.stopped!==1||rr.host.log.at(-1).action!=="STOP")throw Error("STOP/cap violation");
      runs[mode]=rr;rollouts++;
      if(mode in score){score[mode].goals+=rr.out.terminalGoal;score[mode].utility+=rr.out.utility}
     }
     tested.push({first,second,runs});
    }
    const twin=winner(Object.values(score)); // ex-post paired-fork supervisor, NOT observable signal.
    for(const {first,second,runs} of tested){
     conditions++;
     const hidden=winner(MODES.map(mode=>({mode,goals:runs[mode].out.terminalGoal,utility:runs[mode].out.utility})));
     const sel={feasible:"feasible",late:"late",early:"early",both:"both",frozen:frozenMode,trained,twin_oracle:twin,hidden_oracle:hidden};
     const outcome={};
     for(const arm of ARMS){
      const mode=sel[arm],r=runs[mode].out,s=summary[group][arm];
      s.count++;s.goals+=r.terminalGoal;s.materials+=r.goal;s.utility+=r.utility;
      s.calls+=r.calls;s.checks+=r.checks;cohorts[cohort][arm]+=r.terminalGoal;
      for(const a of c.ACT)s.actions[a]+=r.counts[a];
      outcome[arm]={mode,goal:r.terminalGoal,utility:r.utility,calls:r.calls,hash:r.hash};
     }
     cases.push({group,seed,hor,cap,first,second,trained,twin,hidden,
      outcomes:Object.fromEntries(MODES.map(m=>[m,outcome[m]]))});
     const fail=outcome.trained.goal<outcome.feasible.goal||
      outcome.trained.goal<outcome.twin_oracle.goal||
      outcome.twin_oracle.goal<outcome.hidden_oracle.goal||
      outcome.trained.utility<outcome.feasible.utility;
     if(fail){
      negative.push({group,seed,hor,cap,first,second,trained,twin,hidden,outcome});
      if(paths.length<24&&(outcome.trained.goal<outcome.twin_oracle.goal||
          outcome.twin_oracle.goal<outcome.hidden_oracle.goal)){
       for(const mode of new Set(["feasible",trained,twin]))paths.push({group,seed,hor,cap,first,second,mode,goal:runs[mode].out.terminalGoal,positions:runs[mode].out.positions});
      }
     }
    }
   }
  }
 }
 const bad=new c.Host(7,36,"train");bad.step("E");
 for(const x of [{...bad.log[0],action:"TELEPORT"},{...bad.log[0],after:[999,999]},
  {...bad.log[0],tick:3},{...bad.log[0],menu:["STOP"]}]){try{audit([x])}catch{corruptions++}}
 if(conditions!==1200||rollouts<4800||replayed<150000||independent!==replayed||corruptions!==4)
  throw Error("campaign incomplete "+JSON.stringify({conditions,rollouts,replayed,independent,corruptions}));
 for(const block of Object.values(summary))for(const s of Object.values(block))s.meanUtility=+(s.utility/s.count).toFixed(3);
 return{experiment:"TWIN_SUPERVISION_021",
  provenance:{dependencyGitBlobs:PINS,TRAIN:"120100..120169; frozen 60100..60169",DEV:"130100..130149 + shifted 140100..140149",structuralHELDOUT:"unopened"},
  limits:"Counterfactual twin/hidden supervisors are ex-post hindsight bounds; not legal host observations. Same-source synthetic replay and independent receipt-field checks, NOT independent world replay or native CyberSand. STOP terminal requirement hypothetical.",
  training:{model:"four-class softmax, no neural SLM checkpoint",updates:fitModel.updates,
   labels:fitModel.labels,examples:fitModel.samples,trainingRollouts:fitModel.rollouts,
   weights:fitModel.weights,frozenWeights:frozen.weights},
  run:{conditions,uniqueRollouts:rollouts,replayedReceipts:replayed,independentReceiptFields:independent,
   corruptionRejections:corruptions,negativeCaseRecords:negative.length,v8WallMs:Date.now()-started},
  summary,cohorts,negativeCases:negative,caseRows:cases,paths};
}
function gallery(rows){
 const w=1200,h=Math.ceil(rows.length/3)*164,s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'"><rect width="100%" height="100%" fill="#122431"/>'];
 for(let i=0;i<rows.length;i++){
  const r=rows[i],x=i%3*400,y=Math.floor(i/3)*164;
  const pts=r.positions.map(p=>(x+12+p[0]*26)+","+(y+17+p[1]*13)).join(" ");
  s.push('<rect x="'+(x+3)+'" y="'+(y+3)+'" width="394" height="158" fill="#253b49" stroke="#698899"/>');
  s.push('<polyline points="'+pts+'" fill="none" stroke="'+(r.goal?"#92dfbc":"#f0a0a0")+'" stroke-width="2"/>');
  s.push('<text x="'+(x+9)+'" y="'+(y+154)+'" font-family="monospace" font-size="11" fill="white">'+r.group+' '+r.seed+' '+r.hor+'/'+r.cap+' '+r.mode+' goal='+r.goal+'</text>');
 }
 return s.concat('</svg>').join("\n")+"\n";
}
if(typeof module!=="undefined")module.exports={study,gallery,audit};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),vm=require("vm"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))throw Error("Explicit opt-in only: node twin_supervision_021.js /absolute/output");
 function load(i,expr){
  const bytes=fs.readFileSync(path.join(__dirname,FILES[i]));
  const blob=crypto.createHash("sha1").update("blob "+bytes.length+"\0").update(bytes).digest("hex");
  if(blob!==PINS[i])throw Error("dependency blob drift "+FILES[i]);
  return vm.runInNewContext(bytes.toString("utf8")+"\n;("+expr+")",{});
 }
 const c=load(0,"{Host,prefix,move,correct,bfs,TARGETS,ACT,DIR,eq,assertReceipts}");
 const e=load(1,"{variant,runPolicy,publicFeatures,train,probs,CANDIDATES}");
 const t=load(2,"{preempt,replay,freeze,prefix}");
 const result=study(c,e,t);
 result.provenance.scriptSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 result.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 result.run.nodeRSSBytes=process.memoryUsage().rss;
 const dir=process.argv[2];fs.mkdirSync(dir,{recursive:true});
 fs.writeFileSync(path.join(dir,"TWIN_SUPERVISION_021_EVIDENCE.json"),JSON.stringify(result,null,2)+"\n");
 fs.writeFileSync(path.join(dir,"TWIN_SUPERVISION_021_PATHS.svg"),gallery(result.paths));
 console.log(JSON.stringify({run:result.run,summary:result.summary,cohorts:result.cohorts},null,2));
}