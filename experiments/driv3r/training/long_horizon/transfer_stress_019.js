"use strict";
// TRANSFER-STRESS-019: opt-in, bounded, synthetic, frozen 018 policy stress test.
// No structural HELDOUT, speculative notifications, production files or automatic training.
const H_BLOB="45a426f893203f93fa9a3972d63bd4a110ceaa3d";
const E_BLOB="4e7f4b4b20d4db42d601961a7689f3729477aaf6";
const HC=[[36,36],[72,64],[128,64],[128,128]];
const NAMES=["feasible","censor_aware","student","route_budget","early","late","never"];
const EXTRA=["5,4","9,4","6,6"];
function freeze018(c,e){
 const examples=[],labels={never:0,early:0,feasible:0,both:0};
 for(let seed=60100;seed<60170;seed++)for(const [hor,cap] of HC){
  const z=c.prefix(seed,hor,"train",cap);
  const ranking=e.CANDIDATES.map(mode=>{
   let g=0,u=0;
   for(const a of [0,1])for(const b of hor===128?[0,1]:[0]){
    const r=e.runPolicy(c,e.variant(z,a),cap,mode,b).out;
    g+=r.terminalGoal;u+=r.utility;
   }
   return {mode,g,u};
  }).sort((x,y)=>y.g-x.g||y.u-x.u||
      e.CANDIDATES.indexOf(x.mode)-e.CANDIDATES.indexOf(y.mode));
  labels[ranking[0].mode]++;
  examples.push({x:e.publicFeatures(c,z,cap),label:e.CANDIDATES.indexOf(ranking[0].mode)});
 }
 if(examples.length!==280||JSON.stringify(labels)!==
   JSON.stringify({never:86,early:124,feasible:70,both:0}))
  throw Error("frozen training counts drift");
 return {weights:e.train(examples),labels};
}
function makePrefix(c,seed,hor,cap,stress){
 if(!stress)return c.prefix(seed,hor,"dev",cap);
 const z=new c.Host(seed,hor,"dev");
 for(const k of EXTRA)z.walls.add(k);
 for(let i=0;i<(hor===36?2:3);i++)
  if(!c.move(z,c.TARGETS[i],cap)||!c.correct(z,cap))break;
 return z;
}
function routeChoice(c,z,cap){
 let p=z.pos,cost=0;
 for(const target of (z.horizon===128?c.TARGETS.slice(3,5):[])){
  const path=c.bfs(p,target,z.walls);
  if(!path.length&&!c.eq(p,target))return "feasible";
  cost+=path.length+1;p=target;
 }
 const back=c.bfs(p,c.TARGETS[0],z.walls);
 if(!back.length&&!c.eq(p,c.TARGETS[0]))return "feasible";
 return cost+back.length+2>cap-z.tick?"early":"feasible";
}
function exactReplay(c,run,seed,hor,cap,pre,a,b,stress){
 const z=new c.Host(seed,hor,"dev"),k="2,1";
 if(stress)for(const w of EXTRA)z.walls.add(w);
 const logs=run.host.log;
 if(logs.length>cap)throw Error("cap exceeded");
 for(let i=0;i<logs.length;i++){
  if(i===pre){
   if(!z.paintedA||!z.hazard)throw Error("intervention precondition");
   z.ink[k]=1;z.damage=!!a;if(a)delete z.ink[k];
  }
  const old=z.ink[k]??null,got=z.step(logs[i].action);
  if(JSON.stringify(got)!==JSON.stringify(logs[i]))
   throw Error("receipt mismatch "+seed+" "+hor+" tick "+i);
  if(hor===128&&got.action==="PAINT"&&
     c.eq(got.after,c.TARGETS[4])&&z.secondHazard){
   if(b)delete z.ink[k];else if(old===null)delete z.ink[k];
   else z.ink[k]=old;
   z.secondDamage=!!b;
  }
 }
 if(z.tick!==run.host.tick||!c.eq(z.pos,run.host.pos)||
    JSON.stringify(z.ink)!==JSON.stringify(run.host.ink)||
    z.secondDamage!==run.host.secondDamage)
  throw Error("final state replay drift");
 return c.assertReceipts(logs);
}
function science(c,e){
 const started=Date.now(),frozen=freeze018(c,e),groups=[
  {name:"fresh_dev",first:80100,n:80,stress:false},
  {name:"barrier_shift",first:90100,n:80,stress:true}];
 let n=0,rollouts=0,receipts=0,corrupt=0;
 const summary={},cohorts={},paired=[],negatives=[],paths=[];
 for(const g of groups){
  summary[g.name]={};
  for(const policy of NAMES)summary[g.name][policy]={
   episodes:0,terminal:0,material:0,nonterminalMaterial:0,
   calls:0,checks:0,utility:0,actions:Object.fromEntries(c.ACT.map(a=>[a,0]))};
  for(const [hor,cap] of HC){
   const ck=g.name+"/"+hor+"/"+cap;cohorts[ck]={events:0};
   for(const policy of NAMES)cohorts[ck][policy]={terminal:0,material:0,calls:0};
   for(let seed=g.first;seed<g.first+g.n;seed++){
    const pref=makePrefix(c,seed,hor,cap,g.stress);
    if(!pref.hazard||!pref.paintedA)throw Error("unqualified synthetic prefix");
    const p=e.probs(frozen.weights,e.publicFeatures(c,pref,cap));
    const chosen=e.CANDIDATES[p.indexOf(Math.max(...p))];
    const route=routeChoice(c,pref,cap);
    for(const first of [0,1])for(const second of hor===128?[0,1]:[0]){
     n++;cohorts[ck].events++;
     const root=e.variant(pref,first),counter=e.variant(pref,1-first);
     if(JSON.stringify(root.observation())!==JSON.stringify(counter.observation()))
      throw Error("offscreen material leaked into observation");
     const result={};
     for(const policy of NAMES){
      const mode=policy==="student"?chosen:policy==="route_budget"?route:
       policy==="censor_aware"?(hor===128&&cap===64?"early":"feasible"):policy;
      const run=e.runPolicy(c,root,cap,mode,second),r=run.out;
      receipts+=exactReplay(c,run,seed,hor,cap,pref.tick,first,second,g.stress);
      rollouts++;
      const q=summary[g.name][policy],co=cohorts[ck][policy];
      q.episodes++;q.terminal+=r.terminalGoal;q.material+=r.goal;
      q.nonterminalMaterial+=r.goal&&!r.stopped?1:0;
      q.calls+=r.calls;q.checks+=r.checks;q.utility+=r.utility;
      co.terminal+=r.terminalGoal;co.material+=r.goal;co.calls+=r.calls;
      for(const act of c.ACT)q.actions[act]+=r.counts[act];
      result[policy]={terminal:r.terminalGoal,material:r.goal,
       utility:+r.utility.toFixed(3),calls:r.calls,receiptHash:r.hash};
      if([80103,80122,80147,90103,90122,90147].includes(seed)&&
       hor===128&&cap===64&&first===1&&second===1&&
       ["feasible","censor_aware","student","route_budget"].includes(policy))
       paths.push({group:g.name,seed,hor,cap,policy,goal:r.terminalGoal,positions:r.positions});
     }
     const row={group:g.name,seed,hor,cap,first,second,student:chosen,route,result};
     paired.push(row);
     if(["student","censor_aware","route_budget"].some(t=>
       result[t].terminal<result.feasible.terminal))negatives.push(row);
    }
   }
  }
 }
 const bad=new c.Host(17,36,"train");bad.step("E");
 for(const t of [[{...bad.log[0],action:"TELEPORT"}],
     [{...bad.log[0],after:[999,999]}]]){
  let rejected=false;try{c.assertReceipts(t)}catch(err){rejected=true}
  if(!rejected)throw Error("bad receipt accepted");corrupt++;
 }
 if(n!==1920||rollouts!==13440||receipts<200000||corrupt!==2)
  throw Error("cohort incomplete: "+JSON.stringify({n,rollouts,receipts,corrupt}));
 const sample=e.publicFeatures(c,makePrefix(c,80100,128,64,false),64);
 let checksum=0;const start=Date.now();
 for(let j=0;j<100000;j++)checksum+=e.probs(frozen.weights,sample)[j%4];
 const batchMs=Date.now()-start;
 for(const g of Object.values(summary))for(const s of Object.values(g))
  s.meanUtility=+(s.utility/s.episodes).toFixed(3);
 return {name:"TRANSFER_STRESS_019",provenance:{
  hostGitBlob:H_BLOB,source018GitBlob:E_BLOB,train:"018 TRAIN 60100-60169",
  dev:"80100-80179 and 90100-90179",structuralHeldout:"untouched"},
  qualification:"Exact synthetic source-host replay including physical interventions. Not independent/native game acceptance.",
  run:{groups,horizonCaps:HC,trainOptimizerUpdates:1100,trainSamples:280,
   labelCounts:frozen.labels,conditions:n,rollouts,replayedReceipts:receipts,
   corruptionRejections:corrupt,wallMs:Date.now()-started,
   isolatedPredictionCount:100000,predictionBatchMs:batchMs,predictionChecksum:checksum,
   rss:"not available in this V8 run; Node CLI records peak and current RSS"},
  weights:frozen.weights,summary,cohorts,pairedNegativeCases:negatives,
  allPairedConditions:paired,paths};
}
function gallery(r){
 const subset=r.slice(0,24),w=1200,h=Math.ceil(subset.length/4)*160;
 const lines=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'"><rect width="100%" height="100%" fill="#15232e"/>'];
 for(let i=0;i<subset.length;i++){
  const x=i%4*300,y=Math.floor(i/4)*160,a=subset[i],
   pts=a.positions.map(p=>[x+10+p[0]*21,y+10+p[1]*13].join(",")).join(" ");
  lines.push('<rect x="'+(x+2)+'" y="'+(y+2)+'" width="296" height="156" fill="#253846" stroke="#7b8a98"/>');
  lines.push('<polyline points="'+pts+'" fill="none" stroke="'+(a.goal?"#8cdbb1":"#ef9992")+'" stroke-width="2"/>');
  lines.push('<text x="'+(x+6)+'" y="'+(y+151)+'" fill="#fff" font-family="monospace" font-size="10">'+
     a.group+' '+a.seed+' '+a.policy+' '+a.goal+'</text>');
 }
 return lines.concat("</svg>").join("\n")+"\n";
}
if(typeof module!=="undefined")module.exports={science,gallery};
if(typeof module!=="undefined"&&typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),vm=require("vm"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))
  throw Error("Manual CPU run only: node transfer_stress_019.js /absolute/output");
 const dir=__dirname;
 function load(file,sha,tail){
  const src=fs.readFileSync(path.join(dir,file)),blob=crypto.createHash("sha1")
    .update("blob "+src.length+"\0").update(src).digest("hex");
  if(blob!==sha)throw Error("source mismatch "+file);
  return vm.runInNewContext(src.toString("utf8")+tail,{});
 }
 const c=load("physical_inspection_016.js",H_BLOB,
  "\n;({Host,prefix,move,correct,bfs,TARGETS,ACT,eq,assertReceipts})");
 const e=load("multipulse_physical_018.js",E_BLOB,
  "\n;({variant,runPolicy,publicFeatures,train,probs,CANDIDATES})");
 const result=science(c,e);
 result.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 result.run.nodeRSSBytes=process.memoryUsage().rss;
 result.provenance.scriptSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 const d=process.argv[2];fs.mkdirSync(d,{recursive:true});
 fs.writeFileSync(path.join(d,"TRANSFER_STRESS_019_EVIDENCE.json"),JSON.stringify(result,null,2)+"\n");
 fs.writeFileSync(path.join(d,"TRANSFER_STRESS_019_PATHS.svg"),gallery(result.paths));
 console.log(JSON.stringify({run:result.run,summary:result.summary,cohorts:result.cohorts,
  negativeCases:result.pairedNegativeCases.length},null,2));
}
