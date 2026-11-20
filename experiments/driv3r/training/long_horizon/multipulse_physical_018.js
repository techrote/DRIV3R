"use strict";
// MULTIPULSE-PHYSICAL-018. Exact imported 016 host; bounded opt-in synthetic study.
// Structural HELDOUT excluded. No notifications or speculative CyberSand ABI fields.
// A material loss is physically observable only by walking back to A.
const fsRequiredOnlyOnExplicitRun=true;
const HOST_GIT_BLOB="45a426f893203f93fa9a3972d63bd4a110ceaa3d";
const POLICIES=["never","early","late","both","feasible","tight","censor_aware","student","postpulse_oracle"];
const CANDIDATES=["never","early","feasible","both"];
const CAPS=[[36,36],[72,64],[128,64],[128,128]];
const A=[2,1];
function publicFeatures(c,root,cap){
 const o=root.observation(),dist=c.bfs(o.pos,A,root.walls).length;
 return [1,+(root.horizon===36),+(root.horizon===72),+(root.horizon===128),
 +(cap===64),+(cap===128),o.remaining/128,(cap-o.tick)/128,
 o.tick/128,dist/30,root.walls.size/30,o.pos[0]/13,o.pos[1]/9];
}
function variant(root,first){
 const h=root.clone(),k="2,1";
 h.ink[k]=1; h.damage=!!first;
 if(first)delete h.ink[k];
 return h;
}
function runPolicy(c,root,cap,mode,second){
 const h=root.clone(),n={36:2,72:3,128:5}[h.horizon],k="2,1";
 let checks=0,lateApplied=false;
 function check(){
  if(h.tick>=cap)return;
  if(c.move(h,A,cap)){checks++;c.correct(h,cap)}
 }
 function guard(extra){
  let dist=c.bfs(h.pos,A,h.walls).length;
  if((dist>0||c.eq(h.pos,A))&&cap-h.tick>=dist+extra)check();
 }
 if(mode==="early"||mode==="both")check();
 for(let i=h.horizon===36?2:3;i<n;i++){
  if(!c.move(h,c.TARGETS[i],cap))break;
  const old=h.ink[k]??null;
  const good=c.correct(h,cap);
  if(h.horizon===128&&i===4&&h.secondHazard){
   if(second)delete h.ink[k];else if(old===null)delete h.ink[k];else h.ink[k]=old;
   h.secondDamage=!!second;lateApplied=true;
  }
  if(!good)break;
 }
 if(mode==="late"||mode==="both")check();
 if(mode==="feasible")guard(2);
 if(mode==="tight")guard(1);
 if(mode==="postpulse_oracle"&&h.ink[k]!==1)guard(2);
 if(h.tick<cap)h.step("STOP");
 const out=h.result(cap,checks);
 out.stopped=+(h.log.length>0&&h.log[h.log.length-1].action==="STOP");
 out.terminalGoal=+(out.goal&&out.stopped);
 return{host:h,out,lateApplied};
}
function replay(c,actual,seed,hor,cap,split,pre,first,second){
 const z=new c.Host(seed,hor,split),k="2,1",logs=actual.host.log;
 for(let i=0;i<logs.length;i++){
  if(i===pre){if(!z.paintedA||!z.hazard)throw Error("bad intervention prefix");z.ink[k]=1;z.damage=!!first;if(first)delete z.ink[k]}
  let rec=logs[i],old=z.ink[k]??null;
  const got=z.step(rec.action);
  if(JSON.stringify(got)!==JSON.stringify(rec))throw Error("source receipt mismatch "+seed+"/"+hor+" tick "+i);
  if(hor===128&&got.action==="PAINT"&&c.eq(got.after,c.TARGETS[4])&&z.secondHazard){
   if(second)delete z.ink[k];else if(old===null)delete z.ink[k];else z.ink[k]=old;
   z.secondDamage=!!second;
  }
 }
 if(logs.length!==z.tick||!c.eq(z.pos,actual.host.pos)||JSON.stringify(z.ink)!==JSON.stringify(actual.host.ink))throw Error("post-intervention replay divergent");
 return c.assertReceipts(logs);
}
function sigmoid(z){return 1/(1+Math.exp(-Math.max(-30,Math.min(30,z))))}
function probs(w,x){
 const z=w.map(row=>row.reduce((s,v,i)=>s+v*x[i],0)),mx=Math.max(...z),e=z.map(v=>Math.exp(v-mx)),total=e.reduce((s,v)=>s+v,0);
 return e.map(v=>v/total);
}
function train(examples){
 const w=Array.from({length:4},()=>Array(examples[0].x.length).fill(0));
 for(let step=0;step<1100;step++){
  const grad=w.map(row=>row.map(()=>0));
  for(let r of examples){const p=probs(w,r.x);for(let j=0;j<4;j++)for(let k=0;k<r.x.length;k++)grad[j][k]+=(p[j]-(j===r.label?1:0))*r.x[k]}
  let rate=.62/(1+step/600);
  for(let j=0;j<4;j++)for(let k=0;k<w[j].length;k++)w[j][k]-=rate*(grad[j][k]/examples.length+.0008*w[j][k]);
 }
 return w;
}
function summary(rows,policy,c){
 const subset=rows.filter(x=>x.policy===policy);
 return{records:subset.length,material_goals:subset.reduce((v,x)=>v+x.goal,0),terminal_goals:subset.reduce((v,x)=>v+x.terminalGoal,0),
 nonterminal_material_goals:subset.filter(x=>x.goal&&!x.stopped).length,
 mean_utility:+(subset.reduce((v,x)=>v+x.utility,0)/subset.length).toFixed(3),
 planner_calls:subset.reduce((v,x)=>v+x.calls,0),
 checks:subset.reduce((v,x)=>v+x.checks,0),
 actions:Object.fromEntries(c.ACT.map(a=>[a,subset.reduce((v,x)=>v+x.counts[a],0)]))};
}
function study(c){
 const begin=Date.now(),examples=[],trainCounts={never:0,early:0,feasible:0,both:0};
 const experiments=[[36,36],[72,64],[128,64],[128,128]];
 const possibilities=(hor)=>hor===128?[0,1]:[0];
 let trainRollouts=0;
 for(let seed=60100;seed<60170;seed++)for(let [hor,cap] of experiments){
  const origin=c.prefix(seed,hor,"train",cap),x=publicFeatures(c,origin,cap);
  if(!origin.paintedA||!origin.hazard)throw Error("missing target A or opportunity");
  const tested=CANDIDATES.map(mode=>{
   let goals=0,utility=0;
   for(let first of [0,1])for(let second of possibilities(hor)){
    const r=runPolicy(c,variant(origin,first),cap,mode,second).out;
    goals+=r.terminalGoal;utility+=r.utility;trainRollouts++;
   }
   return{mode,goals,utility};
  }).sort((a,b)=>b.goals-a.goals||b.utility-a.utility||CANDIDATES.indexOf(a.mode)-CANDIDATES.indexOf(b.mode));
  const label=CANDIDATES.indexOf(tested[0].mode);examples.push({seed,hor,cap,x,label});trainCounts[tested[0].mode]++;
 }
 const weights=train(examples);
 const rows=[],paths=[],negative=[];let receipts=0,twins=0,corrupt=0,firstWins=0,secondWins=0;
 for(let seed=70100;seed<70150;seed++)for(let [hor,cap] of experiments){
  const origin=c.prefix(seed,hor,"dev",cap),x=publicFeatures(c,origin,cap);
  const opts=probs(weights,x),selected=CANDIDATES[opts.indexOf(Math.max(...opts))];
  for(let first of [0,1])for(let second of possibilities(hor)){
   twins++;
   const root=variant(origin,first),before=JSON.stringify(root.observation());
   const counter=variant(origin,1-first);
   if(before!==JSON.stringify(counter.observation())||JSON.stringify(x)!==JSON.stringify(publicFeatures(c,counter,cap)))throw Error("hidden state leaked into public features");
   const group={};
   for(let policy of POLICIES){
    const mode=policy==="student"?selected:policy==="censor_aware"?(hor===128&&cap===64?"early":"feasible"):policy;
    const result=runPolicy(c,root,cap,mode,second);
    receipts+=replay(c,result,seed,hor,cap,"dev",origin.tick,first,second);
    const v=result.out,entry={seed,hor,cap,first,second,lateApplied:result.lateApplied,policy,mode,
      goal:v.goal,terminalGoal:v.terminalGoal,stopped:v.stopped,utility:v.utility,ticks:v.ticks,calls:v.calls,checks:v.checks,counts:v.counts,receiptHash:v.hash};
    rows.push(entry);group[policy]=entry;
    if([70102,70119,70137,70148].includes(seed)&&hor===128&&cap===128&&["early","feasible","student","postpulse_oracle"].includes(policy))
     paths.push({...entry,positions:v.positions});
   }
   if(group.early.terminalGoal<group.feasible.terminalGoal)firstWins++;
   if(group.student.terminalGoal<group.feasible.terminalGoal||group.student.terminalGoal<group.censor_aware.terminalGoal||group.early.terminalGoal<group.feasible.terminalGoal||group.feasible.terminalGoal<group.early.terminalGoal)negative.push({seed,hor,cap,first,second,studentMode:selected,
      earlyGoal:group.early.terminalGoal,guardGoal:group.feasible.terminalGoal,censorGoal:group.censor_aware.terminalGoal,studentGoal:group.student.terminalGoal,
      earlyUtility:group.early.utility,guardUtility:group.feasible.utility,censorUtility:group.censor_aware.utility,studentUtility:group.student.utility});
   if(group.early.terminalGoal<group.late.terminalGoal&&first===0&&second===1)secondWins++;
  }
 }
 const bad=new c.Host(7,36,"train");bad.step("E");
 for(let altered of [[{...bad.log[0],action:"TELEPORT"}],[{...bad.log[0],after:[999,999]}]]){
  try{c.assertReceipts(altered)}catch(e){corrupt++}
 }
 if(examples.length!==280||trainRollouts!==3360||twins!==600||rows.length!==5400||receipts<20000||corrupt!==2)throw Error("incomplete cohort");
 const horizons={};for(let [hor,cap] of experiments)horizons[hor+"/"+cap]=Object.fromEntries(POLICIES.map(p=>[p,rows.filter(r=>r.hor===hor&&r.cap===cap&&r.policy===p).reduce((v,r)=>v+r.terminalGoal,0)]));
 let inferBegin=Date.now(),checksum=0;
 for(let n=0;n<100000;n++){const p=probs(weights,examples[n%examples.length].x);checksum+=p[n%4]}
 let inferMs=Date.now()-inferBegin;
 return{experiment:"MULTIPULSE_PHYSICAL_018",limitations:"Hypothetical synthetic physical state, replayed by fresh same-source host. Not independent/native CyberSand ABI validation.",
 provenance:{hostGitBlob:HOST_GIT_BLOB,split:"TRAIN 60100..60169; DEV 70100..70149; structural HELDOUT untouched"},
 setup:{trainWorldClusters:70,developmentWorldClusters:50,horizons:experiments,trainLabels:examples.length,
 optimizerUpdates:1100,trainRollouts,devCounterfactuals:twins,developmentRollouts:rows.length,replayedReceipts:receipts,
 corruptionRejections:corrupt,firstOnlyGoalLossesVsGuard:firstWins,secondDamageExposedLosses:secondWins},
 training:{weights,labelCounts:trainCounts,trainExamples:examples},development:{summary:Object.fromEntries(POLICIES.map(p=>[p,summary(rows,p,c)])),
 horizons,negativeCases:negative,rows},paths,bench:{campaignWallMs:Date.now()-begin,
 batchPredictions:100000,predictionBatchMs:inferMs,predictionChecksum:checksum}};
}
function gallery(paths){
 const count=Math.min(32,paths.length),w=1200,height=Math.ceil(count/4)*164;
 let s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+height+'"><rect width="100%" height="100%" fill="#101922"/>'];
 for(let i=0;i<count;i++){const r=paths[i],x=i%4*300,y=Math.floor(i/4)*164;
 const pts=r.positions.map(p=>[x+12+p[0]*21,y+19+p[1]*13].join(",")).join(" ");
 s.push('<g><rect x="'+(x+3)+'" y="'+(y+3)+'" width="294" height="157" fill="#273848" stroke="#678296"/><polyline points="'+pts+'" fill="none" stroke="'+(r.goal?"#83d5bb":"#f4978e")+'" stroke-width="2"/><text font-family="monospace" fill="#fff" font-size="10" x="'+(x+8)+'" y="'+(y+150)+'">'+r.policy+' '+r.seed+' '+r.hor+'/'+r.cap+' a='+r.first+' b='+r.second+' success='+r.goal+'</text></g>');
 }
 return s.concat("</svg>").join("\n")+"\n";
}
// Explicit manual Node execution only; no optimizer runs on import / CI smoke.
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("fs"),path=require("path"),vm=require("vm"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))throw Error("Usage: node multipulse_physical_018.js /absolute/output/directory");
 const hostFile=path.join(__dirname,"physical_inspection_016.js"),src=fs.readFileSync(hostFile);
 const blob=crypto.createHash("sha1").update("blob "+src.length+"\0").update(src).digest("hex");
 if(blob!==HOST_GIT_BLOB)throw Error("Physical host source drift; requalification required");
 const c=vm.runInNewContext(src.toString("utf8")+"\n;({Host,prefix,finish,move,correct,bfs,walls,TARGETS,CAPS,DIR,ACT,key,eq,assertReceipts,hash})",{});
 const r=study(c);r.bench.nodeRSSBytes=process.memoryUsage().rss;r.bench.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 r.provenance.sourceSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 const dir=process.argv[2];fs.mkdirSync(dir,{recursive:true});
 fs.writeFileSync(path.join(dir,"MULTIPULSE_PHYSICAL_018_EVIDENCE.json"),JSON.stringify(r,null,2)+"\n");
 fs.writeFileSync(path.join(dir,"MULTIPULSE_PHYSICAL_018_PATHS.svg"),gallery(r.paths));
 console.log(JSON.stringify({setup:r.setup,training:r.training.labelCounts,summary:r.development.summary,bench:r.bench},null,2));
}
if(typeof module!=="undefined")module.exports={study,gallery};
