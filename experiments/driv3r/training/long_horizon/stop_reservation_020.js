"use strict";
// STOP-RESERVATION-020: bounded, opt-in synthetic 36/72/128+ game-policy science.
// TRAIN-only frozen 018 model. No sealed structural HELDOUT, native ABI claims,
// paid compute, production changes, model binaries, or automatic CI execution.
const HOST="45a426f893203f93fa9a3972d63bd4a110ceaa3d",MODEL="4e7f4b4b20d4db42d601961a7689f3729477aaf6";
const CAPS=[[36,36],[72,64],[128,64],[128,128]],EXTRA=["5,4","9,4","6,6"];
const P=["feasible","feasible_stop","student","student_stop","censor_aware","censor_stop","late","late_stop","never"];
const PAIRS=[["feasible","feasible_stop"],["student","student_stop"],["censor_aware","censor_stop"],["late","late_stop"]];
function prefix(c,seed,hor,cap,shift){
 if(!shift)return c.prefix(seed,hor,"dev",cap);
 const h=new c.Host(seed,hor,"dev");
 for(const w of EXTRA)h.walls.add(w);
 for(let i=0;i<(hor===36?2:3);i++)
  if(!c.move(h,c.TARGETS[i],cap)||!c.correct(h,cap))break;
 return h;
}
function freeze(c,e){
 const training=[],labels={never:0,early:0,feasible:0,both:0};
 for(let seed=60100;seed<60170;seed++)for(const [hor,cap] of CAPS){
  const z=c.prefix(seed,hor,"train",cap);
  const ranked=e.CANDIDATES.map(mode=>{
   let goal=0,utility=0;
   for(const a of [0,1])for(const b of hor===128?[0,1]:[0]){
    const r=e.runPolicy(c,e.variant(z,a),cap,mode,b).out;
    goal+=r.terminalGoal;utility+=r.utility;
   }
   return{mode,goal,utility};
  }).sort((a,b)=>b.goal-a.goal||b.utility-a.utility||
   e.CANDIDATES.indexOf(a.mode)-e.CANDIDATES.indexOf(b.mode));
  labels[ranked[0].mode]++;
  training.push({x:e.publicFeatures(c,z,cap),label:e.CANDIDATES.indexOf(ranked[0].mode)});
 }
 if(training.length!==280||JSON.stringify(labels)!==
   JSON.stringify({never:86,early:124,feasible:70,both:0}))throw Error("018 training drift");
 return{weights:e.train(training),labels};
}
// Reserving one logical action for STOP is a *synthetic* policy experiment,
// not a claim that real CyberSand requires a STOP receipt for success.
function preempt(c,root,cap,mode,second){
 const h=root.clone(),n={36:2,72:3,128:5}[h.horizon],limit=cap-1,A=c.TARGETS[0],k="2,1";
 let checks=0;
 function inspect(){if(h.tick<limit&&c.move(h,A,limit)){checks++;c.correct(h,limit)}}
 function guard(extra){const d=c.bfs(h.pos,A,h.walls).length;
  if((d>0||c.eq(h.pos,A))&&limit-h.tick>=d+extra)inspect()}
 if(mode==="early"||mode==="both")inspect();
 for(let i=h.horizon===36?2:3;i<n;i++){
  if(!c.move(h,c.TARGETS[i],limit))break;
  const old=h.ink[k]??null,good=c.correct(h,limit);
  if(h.horizon===128&&i===4&&h.secondHazard){
   if(second)delete h.ink[k];else if(old===null)delete h.ink[k];else h.ink[k]=old;
   h.secondDamage=!!second;
  }
  if(!good)break;
 }
 if(mode==="late"||mode==="both")inspect();
 if(mode==="feasible")guard(2);
 if(h.tick<cap)h.step("STOP");
 const out=h.result(cap,checks);
 out.stopped=+(h.log.at(-1).action==="STOP");
 out.terminalGoal=+(out.goal&&out.stopped);
 if(!out.stopped)throw Error("reserved STOP not delivered");
 return{host:h,out};
}
// Reconstruct all action receipts from a separate instance of pinned synthetic
// host, including two hidden material interventions and final material state.
function replay(c,run,seed,hor,cap,pre,first,second,shift){
 const z=new c.Host(seed,hor,"dev"),k="2,1",logs=run.host.log;
 if(shift)for(const w of EXTRA)z.walls.add(w);
 if(logs.length>cap)throw Error("cap exceeded");
 for(let i=0;i<logs.length;i++){
  if(i===pre){
   if(!z.paintedA||!z.hazard)throw Error("bad first intervention precondition");
   z.ink[k]=1;z.damage=!!first;if(first)delete z.ink[k];
  }
  const old=z.ink[k]??null,receipt=z.step(logs[i].action);
  if(JSON.stringify(receipt)!==JSON.stringify(logs[i]))
   throw Error("replay failure "+[seed,hor,cap,first,second,i].join("/"));
  if(hor===128&&receipt.action==="PAINT"&&
     c.eq(receipt.after,c.TARGETS[4])&&z.secondHazard){
   if(second)delete z.ink[k];else if(old===null)delete z.ink[k];else z.ink[k]=old;
   z.secondDamage=!!second;
  }
 }
 if(z.tick!==run.host.tick||!c.eq(z.pos,run.host.pos)||
    JSON.stringify(z.ink)!==JSON.stringify(run.host.ink))
  throw Error("final material/pose replay mismatch");
 return c.assertReceipts(logs);
}
function study(c,e){
 const begin=Date.now(),model=freeze(c,e),summary={},cohorts={},negatives=[],paths=[];
 let conditions=0,rollouts=0,receipts=0,corruptions=0;
 for(const [name,beginSeed,shift] of [["fresh",100100,false],["shift",110100,true]]){
  summary[name]={};
  for(const p of P)summary[name][p]={n:0,terminal:0,material:0,stopped:0,
   nonterminalMaterial:0,calls:0,checks:0,utility:0,
   actions:Object.fromEntries(c.ACT.map(a=>[a,0]))};
  for(const [hor,cap] of CAPS){
   const ck=name+"/"+hor+"/"+cap;cohorts[ck]={};
   for(const p of P)cohorts[ck][p]={terminal:0,material:0,stopped:0};
   for(let seed=beginSeed;seed<beginSeed+50;seed++){
    const root=prefix(c,seed,hor,cap,shift);
    if(!root.hazard||!root.paintedA)throw Error("unqualified world prefix");
    const probs=e.probs(model.weights,e.publicFeatures(c,root,cap));
    const selected=e.CANDIDATES[probs.indexOf(Math.max(...probs))];
    for(const first of [0,1])for(const second of hor===128?[0,1]:[0]){
     conditions++;
     const fork=e.variant(root,first);
     if(JSON.stringify(fork.observation())!==
       JSON.stringify(e.variant(root,1-first).observation()))
      throw Error("offscreen material leaks into public observation");
     const outcomes={},routePaths={};
     for(const p of P){
      const mode=p.startsWith("feasible")?"feasible":
       p.startsWith("student")?selected:
       p.startsWith("censor")?(hor===128&&cap===64?"early":"feasible"):
       p.startsWith("late")?"late":"never";
      const rr=p.endsWith("_stop")?preempt(c,fork,cap,mode,second):
       e.runPolicy(c,fork,cap,mode,second);
      const r=rr.out;
      receipts+=replay(c,rr,seed,hor,cap,root.tick,first,second,shift);
      rollouts++;
      const s=summary[name][p],q=cohorts[ck][p];
      s.n++;s.terminal+=r.terminalGoal;s.material+=r.goal;s.stopped+=r.stopped;
      s.nonterminalMaterial+=+(r.goal&&!r.stopped);
      s.calls+=r.calls;s.checks+=r.checks;s.utility+=r.utility;
      for(const a of c.ACT)s.actions[a]+=r.counts[a];
      q.terminal+=r.terminalGoal;q.material+=r.goal;q.stopped+=r.stopped;
      outcomes[p]={terminal:r.terminalGoal,material:r.goal,stopped:r.stopped,
       calls:r.calls,receiptHash:r.hash};
      if(["late","late_stop","censor_aware","censor_stop"].includes(p))
       routePaths[p]=r.positions;
     }
     for(const [base,reserved] of PAIRS){
      if(outcomes[reserved].terminal<outcomes[base].terminal)
       throw Error("reserved arm lost terminal goal");
      if(!outcomes[reserved].stopped)throw Error("missing reserved STOP");
     }
     const lateGap=outcomes.late.material&&!outcomes.late.stopped;
     const censorGap=outcomes.censor_aware.material&&!outcomes.censor_aware.stopped;
     if(lateGap||censorGap||outcomes.student.terminal!==outcomes.feasible.terminal){
      negatives.push({name,seed,hor,cap,first,second,studentMode:selected,outcomes});
      if(paths.length<16&&(lateGap||censorGap)){
       const p=lateGap?"late":"censor_aware",s=p==="late"?"late_stop":"censor_stop";
       for(const z of [p,s])paths.push({group:name,seed,hor,cap,first,second,
        policy:z,terminal:outcomes[z].terminal,positions:routePaths[z]});
      }
     }
    }
   }
  }
 }
 const host=new c.Host(7,36,"train");host.step("E");
 for(const invalid of [[{...host.log[0],action:"TELEPORT"}],
  [{...host.log[0],after:[999,999]}]]){
  try{c.assertReceipts(invalid)}catch{corruptions++}
 }
 if(conditions!==1200||rollouts!==10800||receipts<400000||corruptions!==2)
  throw Error("incomplete campaign");
 for(const block of Object.values(summary))for(const s of Object.values(block))
  s.meanUtility=+(s.utility/s.n).toFixed(3);
 return{experiment:"STOP_RESERVATION_020",
  provenance:{hostGitBlob:HOST,studentGitBlob:MODEL,
   train:"TRAIN 60100-60169",dev:"100100-100149;110100-110149",
   structuralHeldout:"unopened"},
  qualifier:"Synthetic exact-source replay only, NOT independent native CyberSand",
  run:{conditions,rollouts,replayedReceipts:receipts,corruptionRejections:corruptions,
   negativeCaseRecords:negatives.length,trainSamples:280,
   trainOptimizerUpdates:1100,trainLabels:model.labels,
   v8WallMs:Date.now()-begin,peakRSS:"not available in V8; manual Node emits"},
  weights:model.weights,summary,cohorts,negativeCases:negatives,paths};
}
function gallery(paths){
 const a=paths.slice(0,16),w=1200,h=Math.ceil(a.length/4)*164;
 let s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+h+'"><rect width="100%" height="100%" fill="#122330"/>'];
 for(let i=0;i<a.length;i++){
  const r=a[i],x=i%4*300,y=Math.floor(i/4)*164;
  const pts=r.positions.map(p=>[x+10+p[0]*21,y+16+p[1]*13].join(",")).join(" ");
  s.push('<rect x="'+(x+2)+'" y="'+(y+2)+'" width="296" height="160" fill="#243c4a" stroke="#668298"/>');
  s.push('<polyline points="'+pts+'" stroke="'+(r.terminal?"#8adbb7":"#ef9c98")+'" fill="none" stroke-width="2"/>');
  s.push('<text x="'+(x+8)+'" y="'+(y+155)+'" font-family="monospace" font-size="10" fill="white">'+r.group+' '+r.seed+' '+r.hor+'/'+r.cap+' '+r.policy+' '+r.terminal+'</text>');
 }
 return s.concat("</svg>").join("\n")+"\n";
}
if(typeof module!=="undefined")module.exports={study,gallery};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),vm=require("vm"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))
  throw Error("Explicit CPU run only: node stop_reservation_020.js /absolute/output");
 function load(file,pin,tail){
  const bytes=fs.readFileSync(path.join(__dirname,file));
  const blob=crypto.createHash("sha1").update("blob "+bytes.length+"\0").update(bytes).digest("hex");
  if(blob!==pin)throw Error("dependency Git blob drift "+file);
  return vm.runInNewContext(bytes.toString("utf8")+tail,{});
 }
 const c=load("physical_inspection_016.js",HOST,
  "\n;({Host,prefix,move,correct,bfs,TARGETS,ACT,eq,assertReceipts})");
 const e=load("multipulse_physical_018.js",MODEL,
  "\n;({variant,runPolicy,publicFeatures,train,probs,CANDIDATES})");
 const r=study(c,e),dst=process.argv[2];
 r.run.nodePeakRSSKiB=process.resourceUsage().maxRSS;
 r.run.nodeRSSBytes=process.memoryUsage().rss;
 r.provenance.scriptSHA256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex");
 fs.mkdirSync(dst,{recursive:true});
 fs.writeFileSync(path.join(dst,"STOP_RESERVATION_020_EVIDENCE.json"),JSON.stringify(r,null,2)+"\n");
 fs.writeFileSync(path.join(dst,"STOP_RESERVATION_020_PATHS.svg"),gallery(r.paths));
 console.log(JSON.stringify({run:r.run,summary:r.summary,cohorts:r.cohorts},null,2));
}