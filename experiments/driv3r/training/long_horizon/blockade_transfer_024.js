"use strict";
// BLOCKADE-TRANSFER-024: explicit opt-in retrospective mode selection, no routine CI optimizer.
// Read-only source 023 evidence. Structural HELDOUT sealed. Synthetic game host != native ABI.
const LEDGER_GIT_BLOB="cf4dc3bcd96057c2dcf7fb6dd4446ea584ec45d4";
function analyze(d){
 const C=["control","one","two"],H=["36/36","72/64","128/64","128/128"],M=["guard","early","late","both"],A=["guard","student021","early","late","both","hindsight"],bad=s=>{throw Error("024 ledger invalid: "+s)};
 if(d.study!=="INTERCEPT_BLOCKADE_023"||d.rows?.length!==1080||d.run.sourceReplayedReceipts!==253022)bad("source");
 const indexed=Object.fromEntries(C.map(c=>[c,new Map()])),tot=Object.fromEntries(C.map(c=>[c,Object.fromEntries(A.map(a=>[a,{goals:0,calls:0}]))]));
 let changedHashes=0;
 for(const r of d.rows){
  if(!C.includes(r.condition)||!H.includes(r.h+"/"+r.cap)||r.seed<160100||r.seed>=160130||r.h!==128&&r.second!==0)bad("cohort");
  const k=[r.seed,r.h,r.cap,r.first,r.second].join("/");
  if(indexed[r.condition].has(k))bad("duplicate");indexed[r.condition].set(k,r);
  for(const a of A){const o=r.outcome[a];if(!o||![0,1].includes(o.goal)||!Number.isInteger(o.calls)||!o.hash)bad("invalid arm");tot[r.condition][a].goals+=o.goal;tot[r.condition][a].calls+=o.calls}
  const sel={feasible:"guard",early:"early",late:"late",both:"both"}[r.student];
  if(!sel||r.outcome.student021.goal!==r.outcome[sel].goal||r.outcome.student021.calls!==r.outcome[sel].calls||r.outcome.student021.hash!==r.outcome[sel].hash)bad("student audit");
  changedHashes+=+(r.outcome.student021.hash!==r.outcome.guard.hash);
 }
 for(const c of C){
  if(indexed[c].size!==360)bad("count");
  for(const a of A)for(const f of ["goals","calls"])if(tot[c][a][f]!==d.summary[c][a][f])bad("aggregate "+c+a+f);
 }
 for(const [k,v] of indexed.control){
  const o=indexed.one.get(k),t=indexed.two.get(k);
  if(!o||!t||v.events.length!==0||o.events.length>1||t.events.length>2||
   o.events.length&&JSON.stringify(o.events[0])!==JSON.stringify(t.events[0]))bad("pair "+k);
 }
 const fitRows=d.rows.filter(r=>r.seed<160118),testRows=d.rows.filter(r=>r.seed>=160118);
 if(fitRows.length!==648||testRows.length!==432)bad("split");
 const summary=(rs,arm,c,h)=>{
  const x=rs.filter(r=>(c===null||r.condition===c)&&(h===null||r.h+"/"+r.cap===h));
  return{n:x.length,goals:x.reduce((v,r)=>v+r.outcome[arm].goal,0),calls:x.reduce((v,r)=>v+r.outcome[arm].calls,0)};
 };
 function fit(conds,minimax){
  const choice={},scores={};
  for(const h of H){
   const items=M.map(mode=>{const a=conds.map(c=>summary(fitRows,mode,c,h));return{mode,minGoals:Math.min(...a.map(x=>x.goals)),totalGoals:a.reduce((v,x)=>v+x.goals,0),calls:a.reduce((v,x)=>v+x.calls,0),byCondition:Object.fromEntries(conds.map((c,i)=>[c,a[i]]))}});
   items.sort((a,b)=>minimax?(b.minGoals-a.minGoals||b.totalGoals-a.totalGoals||a.calls-b.calls||M.indexOf(a.mode)-M.indexOf(b.mode)):(b.totalGoals-a.totalGoals||a.calls-b.calls||M.indexOf(a.mode)-M.indexOf(b.mode)));
   choice[h]=items[0].mode;scores[h]=items;
  }
  return{choice,scores};
 }
 const mild=fit(["control","one"],false),robust=fit(C,true),guard=Object.fromEntries(H.map(h=>[h,"guard"]));
 const compute=(rs,ch,c)=>{
  const x=rs.filter(r=>c===null||r.condition===c);
  return{n:x.length,goals:x.reduce((s,r)=>s+r.outcome[ch[r.h+"/"+r.cap]].goal,0),calls:x.reduce((s,r)=>s+r.outcome[ch[r.h+"/"+r.cap]].calls,0)};
 };
 const testing=Object.fromEntries(["control","one","two","all"].map(x=>[x,{guard:compute(testRows,guard,x==="all"?null:x),mild:compute(testRows,mild.choice,x==="all"?null:x),robust:compute(testRows,robust.choice,x==="all"?null:x),student021:summary(testRows,"student021",x==="all"?null:x,null)}]));
 const byHorizon=Object.fromEntries(H.map(h=>[h,Object.fromEntries(C.map(c=>{const x=testRows.filter(r=>r.h+"/"+r.cap===h&&r.condition===c);return[c,{guard:compute(x,guard,null),mild:compute(x,mild.choice,null),robust:compute(x,robust.choice,null)}]}))]));
 const severe=testRows.filter(r=>r.h===128&&r.cap===64&&r.condition==="two"),selected=mild.choice["128/64"];
 if(severe.length!==48)bad("severe");
 const lost=severe.filter(r=>r.outcome[selected].goal<r.outcome.guard.goal).map(r=>({seed:r.seed,firstDamage:r.first,secondDamage:r.second,events:r.events,guard:r.outcome.guard,chosen:r.outcome[selected]}));
 const gained=severe.filter(r=>r.outcome[selected].goal>r.outcome.guard.goal).length;
 const deltas=Array.from({length:12},(_,i)=>{const seed=160118+i,x=severe.filter(r=>r.seed===seed);return{seed,delta:x.reduce((s,r)=>s+r.outcome[selected].goal-r.outcome.guard.goal,0),walls:x[0].events.map(y=>y.cell)}});
 let state=0xc0ffee;const rnd=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296};
 let boots=[];for(let i=0;i<4096;i++){let sum=0;for(let j=0;j<12;j++)sum+=deltas[Math.floor(rnd()*12)].delta;boots.push(sum)}boots.sort((a,b)=>a-b);
 return{experiment:"BLOCKADE_TRANSFER_024",method:"retrospective, four-mode exhaustive optimization over source-replayed 023 candidate trajectories",
  provenance:{sourceEvidenceGitBlob:"cf4dc3bcd96057c2dcf7fb6dd4446ea584ec45d4",train:"UNSEALED DEVELOPMENT 160100..160117",test:"UNSEALED DEVELOPMENT 160118..160129",structuralHELDOUT:"UNOPENED"},
  mild:{objective:"maximize terminal goals on zero/one closure; then minimize planner calls",...mild},
  robust:{objective:"maximize worst-condition goals over zero/one/two closures; then sum goals then minimize calls",...robust},
  testing,byHorizon,
  severe128_64:{n:severe.length,selected,lost:lost.length,gained,goalDelta:deltas.reduce((s,x)=>s+x.delta,0),bootstrap95:[boots[Math.floor(.025*(boots.length-1))],boots[Math.floor(.975*(boots.length-1))]],bootstrapN:4096,bootstrapUnit:"12 independent seed clusters, four correlated material forks per seed",seedDeltas:deltas,negativeCases:lost},
  audit:{sourceRows:d.rows.length,matchedPairedConditions:360,studentModeVerified:1080,changedStudentReceiptHashes:changedHashes,reconciledFields:36,sourceHostReplayedReceipts:d.run.sourceReplayedReceipts,newHostReplays:0},
  limitations:"Posthoc reuse of previously inspected unsealed DEV; not prospective generalization. New 024 lookup optimizer is not a neural SLM. 023 same-source host-replayed records only, not independently verified native ABI. Synthetic STOP and dynamic walls are research assumptions."};
}
if(typeof module!=="undefined")module.exports={analyze};
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),path=require("path"),crypto=require("crypto");
 if(process.argv.length!==4||!path.isAbsolute(process.argv[2])||!path.isAbsolute(process.argv[3]))
  throw Error("Usage: node blockade_transfer_024.js /absolute/023_EVIDENCE.json /absolute/output");
 const raw=fs.readFileSync(process.argv[2]);
 const gitBlob=crypto.createHash("sha1").update("blob "+raw.length+String.fromCharCode(0)).update(raw).digest("hex");
 if(gitBlob!==LEDGER_GIT_BLOB)throw Error("Source 023 evidence Git blob mismatch: "+gitBlob);
 const result=analyze(JSON.parse(raw.toString("utf8")));
 fs.mkdirSync(process.argv[3],{recursive:true});
 fs.writeFileSync(path.join(process.argv[3],"BLOCKADE_TRANSFER_024_EVIDENCE.json"),JSON.stringify(result,null,2)+String.fromCharCode(10));
 console.log(JSON.stringify({selection:{mild:result.mild.choice,robust:result.robust.choice},testing:result.testing,severe:result.severe128_64,audit:result.audit},null,2));
}
