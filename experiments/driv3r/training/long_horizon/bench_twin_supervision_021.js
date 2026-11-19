"use strict";
// TWIN-SUPERVISION-021: opt-in predictor-only Node microbenchmark.
// Requires a local evidence JSON; does not qualify game-policy inference.
const fs=require("fs"),path=require("path");
if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))throw Error("node bench_twin_supervision_021.js /absolute/evidence.json");
const ev=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));
if(ev.experiment!=="TWIN_SUPERVISION_021")throw Error("wrong evidence");
const w=ev.training.weights;
if(w.length!==4||w.some(r=>r.length!==13||r.some(v=>!Number.isFinite(v))))throw Error("weights invalid");
function probs(x){const z=w.map(row=>row.reduce((s,v,i)=>s+v*x[i],0)),mx=Math.max(...z),e=z.map(v=>Math.exp(v-mx)),total=e.reduce((s,v)=>s+v,0);return e.map(v=>v/total)}
const xs=Array.from({length:128},(_,i)=>[1,+(i%4===0),+(i%4===1),+(i%4>=2),+(i%4===1||i%4===2),+(i%4===3),.2+i/256,.1+i/256,.1+i/384,i/200,17/30,11/13,7/9]);
for(let i=0;i<30000;i++)probs(xs[i%128]);
const ns=[],N=30000;let checksum=0;
for(let i=0;i<N;i++){const start=process.hrtime.bigint(),p=probs(xs[i%128]);ns.push(Number(process.hrtime.bigint()-start));checksum+=p[i%4]}
ns.sort((a,b)=>a-b);
console.log(JSON.stringify({model:"021 four-class softmax only",node:process.version,N,p50Ns:ns[Math.floor(N*.5)],p95Ns:ns[Math.floor(N*.95)],p99Ns:ns[Math.floor(N*.99)],peakRSSKiB:process.resourceUsage().maxRSS,rssBytes:process.memoryUsage().rss,checksum},null,2));
