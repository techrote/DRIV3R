"use strict";
// Predictor-only benchmark, explicit invocation. Never started from CI or import.
// Loaded weights from committed 022 JSON; does not execute world rollout or native game.
const fs=require("fs"),path=require("path");
if(process.argv.length!==3||!path.isAbsolute(process.argv[2])){
 throw Error("Usage: node bench_route_invalidation_022.js /absolute/path/to/ROUTE_INVALIDATION_022_EVIDENCE.json");
}
const evidence=JSON.parse(fs.readFileSync(process.argv[2],"utf8"));
if(evidence.experiment!=="ROUTE_INVALIDATION_022")throw Error("Unexpected experiment evidence");
const w=evidence.training.weights;
function probs(weights,x){
 const z=weights.map(row=>row.reduce((s,v,i)=>s+v*x[i],0)),mx=Math.max(...z),e=z.map(v=>Math.exp(v-mx)),total=e.reduce((s,v)=>s+v,0);
 return e.map(v=>v/total);
}
// A real public feature vector from the 150100 128/64 shifted prefix.
const input=[1,0,0,1,1,0,0.7734375,0.2734375,0.2265625,0.5,0.6666666666666666,0.8461538461538461,0.7777777777777778];
const reference=[0.5733769608837056,0.0010427495415660484,0.4245375400331623,0.0010427495415660484];
const computed=probs(w,input);
if(computed.some((v,i)=>Math.abs(v-reference[i])>1e-12))throw Error("Frozen predictor numerical drift");
const warmup=30000,count=50000;
let sink=0;
for(let i=0;i<warmup;i++)sink+=probs(w,input)[0];
const timings=new Float64Array(count),start=process.hrtime.bigint();
for(let i=0;i<count;i++){
 const t=process.hrtime.bigint();sink+=probs(w,input)[i%4];
 timings[i]=Number(process.hrtime.bigint()-t);
}
const wall=Number(process.hrtime.bigint()-start)/1e6;
timings.sort();
const pct=p=>timings[Math.floor((count-1)*p)];
console.log(JSON.stringify({experiment:evidence.experiment,method:"predictor-only exact 018 softmax kernel, 021 fitted weights, one actual 128/64 prefix feature vector",node:process.version,calls:count,warmup,p50_ns:pct(.5),p95_ns:pct(.95),p99_ns:pct(.99),batch_wall_ms:wall,peakRSS_KiB:process.resourceUsage().maxRSS,currentRSS_bytes:process.memoryUsage().rss,checksum:sink},null,2));
