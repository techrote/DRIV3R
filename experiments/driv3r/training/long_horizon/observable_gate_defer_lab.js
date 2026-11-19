"use strict";
// OBSERVED-GATE-DEFER-004. Strict source-anchored synthetic policy ablation.
// Does not alter production host ABI, open structural HELDOUT, or train on push.
const EXPECTED_UPSTREAM_BLOB="4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873";
function rewrite(upstream){
 function once(a,b){const i=upstream.indexOf(a);if(i<0||upstream.indexOf(a,i+a.length)>=0)throw Error("Upstream source drift: "+a.slice(0,55));upstream=upstream.slice(0,i)+b+upstream.slice(i+a.length)}
 once('ARMS=["lazy","overlap","epoch","privileged_oracle","no_cue","cue_classifier","cue_utility"]',
 'ARMS=["lazy","overlap","epoch","privileged_oracle","no_cue","cue_classifier","cue_utility","defer1","defer2","defer3","defer4","distance2","distance3","distance4","train_selected"]');
 once('function episode(sp,arm,models,opts={}){',
 'function episode(sp,arm,models,opts={}){ const logicalArm=arm;if(arm==="train_selected")arm=models.selected[sp.horizon];');
 once('let calls=1,invalidations=0,illegal=0,steps=[],counts={},prediction=null,overlap=null;',
 'let calls=1,invalidations=0,illegal=0,steps=[],counts={},prediction=null,overlap=null,closedAt=null,delayed=false;');
 once('overlap=cacheTargets(w,cache).includes(key(cell));',
 'overlap=cacheTargets(w,cache).includes(key(cell));closedAt=overlap?5:null;');
 once('  if(!cache.length){cache=bfs(w)||[];calls++}',
 '  if(closedAt!==null&&tick>5&&!delayed&&w.materials.has(key(w.gate))){const d=cacheTargets(w,cache).indexOf(key(w.gate))+1;const trigger=(arm.startsWith("defer")&&tick-closedAt>=Number(arm.slice(5)))||(arm.startsWith("distance")&&d>0&&d<=Number(arm.slice(8)));if(trigger){cache=[];invalidations++;delayed=true}}\n  if(!cache.length){cache=bfs(w)||[];calls++}');
 once('return{spec:sp.id,arm,cap,cohort:', 'return{spec:sp.id,arm:logicalArm,cap,cohort:');
 once('const models={cue,nocue,utilityRules};',
 'const models={cue,nocue,utilityRules};\nconst selected={};for(const h of H){let best=null;for(const a of ["overlap","defer1","defer2","defer3","defer4","distance2","distance3","distance4"]){let n=0,sum=0;for(const seed of TRAIN)for(const mode of MODES){const r=episode(spec("train",seed,h,mode),a,models,{cap:h===128?128:64});sum+=utility(r);n++}const mean=sum/n;if(best===null||mean>best.mean||(mean===best.mean&&a<best.arm))best={arm:a,mean}}selected[h]=best.arm;}models.selected=selected;');
 once('check(episodes===1344&&results.length===1344','check(episodes===2880&&results.length===2880');
 once('&&(arm==="lazy"||arm==="overlap"||arm==="cue_classifier")){',
 '&&(arm==="lazy"||arm==="overlap"||arm==="defer3")){');
 once('study:"FORECAST-UTILITY-002",version:VERSION,',
 'study:"OBSERVED-GATE-DEFER-004",version:VERSION,');
 once('handoffs,utility_calibration:utilityCalibration,',
 'handoffs,utility_calibration:utilityCalibration,policy_selection:selected,');
 once('FORECAST-UTILITY-002: synthetic 72-decision gate trajectories',
 'OBSERVED-GATE-DEFER-004: synthetic 72-decision gate trajectories');
 return upstream;
}
function execute(upstream){return(new Function(rewrite(upstream)+"\nreturn forecastStudy();"))()}
if(typeof module!=="undefined")module.exports={rewrite,execute,EXPECTED_UPSTREAM_BLOB};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
 const upstream=fs.readFileSync(path.join(__dirname,"forecast_utility_lab.js"),"utf8");
 const hash=crypto.createHash("sha1").update("blob "+Buffer.byteLength(upstream)+"\0").update(upstream).digest("hex");
 if(hash!==EXPECTED_UPSTREAM_BLOB)throw Error("Upstream Git blob mismatch: "+hash);
 if(process.argv.length!==3)throw Error("Usage: node observable_gate_defer_lab.js /absolute/output/dir");
 const dir=path.resolve(process.argv[2]);fs.mkdirSync(dir,{recursive:true});
 const rss0=process.memoryUsage().rss,t0=process.hrtime.bigint(),result=execute(upstream);
 result.evidence.source_upstream_git_blob=hash;
 result.evidence.rss_endpoint_max_bytes=Math.max(rss0,process.memoryUsage().rss);
 result.evidence.node_wall_ms=Number(process.hrtime.bigint()-t0)/1e6;
 result.evidence.node_maxrss_kib=process.resourceUsage().maxRSS;
 fs.writeFileSync(path.join(dir,"OBSERVED_GATE_DEFER_EVIDENCE.json"),JSON.stringify(result.evidence,null,2)+"\n");
 fs.writeFileSync(path.join(dir,"OBSERVED_GATE_DEFER_TRAJECTORIES.svg"),result.svg+"\n");
 process.stdout.write(JSON.stringify({study:result.evidence.study,episodes:result.evidence.development_episodes,receipts:result.evidence.action_receipts_replayed,tests:result.evidence.tests_passed,selection:result.evidence.policy_selection,summary:result.evidence.summary,node_wall_ms:result.evidence.node_wall_ms,node_maxrss_kib:result.evidence.node_maxrss_kib},null,2)+"\n");
}