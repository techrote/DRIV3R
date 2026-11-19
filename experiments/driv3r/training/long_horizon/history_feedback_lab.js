"use strict";
// HISTORY-FEEDBACK-007: source-pinned, explicitly invoked CPU-only synthetic science.
// Early material observations are simulated, not a production CyberSand ABI feature.
const BLOBS=["4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873","b24321ece46e839ec29458922a0031476c0700fe","20b9006a4535d1298a0f3e6ded42deeda63726ea"];
function rewrite(upstream,parent,transfer){
 let s=transfer(upstream,parent);
 function patch(a,b){const i=s.indexOf(a);if(i<0||s.indexOf(a,i+a.length)>=0)throw Error("Source drift/nonunique anchor: "+a.slice(0,65));s=s.slice(0,i)+b+s.slice(i+a.length)}
 patch('"distance4","train_selected"]','"distance4","train_selected","learned_history","learned_nohistory"]');
 patch("tick===5&&!w.materials.has(p)&&!w.ink.has(p)","(tick===2||tick===5)&&!w.materials.has(p)&&!w.ink.has(p)");
 patch("  if(tick===5){",
 '  if(tick===2&&sp.mode!=="irrelevant"&&(sp.mode==="early"?sp.seed%4!==0:sp.seed%4===0))event=w.event(tick,"close");\n'+
 '  if(tick===3&&w.materials.has(key(w.gate))&&w.epoch===1)event=w.event(tick,"open");\n  if(tick===5){');
 patch("   let invalidate=false;",
 '   if(logicalArm==="learned_history"||logicalArm==="learned_nohistory"){\n'+
 '    const x=[1,Number(w.epoch>=2),sp.horizon/128,w.gate[0]/50];\n'+
 '    if(logicalArm==="learned_nohistory")x[1]=0;\n'+
 '    prediction=logistic(x,logicalArm==="learned_history"?models.history.weights:models.nohistory.weights);\n'+
 '    arm=prediction>=0.5?"defer3":"overlap";\n   }\n   let invalidate=false;');
 patch("models.selected=selected;",
 'models.selected=selected;\n'+
 'function historyInput(sp,includeHistory){\n'+
 ' const pulse=sp.mode==="irrelevant"?false:(sp.mode==="early"?sp.seed%4!==0:sp.seed%4===0);\n'+
 ' return[1,Number(includeHistory&&pulse),sp.horizon/128,world(sp).gate[0]/50];\n}\n'+
 'function trainHistory(includeHistory){\n'+
 ' const w=[0,0,0,0],samples=[];\n'+
 ' for(const h of H)for(const seed of TRAIN)for(const mode of MODES){\n'+
 '  const sp=spec("train",seed,h,mode),cap=h===128?128:64;\n'+
 '  const a=episode(sp,"overlap",models,{cap}),b=episode(sp,"defer3",models,{cap});\n'+
 '  const d=utility(b)-utility(a);\n'+
 '  samples.push({x:historyInput(sp,includeHistory),y:Number(d>0),weight:Math.max(1,Math.min(12,Math.abs(d)/20))});\n'+
 ' }\n'+
 ' for(let iter=0;iter<900;iter++){\n'+
 '  const g=[0,0,0,0];for(const r of samples){\n'+
 '   const e=(logistic(r.x,w)-r.y)*r.weight/samples.length;\n'+
 '   for(let j=0;j<4;j++)g[j]+=e*r.x[j];\n'+
 '  }\n'+
 '  for(let j=0;j<4;j++)w[j]-=.08*(g[j]+.0005*w[j]);\n'+
 ' }\n'+
 ' const correct=samples.filter(r=>(logistic(r.x,w)>=.5)===Boolean(r.y)).length;\n'+
 ' return {weights:w,updates:900,train_n:samples.length,label_agreement:correct/samples.length,positive_labels:samples.filter(r=>r.y).length};\n'+
 '}\nmodels.history=trainHistory(true);models.nohistory=trainHistory(false);');
 patch('development_detail:results.filter(r=>["lazy","overlap","epoch","defer1","defer3","train_selected"].includes(r.arm))',
       'development_detail:results.filter(r=>["lazy","overlap","epoch","defer1","defer3","train_selected","learned_history","learned_nohistory"].includes(r.arm))');
 patch('check(episodes===2880&&results.length===2880','check(episodes===3264&&results.length===3264');
 patch('check(receipts>80000,"all receipts independently replayed");',
 'check(receipts>80000,"all receipts independently replayed");\n'+
 'check(models.history.train_n===432&&models.nohistory.train_n===432,"TRAIN-only samples");\n'+
 'check(models.history.updates===900&&models.nohistory.updates===900,"bounded optimizer");\n'+
 'check(results.filter(r=>r.arm==="learned_history").length===192&&results.filter(r=>r.arm==="learned_nohistory").length===192,"complete learned cohorts");');
 patch('study:"PATIENCE-TRANSFER-005"','study:"HISTORY-FEEDBACK-007"');
 patch('PATIENCE-TRANSFER-005: shifted DEV 72-decision trajectories','HISTORY-FEEDBACK-007: synthetic observable history paths');
 patch('(arm==="lazy"||arm==="overlap"||arm==="train_selected")','(arm==="overlap"||arm==="learned_history"||arm==="learned_nohistory")');
 patch('policy_selection:selected,','policy_selection:selected,history_train:{observed:models.history,ablation:models.nohistory},');
 return s;
}
function execute(upstream,parent,transfer){return(new Function(rewrite(upstream,parent,transfer)+"\nreturn forecastStudy();"))()}
function summarize(o){
 const e=o.evidence,rows=e.development_detail,arms=["overlap","defer3","train_selected","learned_history","learned_nohistory"];
 const by=new Map(rows.map(r=>[r.spec+"|"+r.cohort+"|"+r.arm,r])),paired={},negative_cases=[],positive_cases=[];
 const utility=r=>(r.arrived?1000:0)-2*r.steps-3*r.planner_calls-r.backtracks;
 for(const arm of arms.filter(a=>a!=="overlap")){
  const p={compared:0,goals_lost:0,goals_gained:0,utility_wins:0,utility_losses:0,utility_ties:0,utility_delta_total:0};
  for(const b of rows.filter(r=>r.arm==="overlap")){
   const a=by.get(b.spec+"|"+b.cohort+"|"+arm);if(!a)throw Error("Paired case missing "+b.spec+" "+arm);
   const delta=utility(a)-utility(b);p.compared++;p.utility_delta_total+=delta;
   if(b.arrived&&!a.arrived)p.goals_lost++;
   if(!b.arrived&&a.arrived)p.goals_gained++;
   if(delta>0){p.utility_wins++;positive_cases.push({arm,spec:b.spec,cohort:b.cohort,delta,baseline:b,alternative:a})}
   else if(delta<0){p.utility_losses++;negative_cases.push({arm,spec:b.spec,cohort:b.cohort,delta,baseline:b,alternative:a})}
   else p.utility_ties++;
  }
  if(p.compared!==192)throw Error("Incomplete comparison "+arm);
  p.mean_utility_delta=+(p.utility_delta_total/192).toFixed(6);paired[arm]=p;
 }
 return{study:"HISTORY-FEEDBACK-007",date:"2026-11-19",base_main:e.base_main,
  source_git_blobs:{forecast_utility_lab_js:BLOBS[0],observable_gate_defer_lab_js:BLOBS[1],patience_transfer_lab_js:BLOBS[2]},
  experiment:{observed_pulse:"Gate closes at tick 2 and opens at tick 3 on 75% of early and 25% of late world schedules (irrelevant: 0); main closure at tick 5",
   feature_vector:["bias","prior observed material pulse","declared horizon / 128","visible gate x / 50"],
   learned_action:"choose route-overlap invalidation or 3-step deferral at tick 5",future_mode_passed_to_policy:false,
   optimizer:"Cost-weighted batch logistic gradient descent, 900 bounded iterations per arm, 432 TRAIN examples per arm",
   transfer:"DEVELOPMENT shifted barrier geometry and early reopening ticks from PATIENCE-TRANSFER-005"},
  split:{train_seeds:e.train_seeds,development_seeds:e.development_seeds},
  training:e.history_train,prior_train_selection:e.policy_selection,
  development_episodes:e.development_episodes,independent_replay_episodes:e.independent_replay_episodes,
  replayed_action_receipts:e.action_receipts_replayed,tests_passed:e.tests_passed,v8_total_wall_ms:e.wall_ms,
  summary:Object.fromEntries(arms.map(a=>[a,e.summary[a]])),aggregates:e.aggregates.filter(a=>arms.includes(a.arm)),
  paired,negative_cases,positive_cases,examples:e.examples,svg_fnv_diagnostic:e.gallery_fnv64,
  boundaries:{sealed_structural_heldout_opened:false,native_runs:0,native_proxy_paint_censor:64,shadow128_is_synthetic_only:true,no_paid_compute:true,no_model_binaries_committed:true},
  limitations:["Synthesized material pulse is not an established host ABI field","Source-replay is JS-vs-JS, not independent Python/native replay",
   "FNV diagnostic fingerprints are not cryptographic signatures","V8 runner has no process RSS or per-inference percentiles; explicit Node CLI records Node wall/RSS",
   "Trained four-scalar weight vectors are included as text for reproducibility; no checkpoint binary is retained"]};
}
if(typeof module!=="undefined")module.exports={rewrite,execute,summarize,BLOBS};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
 if(process.argv.length!==3)throw Error("Explicit CPU invocation required: node history_feedback_lab.js /output-directory");
 const dir=__dirname,paths=["forecast_utility_lab.js","observable_gate_defer_lab.js","patience_transfer_lab.js"].map(n=>path.join(dir,n));
 const gitBlob=p=>{const b=fs.readFileSync(p);return crypto.createHash("sha1").update(Buffer.from("blob "+b.length+"\0")).update(b).digest("hex")};
 for(let i=0;i<3;i++)if(gitBlob(paths[i])!==BLOBS[i])throw Error("Pinned upstream blob mismatch: "+paths[i]);
 const parent=require("./observable_gate_defer_lab.js"),transfer=require("./patience_transfer_lab.js");
 const start=process.hrtime.bigint(),startRSS=process.memoryUsage().rss;
 const out=execute(fs.readFileSync(paths[0],"utf8"),parent.rewrite,transfer.rewriteTransfer),ev=summarize(out);
 ev.node_elapsed_ms=Number(process.hrtime.bigint()-start)/1e6;
 ev.node_maxrss_kib=process.resourceUsage().maxRSS;
 ev.node_rss_bytes=process.memoryUsage().rss;ev.node_rss_delta_bytes=ev.node_rss_bytes-startRSS;
 const dest=path.resolve(process.argv[2]);fs.mkdirSync(dest,{recursive:true});
 fs.writeFileSync(path.join(dest,"HISTORY_FEEDBACK_EVIDENCE.json"),JSON.stringify(ev,null,2)+"\n");
 fs.writeFileSync(path.join(dest,"HISTORY_FEEDBACK_PATHS.svg"),out.svg+"\n");
 console.log(JSON.stringify({episodes:ev.development_episodes,receipts:ev.replayed_action_receipts,tests:ev.tests_passed,paired:ev.paired,node_elapsed_ms:ev.node_elapsed_ms,node_maxrss_kib:ev.node_maxrss_kib},null,2));
}
