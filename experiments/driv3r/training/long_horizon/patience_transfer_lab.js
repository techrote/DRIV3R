"use strict";
// PATIENCE-TRANSFER-005: bounded source-pinned development topology/reopening transfer.
// No HELDOUT access, no production ABI, no automatic training trigger.
const UPSTREAM_GIT_BLOB="4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873";
const PARENT_GIT_BLOB="b24321ece46e839ec29458922a0031476c0700fe";
function rewriteTransfer(upstream,rewriteParent){
  let code=rewriteParent(upstream);
  function patch(a,b){const i=code.indexOf(a);if(i<0||code.indexOf(a,i+a.length)>=0)throw Error("Nonunique upstream anchor: "+a);code=code.slice(0,i)+b+code.slice(i+a.length)}
  patch("const gap=[3,4,12,13][sp.seed%4],materials=new Set();",
        "const gap=(sp.split==='development'?[2,5,11,14]:[3,4,12,13])[sp.seed%4],materials=new Set();");
  patch("for(let x=7;x<gatex;x++){materials.add(key([x,7]));materials.add(key([x,9]))}",
        "for(let x=7;x<gatex;x++){const top=sp.split==='development'?6:7,bottom=sp.split==='development'?10:9;materials.add(key([x,top]));materials.add(key([x,bottom]))}");
  patch('if(tick===8&&(sp.mode==="early"||sp.mode==="irrelevant"))event=w.event(tick,"open");',
        'if(tick===(sp.split==="development"&&sp.mode==="early"?[6,8,11,15][sp.seed%4]:8)&&(sp.mode==="early"||sp.mode==="irrelevant"))event=w.event(tick,"open");');
  patch('check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===36&&x.mode==="late"&&x.arm==="lazy").goals===0,"36 late lazy failure control");',
        'check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===36&&x.mode==="late"&&x.arm==="lazy").n===16,"36 late lazy cohort accounted");');
  patch('check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===72&&x.mode==="early"&&x.arm==="lazy").goals===16,"72 early lazy success control");',
        'check(aggregates.find(x=>x.cohort==="native64_proxy"&&x.horizon===72&&x.mode==="early"&&x.arm==="lazy").n===16,"72 early lazy cohort accounted");');
  patch('study:"OBSERVED-GATE-DEFER-004",version:VERSION,',
        'study:"PATIENCE-TRANSFER-005",version:VERSION,');
  patch('return{svg,evidence:{',
        'return{svg,evidence:{\n development_detail:results.filter(r=>["lazy","overlap","epoch","defer1","defer3","train_selected"].includes(r.arm)).map(r=>({spec:r.spec,arm:r.arm,cohort:r.cohort,arrived:r.arrived,steps:r.steps,paint_moves:r.paint_moves,backtracks:r.backtracks,planner_calls:r.planner_calls,illegal_prevented:r.illegal_prevented,action_counts:r.action_counts})),\n transfer_design:{topology:"DEV shifted barrier rows 6/10 and gap [2,5,11,14] vs TRAIN rows 7/9 and gap [3,4,12,13]",early_reopen_ticks:[6,8,11,15],dev_observation_only:true},');
  patch('OBSERVED-GATE-DEFER-004: synthetic 72-decision gate trajectories',
        'PATIENCE-TRANSFER-005: shifted DEV 72-decision trajectories');
  patch('if(seed===3001&&h===72&&(mode==="early"||mode==="late")&&(arm==="lazy"||arm==="overlap"||arm==="defer3")){',
        'if(h===72&&((seed===3003&&mode==="early")||(seed===3001&&mode==="late"))&&(arm==="lazy"||arm==="overlap"||arm==="train_selected")){');
  return code;
}
function execute(upstream,parentRewrite){return(new Function(rewriteTransfer(upstream,parentRewrite)+"\nreturn forecastStudy();"))()}
if(typeof module!=="undefined")module.exports={rewriteTransfer,execute};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("node:fs"),crypto=require("node:crypto"),path=require("node:path");
 const dir=__dirname,upPath=path.join(dir,"forecast_utility_lab.js"),parentPath=path.join(dir,"observable_gate_defer_lab.js");
 function sha(p){const b=fs.readFileSync(p);return crypto.createHash("sha1").update(Buffer.from("blob "+b.length+"\0")).update(b).digest("hex")}
 if(sha(upPath)!==UPSTREAM_GIT_BLOB||sha(parentPath)!==PARENT_GIT_BLOB)throw Error("Pinned source drift");
 if(process.argv.length!==3)throw Error("Usage: node patience_transfer_lab.js /absolute/output-directory");
 const t=process.hrtime.bigint(),res=execute(fs.readFileSync(upPath,"utf8"),require("./observable_gate_defer_lab.js").rewrite);
 const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
 const raw=res.evidence.development_detail;
 const lossCases=[];const byKey=new Map(raw.map(r=>[r.spec+"|"+r.cohort+"|"+r.arm,r]));
 for(const r of raw.filter(r=>r.arm==="overlap"&&r.arrived)){
  const other=byKey.get(r.spec+"|"+r.cohort+"|train_selected");
  if(other&&!other.arrived)lossCases.push({spec:r.spec,cohort:r.cohort,overlap:r,train_selected:other});
 }
 delete res.evidence.development_detail;
 res.evidence.transfer_goal_loss_cases=lossCases;
 res.evidence.node_wall_ms=Number(process.hrtime.bigint()-t)/1e6;
 res.evidence.node_maxrss_kib=process.resourceUsage().maxRSS;
 res.evidence.process_rss_bytes=process.memoryUsage().rss;
 fs.writeFileSync(path.join(out,"PATIENCE_TRANSFER_EVIDENCE.json"),JSON.stringify(res.evidence,null,2)+"\n");
 fs.writeFileSync(path.join(out,"PATIENCE_TRANSFER_TRAJECTORIES.svg"),res.svg+"\n");
 console.log(JSON.stringify({study:res.evidence.study,tests:res.evidence.tests_passed,episodes:res.evidence.development_episodes,receipts:res.evidence.action_receipts_replayed,summary:res.evidence.summary,selection:res.evidence.policy_selection,node_wall_ms:res.evidence.node_wall_ms,node_maxrss_kib:res.evidence.node_maxrss_kib},null,2));
}
