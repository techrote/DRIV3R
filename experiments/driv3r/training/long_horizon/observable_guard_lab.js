"use strict";
// OBSERVABLE-GUARD-006: closed-material route guard, train-only threshold selection, shifted-topology DEV replay.
// Scientific synthetic replica only; NOT CyberSand ABI or native game qualification.
const UPSTREAM_GIT_BLOB="4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873";
const PARENT_GIT_BLOB="b24321ece46e839ec29458922a0031476c0700fe";
const TRANSFER_GIT_BLOB="20b9006a4535d1298a0f3e6ded42deeda63726ea";
function rewriteHistory(base,parentRewrite,transferRewrite){
 let code=transferRewrite(base,parentRewrite);
 function patch(a,b){
  const i=code.indexOf(a);
  if(i<0||code.indexOf(a,i+a.length)>=0)throw Error("Missing/ambiguous pinned-source anchor: "+a);
  code=code.slice(0,i)+b+code.slice(i+a.length);
 }
 patch('"distance4","train_selected"]',
       '"distance4","train_selected","guard1","guard2","guard4","guard8","guard12","guard_train"]');
 patch('if(arm==="train_selected")arm=models.selected[sp.horizon];',
       'if(arm==="train_selected")arm=models.selected[sp.horizon];if(arm==="guard_train")arm=models.guard_selected[sp.horizon];');
 // No future mode/forecast used by guarded policies. Gate presence and distance to that cell are host-observable synthetic state.
 patch('||(arm.startsWith("distance")&&d>0&&d<=Number(arm.slice(8)))',
       '||(arm.startsWith("distance")&&d>0&&d<=Number(arm.slice(8)))||(arm.startsWith("guard")&&d>0&&d<=Number(arm.slice(5)))');
 // TRAIN calibration finishes before any DEVELOPMENT rollout; fixed candidate family including conservative overlap.
 patch('models.selected=selected;',
       'models.selected=selected;const guardSelected={},guardScores={};for(const h of H){let best=null,score={};for(const a of ["overlap","guard1","guard2","guard4","guard8","guard12"]){let n=0,sum=0;for(const seed of TRAIN)for(const mode of MODES){const r=episode(spec("train",seed,h,mode),a,models,{cap:h===128?128:64});sum+=utility(r);n++}const mean=sum/n;score[a]=+mean.toFixed(6);if(!best||mean>best.mean||(mean===best.mean&&a<best.arm))best={arm:a,mean};}guardSelected[h]=best.arm;guardScores[h]=score;}models.guard_selected=guardSelected;');
 patch('episodes===2880&&results.length===2880','episodes===4032&&results.length===4032');
 patch('"defer1","defer3","train_selected"].includes(r.arm)',
       '"guard1","guard2","guard4","guard8","guard12","guard_train","train_selected"].includes(r.arm)');
 patch('study:"PATIENCE-TRANSFER-005"','study:"OBSERVABLE-GUARD-006"');
 patch('transfer_design:{topology:',
       'guard_selection:guardSelected,guard_train_scores:guardScores,transfer_design:{topology:');
 patch('PATIENCE-TRANSFER-005: shifted DEV 72-decision trajectories',
       'OBSERVABLE-GUARD-006: shifted DEV gate trajectories');
 patch('(arm==="lazy"||arm==="overlap"||arm==="train_selected")',
       '(arm==="overlap"||arm==="guard4"||arm==="guard_train")');
 patch('const early=spec("development",3001,36,"early")',
       'check(H.every(h=>["overlap","guard1","guard2","guard4","guard8","guard12"].includes(models.guard_selected[h])),"guard choice exclusively TRAIN-calibrated");\nconst early=spec("development",3001,36,"early")');
 patch('check(episodes===4032&&results.length===4032,"full development factorial");',
       'check(episodes===4032&&results.length===4032,"full development factorial");\ncheck(results.filter(r=>r.arm==="guard_train").length===192,"guard train selection held fixed on DEV");');
 return code;
}
function execute(base,parentRewrite,transferRewrite){
 return (new Function(rewriteHistory(base,parentRewrite,transferRewrite)+"\nreturn forecastStudy();"))();
}
function summarize(result){
 const e=result.evidence,needed=["overlap","guard1","guard2","guard4","guard8","guard12","guard_train","train_selected"];
 const rows=e.development_detail,by=new Map(rows.map(r=>[r.spec+"|"+r.cohort+"|"+r.arm,r]));
 const losses=[],paired={};
 for(const arm of needed.filter(x=>x!=="overlap")){
  let wins=0,lost=0,both=0,neither=0,examples=0;
  for(const r of rows.filter(r=>r.arm==="overlap")){
   const v=by.get(r.spec+"|"+r.cohort+"|"+arm);
   if(!v)continue;
   if(r.arrived&&!v.arrived){lost++;losses.push({arm,spec:r.spec,cohort:r.cohort,baseline:r,alternative:v});}
   else if(!r.arrived&&v.arrived)wins++;
   else if(r.arrived&&v.arrived)both++;
   else neither++;
   examples++;
  }
  if(examples!==192)throw Error("Incomplete paired development comparison for "+arm+": "+examples);
  paired[arm]={compared:examples,goals_lost:lost,goals_gained:wins,both_arrive:both,neither_arrive:neither};
 }
 return{
  study:"OBSERVABLE-GUARD-006",date:"2026-11-19",source_study:e.study,base_main:e.base_main,
  train_seeds:e.train_seeds,development_seeds:e.development_seeds,transfer_design:e.transfer_design,
  limits:{native_proxy_logical_paints:64,shadow_128_synthetic_only:true,structural_heldout_opened:false,native_receipts:0,paid_compute:false},
  rule:"After observed gate closure, retain cached route until still-closed material is <= 1/2/4/8/12 cached moves away; recheck material every step. No future mode or forecast is passed to guard policy.",
  train_candidate_scores:e.guard_train_scores,train_selected:e.guard_selection,
  tests_passed:e.tests_passed,development_episodes:e.development_episodes,
  independent_replay_episodes:e.independent_replay_episodes,action_receipts_replayed:e.action_receipts_replayed,
  summary:Object.fromEntries(needed.map(k=>[k,e.summary[k]])),
  aggregates:e.aggregates.filter(x=>needed.includes(x.arm)),
  paired,negative_cases:losses,
  examples:e.examples.filter(x=>needed.includes(x.arm)),
  svg_diagnostic_fingerprint:e.gallery_fnv64,
  warnings:["Guard threshold selection uses TRAIN only and an observable synthetic gate occupancy/route distance; no new neural checkpoint","Future-conditioned non-ABI cue arms run as historical comparators but are excluded from deployable-arm summary","Current V8 runner does not expose RSS or per-inference percentiles; Node CLI records process RSS/resourceUsage when independently run","Receipt fingerprints in synthetic source are FNV diagnostics, not SHA-256 and not evidence of native ABI","Every DEVELOPMENT episode is independently replayed by the source simulator; cross-language Python/native replay not established"]
 };
}
if(typeof module!=="undefined")module.exports={rewriteHistory,execute,summarize,UPSTREAM_GIT_BLOB,PARENT_GIT_BLOB,TRANSFER_GIT_BLOB};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
 const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
 if(process.argv.length!==3)throw Error("Usage: node observable_guard_lab.js /absolute/output/dir (CPU explicit opt-in)");
 const dir=__dirname;
 const paths=["forecast_utility_lab.js","observable_gate_defer_lab.js","patience_transfer_lab.js"].map(p=>path.join(dir,p));
 function gitBlob(p){const b=fs.readFileSync(p);return crypto.createHash("sha1").update(Buffer.from("blob "+b.length+"\0")).update(b).digest("hex")}
 const hashes=paths.map(gitBlob),expected=[UPSTREAM_GIT_BLOB,PARENT_GIT_BLOB,TRANSFER_GIT_BLOB];
 for(let i=0;i<3;i++)if(hashes[i]!==expected[i])throw Error("Pinned source drift "+paths[i]+": "+hashes[i]);
 const parent=require("./observable_gate_defer_lab.js"),transfer=require("./patience_transfer_lab.js");
 const t=process.hrtime.bigint();
 const out=execute(fs.readFileSync(paths[0],"utf8"),parent.rewrite,transfer.rewriteTransfer);
 const evidence=summarize(out);
 evidence.node_wall_ms=Number(process.hrtime.bigint()-t)/1e6;
 evidence.node_maxrss_kib=process.resourceUsage().maxRSS;
 evidence.node_process_rss_bytes=process.memoryUsage().rss;
 const dest=path.resolve(process.argv[2]);fs.mkdirSync(dest,{recursive:true});
 fs.writeFileSync(path.join(dest,"OBSERVABLE_GUARD_EVIDENCE.json"),JSON.stringify(evidence,null,2)+"\n");
 fs.writeFileSync(path.join(dest,"OBSERVABLE_GUARD_PATHS.svg"),out.svg+"\n");
 console.log(JSON.stringify({study:evidence.study,tests:evidence.tests_passed,episodes:evidence.development_episodes,replayed:evidence.action_receipts_replayed,selection:evidence.train_selected,paired:evidence.paired,node_wall_ms:evidence.node_wall_ms,node_maxrss_kib:evidence.node_maxrss_kib},null,2));
}