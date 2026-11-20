#!/usr/bin/env node
'use strict';
// EVALUATOR-ONLY v77: behavioural memory ablations, never training features.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
function check(x,m){if(!x)throw Error('v77 SCORE '+m);}
function load(f,m){const body=fs.readFileSync(f,'utf8');check(sha(body)===m.files[path.basename(f)].sha256,'shard hash');return body.trimEnd().split('\n').map(JSON.parse);}
function score(dir){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 check(mf.schema==='driv3r/corpus_factory/dynamic-gate-memory-v77/1','manifest');
 const result={schema:'driv3r/corpus_factory/gate-report-memory-controls-v77/1',synthetic_not_native_CyberSand:true,model_inference:false,splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const name=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  const streams=load(path.join(dir,name+'_gate_memory_stream.jsonl'),mf),certs=load(path.join(dir,name+'_gate_memory_certificates_EVALUATOR_ONLY.jsonl'),mf);
  const stats={episodes:streams.length,source_time_correct:0,arrival_order_correct:0,current_only_top_correct:0,last128_top_correct:0,conflicting_current_groups:0,distinct_full_histories:0};
  const groups=new Map(),fullHistory=new Map();
  streams.forEach((episode,i)=>{
   const c=certs[i],step=c.decision_step,inputs=episode.steps.slice(0,step).map(x=>x.input);
   check(step===160&&inputs.length===160,'decision or history');
   const reports=inputs.flatMap(x=>x.reports);
   check(reports.length===4,'report count');
   const latest={},last={};
   for(const r of reports){
    check(['TOP','BOTTOM'].includes(r.lane)&&[0,10].includes(r.measure_tick),'source-time semantics');
    if(!latest[r.lane]||r.measure_tick>latest[r.lane].measure_tick)latest[r.lane]=r;
    last[r.lane]=r;
   }
   check(latest.TOP.measure_tick===10&&latest.BOTTOM.measure_tick===10,'fresh report lost');
   check(latest.TOP.status!==latest.BOTTOM.status,'not a discriminative physical gate report');
   const sourceChoice=latest.TOP.status==='OPEN'?'MOVE_N':'MOVE_S';
   const arrivalChoice=last.TOP.status==='OPEN'?'MOVE_N':'MOVE_S';
   const recent=inputs.slice(-128).flatMap(x=>x.reports);
   check(recent.length===0,'history still fits in last128');
   const label=c.first_optimal_action;
   stats.source_time_correct+=Number(sourceChoice===label);
   stats.arrival_order_correct+=Number(arrivalChoice===label);
   stats.current_only_top_correct+=Number('MOVE_N'===label);
   stats.last128_top_correct+=Number('MOVE_N'===label);
   const h=sha(inputs),local=sha(episode.steps[step].input);
   if(!groups.has(local))groups.set(local,new Set());groups.get(local).add(label);
   if(fullHistory.has(h))check(fullHistory.get(h)===label,'contradictory full-history supervision');
   fullHistory.set(h,label);
  });
  stats.conflicting_current_groups=[...groups.values()].filter(x=>x.size===2).length;
  stats.distinct_full_histories=fullHistory.size;
  check(stats.source_time_correct===stats.episodes,'source history not sufficient');
  check(stats.current_only_top_correct*2===stats.episodes,'critical labels not balanced');
  check(stats.arrival_order_correct*4===stats.episodes*3,'arrival-order shortcut result changed');
  check(stats.conflicting_current_groups===mf.splits[split].worlds.length,'paired labels absent');
  result.splits[split]=stats;
 }
 return result;
}
if(require.main===module){check(process.argv[2],'usage score_gate_report_memory_v77.js DATA_DIR');console.log(JSON.stringify(score(path.resolve(process.argv[2])),null,2));}
module.exports={score};
