#!/usr/bin/env node
'use strict';
// Evaluation-only descriptive controls. Not a policy trainer or native game metric.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error('v78 SCORE '+m)};
function load(dir,file,m){const body=fs.readFileSync(path.join(dir,file),'utf8');ok(sha(body)===m.files[file].sha256,'stream hash');return body.trimEnd().split('\n').map(JSON.parse);}
function selectReport(inputs,section,mode){let reports=inputs.flatMap(x=>x.reports).filter(x=>x.section===section);
 if(mode==='last128')reports=inputs.slice(-128).flatMap(x=>x.reports).filter(x=>x.section===section);
 if(!reports.length)return 'MOVE_N';
 const latest={},last={};for(const r of reports){if(!latest[r.lane]||r.measure_tick>latest[r.lane].measure_tick)latest[r.lane]=r;last[r.lane]=r;}
 const chosen=mode==='arrival'?last:latest;return chosen.TOP.status==='OPEN'?'MOVE_N':'MOVE_S';}
function score(dir){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 ok(mf.schema==='driv3r/corpus_factory/relay-gate-replanning-v78/1','schema');
 const result={schema:'driv3r/corpus_factory/relay-v78-memory-controls/1',synthetic_not_native_CyberSand:true,neural_model_evaluated:false,splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY',streams=load(dir,prefix+'_relay_stream.jsonl',mf,mf),certs=load(dir,prefix+'_relay_certificates_EVALUATOR_ONLY.jsonl',mf);
  ok(streams.length===certs.length,'rows');
  const st={episodes:streams.length,first_choice:{source_time_correct:0,arrival_order_correct:0,last128_top_correct:0,current_only_top_correct:0},
   second_choice_report_intention:{source_time_correct:0,arrival_order_correct:0,last128_top_correct:0,current_only_top_correct:0},
   second_choice_physical_actual:{latest_report_correct:0,unpredictable_flip_count:0},
   matched_first_current_input_conflicts:0,matched_second_current_input_conflicts:0,distinct_full_history_action_pairs:0,
   mean_extra_actions_flip_vs_stable:0,mean_extra_actions_offpolicy_vs_teacher:0};
  const firstGroups=new Map(),secondGroups=new Map(),histories=new Map(),control=new Map();let flipExtra=0,flipN=0,wrongExtra=0,wrongN=0;
  for(let i=0;i<certs.length;i++){
   const c=certs[i],e=streams[i],s1=c.first_decision_step,s2=c.second_decision_step;
   const inp1=e.steps.slice(0,s1).map(x=>x.input),inp2=e.steps.slice(0,s2).map(x=>x.input),label1=c.first_action_label,label2=c.second_action_label;
   ok(s1===200&&s2>200,'decision index');
   ok(inp1.slice(-128).every(x=>x.reports.length===0)&&inp2.slice(-128).every(x=>x.reports.length===0),'receipt within last128');
   for(const [key,sec,label,inputs] of [['first_choice','S1',label1,inp1],['second_choice_report_intention','S2',label2,inp2]]){
    st[key].source_time_correct+=Number(selectReport(inputs,sec,'source_time')===label);
    st[key].arrival_order_correct+=Number(selectReport(inputs,sec,'arrival')===label);
    st[key].last128_top_correct+=Number(selectReport(inputs,sec,'last128')===label);
    st[key].current_only_top_correct+=Number('MOVE_N'===label);
   }
   st.second_choice_physical_actual.latest_report_correct+=Number(label2===(c.midpoint_flip?(label2==='MOVE_N'?'MOVE_S':'MOVE_N'):label2));
   st.second_choice_physical_actual.unpredictable_flip_count+=Number(c.midpoint_flip);
   const a=c.first_decision_input_sha256,b=c.second_decision_input_sha256;
   if(!firstGroups.has(a))firstGroups.set(a,new Set());firstGroups.get(a).add(label1);
   if(!secondGroups.has(b))secondGroups.set(b,new Set());secondGroups.get(b).add(label2);
   for(const [prefix,key,label] of [['S1',sha(inp1),label1],['S2',sha(inp2),label2]]){
    const h=prefix+':'+key;if(histories.has(h))ok(histories.get(h)===label,'contradictory full history label');histories.set(h,label);
   }
   const group=JSON.stringify([c.world_index,c.stage1_open,c.stage2_reported_open,c.receipt_order]);
   if(!control.has(group))control.set(group,{});
   control.get(group)[(c.midpoint_flip?'flip':'stable')+'_'+(c.offpolicy_stage1?'wrong':'teacher')]=e.steps.length;
  }
  for(const four of control.values()){
   ok(Object.keys(four).length===4,'missing paired physical controls');
   for(const suffix of ['teacher','wrong']){const delta=four['flip_'+suffix]-four['stable_'+suffix];ok(delta>0,'flip did not force measurable detour');flipExtra+=delta;flipN++;}
   for(const prefix of ['flip','stable']){const delta=four[prefix+'_wrong']-four[prefix+'_teacher'];ok(delta>0,'wrong route did not add real travel');wrongExtra+=delta;wrongN++;}
  }
  st.matched_first_current_input_conflicts=[...firstGroups.values()].filter(s=>s.size===2).length;
  st.matched_second_current_input_conflicts=[...secondGroups.values()].filter(s=>s.size===2).length;
  st.distinct_full_history_action_pairs=histories.size;
  st.mean_extra_actions_flip_vs_stable=flipExtra/flipN;
  st.mean_extra_actions_offpolicy_vs_teacher=wrongExtra/wrongN;
  ok(st.first_choice.source_time_correct===st.episodes,'source-time S1 error');
  ok(st.second_choice_report_intention.source_time_correct===st.episodes,'source-time S2 imitation error');
  ok(st.first_choice.current_only_top_correct*2===st.episodes,'S1 label imbalance');
  ok(st.second_choice_report_intention.current_only_top_correct*2===st.episodes,'S2 label imbalance');
  ok(st.matched_first_current_input_conflicts>=1&&st.matched_second_current_input_conflicts>=1,'ambiguous observations absent');
  result.splits[split]=st;
 }
 return result;
}
if(require.main===module){ok(process.argv[2],'usage score_relay_memory_v78.js DIR');console.log(JSON.stringify(score(path.resolve(process.argv[2])),null,2));}
module.exports={score};
