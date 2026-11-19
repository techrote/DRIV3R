#!/usr/bin/env node
'use strict';
// Evaluator ONLY. These are symbolic expected-travel controls, NOT learned-model accuracy.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const ok=(v,m)=>{if(!v)throw Error('v79 controls: '+m)};
function run(dir){const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 const result={schema:'driv3r/corpus_factory/v79-evaluator-controls/1',synthetic_not_native_CyberSand:true,neural_policy_trained:false,splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const f=(split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY')+'_calibrated_certificates_EVALUATOR_ONLY.jsonl';
  const raw=fs.readFileSync(path.join(dir,f),'utf8');ok(hash(raw)===manifest.files[f].sha256,'certificate digest');const certs=raw.trimEnd().split('\n').map(JSON.parse);
  const s={episodes:certs.length,expected_action_correct_with_pilot:0,expected_action_correct_without_pilot_report_follow:0,physically_open_first_lane_with_pilot:0,physically_open_first_lane_without_pilot_report_follow:0,mean_expected_extra_actions_without_pilot:0,history_older_than_128_steps:0,paired_full_history_identical_future_counterfactuals:0};
  const groups=new Map();for(const c of certs){
   const reported=c.stage2_reported_open==='TOP'?'MOVE_N':'MOVE_S';const chosen=c.second_action_label;
   s.expected_action_correct_with_pilot++;s.expected_action_correct_without_pilot_report_follow+=Number(chosen===reported);
   s.physically_open_first_lane_with_pilot+=Number(c.first_choice_physically_open);
   s.physically_open_first_lane_without_pilot_report_follow+=Number(!c.hidden_midpoint_flip);
   const cost=c.expected_quarter_costs,optimal=Math.min(cost.TOP.expected_quarter_actions,cost.BOTTOM.expected_quarter_actions),baseline=cost[c.stage2_reported_open].expected_quarter_actions;
   s.mean_expected_extra_actions_without_pilot+=(baseline-optimal)/4;
   // Pilot is delivered at ticks 30..33, junction 2 at step 220+physical route; >128.
   if(c.second_decision_step-33>128)s.history_older_than_128_steps++;
   const key=`${c.split}:${c.world_index}:${c.stage1_open}:${c.stage2_reported_open}:${c.receipt_order}:${c.pilot_change_count}`;
   if(!groups.has(key))groups.set(key,{decision:new Set(),future:new Set(),labels:new Set()});
   const g=groups.get(key);g.decision.add(c.second_decision_input_sha256);g.future.add(c.hidden_midpoint_flip);g.labels.add(c.second_action_label);
  }
  for(const x of groups.values()){ok(x.decision.size===1&&x.future.size===2&&x.labels.size===1,'future-dependent label or degenerate counterfactual');s.paired_full_history_identical_future_counterfactuals++;}
  s.mean_expected_extra_actions_without_pilot/=s.episodes;
  ok(s.expected_action_correct_with_pilot===s.episodes&&s.expected_action_correct_without_pilot_report_follow===s.episodes/2,'label controls incorrect');
  ok(s.physically_open_first_lane_with_pilot===3*s.episodes/4&&s.physically_open_first_lane_without_pilot_report_follow===s.episodes/2,'physical controls incorrect');
  ok(s.history_older_than_128_steps===s.episodes,'pilot memory not long enough');result.splits[split]=s;
 }return result;
}
if(require.main===module){ok(process.argv[2],'usage score_calibrated_risk_v79.js CORPUS');console.log(JSON.stringify(run(path.resolve(process.argv[2])),null,2));}
module.exports={run};
