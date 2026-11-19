#!/usr/bin/env node
'use strict';
// Evaluator-only Bayes ceilings; NOT an SLM evaluation, NOT a policy trainer.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
function score(root){const out={schema:'driv3r/corpus_factory/three-lane-controls-v80/1',synthetic_not_native_CyberSand:true,neural_results:false,splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){const pre=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';const eps=fs.readFileSync(path.join(root,pre+'_three_lane_stream.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  const classes={first_route:new Map(),second_route:new Map(),goal_memory:new Map()},arrival={first:0,second:0,goal:0},totals={first:0,second:0,goal:0};
  for(const ep of eps)for(const row of ep.steps){if(!['first_route','second_route','goal_memory'].includes(row.role)||row.loss_weight!==1)continue;
   const map=classes[row.role],key=JSON.stringify(row.input);if(!map.has(key))map.set(key,new Map());const v=map.get(key);v.set(row.action,(v.get(row.action)||0)+1);
   // Trivially latest-arrived receipt policy is blind after 128+ action gap; it has no current receipts.
   assert.equal(row.input.reports.length,0);
   totals[row.role==='first_route'?'first':row.role==='second_route'?'second':'goal']++;
  }
  const ceiling={};for(const [name,group] of Object.entries(classes)){let correct=0,n=0,conflicts=0,distinct=new Set();for(const actions of group.values()){
   correct+=Math.max(...actions.values());const sum=[...actions.values()].reduce((a,b)=>a+b,0);n+=sum;
   if(actions.size>1)conflicts++;for(const a of actions.keys())distinct.add(a);
  }ceiling[name]={total:n,current_only_bayes_ceiling:correct/n,current_only_ambiguous_groups:conflicts,current_only_groups:group.size,distinct_actions:[...distinct].sort()};}
  out.splits[split]=ceiling;
 }
 return out;}
if(require.main===module){assert.ok(process.argv[2]);console.log(JSON.stringify(score(path.resolve(process.argv[2])),null,2));}module.exports={score};
