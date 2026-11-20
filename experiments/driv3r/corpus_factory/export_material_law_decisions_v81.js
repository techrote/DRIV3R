#!/usr/bin/env node
'use strict';
// Compact trainer-consumable indexes: no oracle certificate data, and no DEV/HELDOUT training leakage.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
function exportIndex(root,out){assert.ok(!fs.existsSync(out),'index dir already exists');fs.mkdirSync(out,{recursive:true});
 const result={schema:'driv3r/corpus_factory/material-law-decision-index-v81/1',source_manifest_sha256:hash(fs.readFileSync(path.join(root,'manifest.json'),'utf8')),
   synthetic_not_native_CyberSand:true,files:{},splits:{},consumer_contract:'Only TRAIN positive-weight model input/action pairs. DEV/HELDOUT evaluator-only. Reconstruct chronological past from stream rows [0,step) after excluding role/loss_weight/certificates. Early physical probe results and remote law-event receipts are legitimate past observations, not privileged law labels.'};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const pre=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';const src=fs.readFileSync(path.join(root,pre+'_material_law_stream.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  const rows=[],seen=new Set(),kinds={first_route:0,second_route:0,goal_memory:0},examples=[];
  src.forEach((ep,ei)=>{ep.steps.forEach((row,step)=>{
   if(!['first_route','second_route','goal_memory'].includes(row.role) ||row.loss_weight!==1)return;
   const past=ep.steps.slice(0,step).map(s=>({input:s.input,action:s.action}));
   const snapshot={input:row.input,action:row.action};const history=hash(past);
   const key=hash({kind:row.role,history,snapshot});if(seen.has(key))return;seen.add(key);kinds[row.role]++;
   const o={episode:ei,step,kind:row.role,action:row.action,history_sha256:history,current_input_sha256:hash(row.input),key_sha256:key,
     model_input_ref:`${pre}_material_law_stream.jsonl:${ei}:${step}`,difficulty:{history_actions:step,oldest_decisive_remote_receipt_age:step-94,physical_probe_age:step-1,post_obstruction:row.role==='goal_memory'&&ep.steps.some(s=>s.role==='physical_recovery')}};
   rows.push(o);examples.push({type:row.role,observation:row.input,action:row.action});
  });});
  const name=pre+'_material_law_decision_index.jsonl';const data=rows.map(x=>JSON.stringify(x)+'\n').join('');fs.writeFileSync(path.join(out,name),data,{flag:'wx'});
  result.files[name]={rows:rows.length,bytes:Buffer.byteLength(data),sha256:hash(data)};result.splits[split]={indexed:rows.length,unique_hashes:seen.size,by_kind:kinds,source_episodes:src.length};
 }
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});return result;}
if(require.main===module){assert.ok(process.argv[2]&&process.argv[3],'usage node export_material_law_decisions_v80.js CORPUS INDEXDIR');console.log(JSON.stringify(exportIndex(path.resolve(process.argv[2]),path.resolve(process.argv[3])).splits));}
module.exports={exportIndex};
