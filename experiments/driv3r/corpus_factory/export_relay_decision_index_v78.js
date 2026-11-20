#!/usr/bin/env node
'use strict';
// Index-only deduplication; no privileged future state enters supervised features.
// Only TRAIN index may be consumed for supervised learning. Other indexes evaluator-only.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(x,m)=>{if(!x)throw Error('v78 index '+m);};
function index(source,destination){check(!fs.existsSync(destination),'output exists');const manifest=JSON.parse(fs.readFileSync(path.join(source,'manifest.json')));
 check(manifest.schema==='driv3r/corpus_factory/relay-gate-replanning-v78/1','manifest schema');fs.mkdirSync(destination,{recursive:true});
 const output={schema:'driv3r/corpus_factory/relay-decision-index-v78/1',synthetic_not_native_CyberSand:true,
  source_manifest_sha256:sha(fs.readFileSync(path.join(source,'manifest.json'),'utf8')),
  note:'Only TRAIN index is trainable. Index entries contain observable historical hashes, reference coordinates and supervised commands; no oracle future event or secret gate state.',files:{},splits:{}};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  const sourceStream=prefix+'_relay_stream.jsonl',sourceCert=prefix+'_relay_certificates_EVALUATOR_ONLY.jsonl';
  const get=name=>{const body=fs.readFileSync(path.join(source,name),'utf8');check(sha(body)===manifest.files[name].sha256,'mismatch '+name);return body.trimEnd().split('\n').map(JSON.parse);};
  const streams=get(sourceStream),certs=get(sourceCert),unique=new Map();
  check(streams.length===certs.length,'pair mismatch');
  streams.forEach((episode,epi)=>{
   const c=certs[epi];for(const [stage,step] of [['S1',c.first_decision_step],['S2',c.second_decision_step]]){
    const row=episode.steps[step];check(row.weight===(stage==='S1'&&c.offpolicy_stage1?0:1),'unexpected supervised weight');
    if(row.weight!==1)continue;
    const inputHistory=episode.steps.slice(0,step+1).map(x=>x.input);
    const key=stage+':'+sha(inputHistory)+':'+row.action;
    const checkKey=stage+':'+sha(inputHistory);let same=[...unique.values()].find(v=>v.stage_history===checkKey);
    check(!same||same.action===row.action,'conflicting actions for identical history');
    const item=unique.get(key);
    if(item){item.duplicate_references++;}
    else unique.set(key,{schema:'driv3r/corpus_factory/relay-observable-decision-ref-v78/1',split,stage,stage_history:checkKey,
     action:row.action,source_stream:sourceStream,representative_episode:epi,representative_step:step,
     provenance:{geometry_sha256:c.geometry_sha256,source_episode_index:epi},
     difficulty:{observed_history_actions:step,distinct_gate_stages:2,stage},
     scenario_tags:['partial_observation','stale_remote_receipts','dynamic_obstructions','physical_route_recovery'],
     duplicate_references:0});
   }
  });
  const file=prefix+'_relay_decision_index.jsonl';const entries=[...unique.values()].sort((a,b)=>a.stage_history.localeCompare(b.stage_history));
  const bytes=entries.map(x=>JSON.stringify(x)+'\n').join('');fs.writeFileSync(path.join(destination,file),bytes,{flag:'wx'});
  output.files[file]={rows:entries.length,bytes:Buffer.byteLength(bytes),sha256:sha(bytes)};
  const allWeight=entries.reduce((n,x)=>n+x.duplicate_references+1,0);
  output.splits[split]={indexed_supervised_decisions:allWeight,distinct_decisions:entries.length,deduplicated_occurrences:allWeight-entries.length};
 }
 fs.writeFileSync(path.join(destination,'manifest.json'),JSON.stringify(output,null,2)+'\n',{flag:'wx'});return output;
}
if(require.main===module){check(process.argv[2]&&process.argv[3],'usage node export_relay_decision_index_v78.js CORPUS_DIR NEW_INDEX_DIR');
 console.log(JSON.stringify(index(path.resolve(process.argv[2]),path.resolve(process.argv[3])),null,2));}
module.exports={index};
