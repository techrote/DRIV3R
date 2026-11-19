#!/usr/bin/env node
'use strict';
// Compact supervised decision INDEX only; no evaluator fields, future flips or label certificates.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error('v79 export '+m)};
function exportIndex(corpus,out){ok(!fs.existsSync(out),'output exists');fs.mkdirSync(out,{recursive:true});const m=JSON.parse(fs.readFileSync(path.join(corpus,'manifest.json'),'utf8'));
 const result={schema:'driv3r/corpus_factory/calibrated-decision-index-v79/1',parent_corpus_sha256:sha(fs.readFileSync(path.join(corpus,'manifest.json'))),synthetic_not_native_CyberSand:true,
  model_input_contract:'History must be obtained solely from source episode inputs/actions up to (excluding) indexed decision, never certificates or future action outcomes.',files:{},splits:{}};
 const global=new Set();for(const split of ['TRAIN','DEV','HELDOUT']){const prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  const s=prefix+'_calibrated_stream.jsonl',c=prefix+'_calibrated_certificates_EVALUATOR_ONLY.jsonl';
  const verify=name=>{const raw=fs.readFileSync(path.join(corpus,name),'utf8');ok(sha(raw)===m.files[name].sha256,'source digest '+name);return raw.trimEnd().split('\n').map(JSON.parse);};
  const rows=verify(s),certs=verify(c);ok(rows.length===certs.length,'source row count');
  const unique=new Map();let total=0,mask=0;for(let i=0;i<rows.length;i++){const ep=rows[i].steps,cert=certs[i];
   for(const [stage,step] of [['S1',cert.first_decision_step],['S2',cert.second_decision_step]]){const row=ep[step];ok(row&&['MOVE_N','MOVE_S'].includes(row.action),'decision not a route action');
    const history=ep.slice(0,step).map(x=>({input:x.input,action:x.action})),priorHash=sha(history),key=stage+':'+priorHash;
    total++;if(unique.has(key)){const existing=unique.get(key);ok(existing.action===row.action&&existing.input_sha256===sha(row.input),'contradictory labels at identical observed history');
     ok(row.weight===0,'repeat decision was trained');existing.matched_replays++;mask++;continue;}
    ok(row.weight===1,'unique decision masked');const entry={split,stage,source_episode:i,world_index:cert.world_index,geometry_sha256:cert.geometry_sha256,
     decision_step:step,action:row.action,input_sha256:sha(row.input),prior_history_sha256:priorHash,matched_replays:1};unique.set(key,entry);
   }
  }
  const list=[...unique.values()],file=prefix+'_calibrated_decision_index.jsonl',body=list.map(x=>JSON.stringify(x)+'\n').join('');
  fs.writeFileSync(path.join(out,file),body);result.files[file]={rows:list.length,bytes:Buffer.byteLength(body),sha256:sha(body)};
  result.splits[split]={raw_decisions:total,deduplicated_decisions:list.length,loss_masked_repeated_decisions:mask,stage1:list.filter(x=>x.stage==='S1').length,stage2:list.filter(x=>x.stage==='S2').length};
  for(const e of list){const id=split+'/'+e.stage+'/'+e.prior_history_sha256;ok(!global.has(id),'split duplicate');global.add(id);}
 }
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(result,null,2)+'\n');return result;
}
if(require.main===module){ok(process.argv[2]&&process.argv[3],'usage export_calibrated_decisions_v79.js CORPUS OUT');console.log(JSON.stringify(exportIndex(path.resolve(process.argv[2]),path.resolve(process.argv[3])),null,2));}
module.exports={exportIndex};
