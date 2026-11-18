'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process');
const v24=require('./compact_trace_v24.js'),v23=require('./noncommutative_mission_v23.js');
function say(s){console.log('PASS '+s);}
const expected={
 'noncommutative_mission_v23.js':'e24ad8fb5e874452b86533041a0f1a91174039a9',
 'curriculum_pulse_v22.js':'b6f744216859eb91bec925a42f053153c85a7d0d',
 'recipe_grounded_v21.js':'19e8b18139263850930666cae119f58f8b993c82'
};
for(const [file,want] of Object.entries(expected)){
 const got=cp.execFileSync('git',['hash-object',path.join(__dirname,file)],{encoding:'utf8'}).trim();assert.equal(got,want,file);
}say('parent source Git blob identity pinned');
assert.throws(()=>v24.generate(path.dirname(__filename)),/refusing overwrite/);say('no overwrite of existing output');
for(const [split,group,variant] of [['TRAIN',0,0],['DEV',0,1],['HELDOUT',0,15]]){
 let count=0,oracle=[];const {header,summary}=v24.makeEpisode(split,group,variant,d=>{oracle.push(d);count++;});
 assert.equal(summary.terminal_success,true);assert.equal(summary.steps,count);
 assert.equal(v24.reconstruct(header,oracle),count);
 const reference=[];v23.replay(v23.makeRecord(split,group,variant),d=>reference.push({target:d.target,model_input_sha256:v24.sha(d.model_input)}));
 assert.deepEqual(oracle.map(d=>({target:d.target,model_input_sha256:d.model_input_sha256})),reference);
}say('per-decision roundtrip exactly matches v23 full-observation labels and hashes');
const dir=fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v24-'));try{
 const paths=[];for(let i=0;i<2;i++){
  const out=path.join(dir,'corpus-'+i);const manifest=v24.generate(out,{quick:true});
  assert.equal(manifest.episodes,48);assert.equal(manifest.decisions,5184);assert.equal(manifest.split_disjoint_programmes,true, 'optional field currently not emitted');
  for(const split of Object.keys(v24.COUNTS)){
   const headers=fs.readFileSync(path.join(out,split+'_episodes.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
   const decisions=fs.readFileSync(path.join(out,split+'_decisions.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
   const groups=new Map();for(const d of decisions){if(!groups.has(d.episode_id))groups.set(d.episode_id,[]);groups.get(d.episode_id).push(d);}
   for(const h of headers)assert.equal(v24.reconstruct(h,groups.get(h.episode_id)),h.steps);
  }
  paths.push({out,manifest});
 }
 assert.deepEqual(paths[0].manifest,paths[1].manifest);
 say('two complete deterministic quick generations and all 48 episode roundtrips');
 const m=paths[0].manifest;assert.ok(m.compact_bytes<m.raw_observation_bytes_estimate/2);
 console.log('QUICK_METRICS '+JSON.stringify({episodes:m.episodes,decisions:m.decisions,reference_model_input_bytes:m.raw_observation_bytes_estimate,compact_bytes:m.compact_bytes,reduction_percent:100*(1-m.compact_bytes/m.raw_observation_bytes_estimate)}));
}finally{fs.rmSync(dir,{recursive:true,force:true});}
