'use strict';
// DRIV3R Corpus Factory v24: lossless delta-encoded v23 demonstration corpus.
// Data/transport only; synthetic host. This is not CyberSand's native ABI.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v23=require('./noncommutative_mission_v23.js');
const COUNTS={TRAIN:12,DEV:6,HELDOUT:6};
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
function fileMeta(file,rows){const fd=fs.openSync(file,'r'),b=Buffer.allocUnsafe(1<<20),h=crypto.createHash('sha256');let n,bytes=0;try{while((n=fs.readSync(fd,b,0,b.length,null))>0){h.update(b.subarray(0,n));bytes+=n;}}finally{fs.closeSync(fd);}return {sha256:h.digest('hex'),bytes,rows};}
function splitObservation(input){const {history,...state}=input;return state;}
function reconstruct(header,rows){
 let hist=clone(header.initial_history),last=-1;
 for(const d of rows){
  if(d.episode_id!==header.episode_id||d.step!==last+1)throw Error('broken episode sequence');
  if(d.step===0&&(d.events_added.length||d.positions_added.length))throw Error('nonempty first delta');
  hist.events.push(...d.events_added);hist.positions.push(...d.positions_added);
  const input={};for(const key of header.observation_keys){input[key]=key==='history'?clone(hist):d.state[key];}
  if(sha(input)!==d.model_input_sha256)throw Error('lossless reconstruction violated at '+d.episode_id+' step '+d.step);
  if(!input.legal_actions.includes(d.target)||!d.receipt.accepted)throw Error('illegal target/rejected receipt in compact dataset');
  last=d.step;
 }
 return last+1;
}
function makeEpisode(split,group,variant,emit){
 const record=v23.makeRecord(split,group,variant),h=v23.createHost(record),initial=v23.observe(h);
 const header={schema:'driv3r/corpus_factory/compact-v24/episode',episode_id:record.episode_id,split,world_id:record.world_id,geometry_sha256:record.world.fingerprint,programme_bank_sha256:sha(v23.BANK.result[split]),counterfactual_group:`${split}-${group}`,difficulty:record.difficulty,initial_history:clone(initial.history),observation_keys:Object.keys(initial),host:'synthetic-v23-not-native'};
 let lastE=initial.history.events.length,lastP=initial.history.positions.length,step=0,moves=0,paints=0,erases=0,revisionCount=0;
 while(!h.done){
  if(step>=650)throw Error('oracle timeout '+record.episode_id);
  const input=v23.observe(h),target=v23.oracleAction(h);
  if(!input.legal_actions.includes(target))throw Error('illegal oracle '+target);
  const deltaE=clone(input.history.events.slice(lastE)),deltaP=input.history.positions.slice(lastP);
  if(input.history.events.length<lastE||input.history.positions.length<lastP)throw Error('history shortened unexpectedly');
  lastE=input.history.events.length;lastP=input.history.positions.length;
  const receipt=v23.apply(h,target);if(!receipt.accepted)throw Error('refused oracle '+target);
  if(target.startsWith('MOVE_'))moves++;if(target.startsWith('PAINT_'))paints++;if(target==='ERASE')erases++;
  if(input.mission_revision>revisionCount)revisionCount=input.mission_revision;
  emit({schema:'driv3r/corpus_factory/compact-v24/decision',episode_id:record.episode_id,step,state:splitObservation(input),events_added:deltaE,positions_added:deltaP,model_input_sha256:sha(input),target,receipt,training_weight:1},Buffer.byteLength(JSON.stringify(input)));step++;
 }
 if(!h.stop_accepted||h.invalid||h.pulses.length!==record.pulse_plan.length)throw Error('invalid v23 terminal '+record.episode_id);
 return {header,summary:{episode_id:record.episode_id,world_id:record.world_id,geometry_sha256:record.world.fingerprint,steps:step,moves,paints,erases,public_revisions:h.pulses.length,terminal_success:true}};
}
function generate(dest,opts={}){
 const quick=opts.quick===true;
 if(fs.existsSync(dest))throw Error('refusing overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},worlds=new Set(),programmeOwner=new Map(),seenAcross=new Map(),globalStats={episodes:0,decisions:0,raw_observation_bytes_estimate:0,compact_bytes:0};
 for(const [split,maxGroups] of Object.entries(COUNTS)){
  const groupCount=quick?1:maxGroups,episodesFile=split+'_episodes.jsonl',decisionsFile=split+'_decisions.jsonl',epPath=path.join(dest,episodesFile),decPath=path.join(dest,decisionsFile);
  const epFd=fs.openSync(epPath,'w'),decFd=fs.openSync(decPath,'w');let episodes=0,decisions=0,duplicates=0,erases=0,paints=0;
  const seen=new Map(),worldSplit=new Set(),firstState=[0,0,0,0];
  try{for(let group=0;group<groupCount;group++){
   // This explicitly verifies the four-way counterfactual alias before exporting.
   const family=v23.verifyFamily(split,group),fingerprint=family[0].world.fingerprint;
   if(worlds.has(fingerprint))throw Error('duplicate world across split/group');worlds.add(fingerprint);worldSplit.add(fingerprint);
   for(let variant=0;variant<16;variant++){
    const rec=family[variant],prg=rec.program.join('');if(programmeOwner.has(prg)&&programmeOwner.get(prg)!==split)throw Error('cross-split programme overlap');programmeOwner.set(prg,split);
    firstState[rec.private_target.program_state]++;
    let step=0;
    const {header,summary}=makeEpisode(split,group,variant,(d,fullBytes)=>{
     if(d.step!==step++)throw Error('out of order emitter');
     const old=seen.get(d.model_input_sha256);
     if(old!==undefined){if(old!==d.target)throw Error('contradictory model input labels');duplicates++;if(split==='TRAIN')d.training_weight=0;}else seen.set(d.model_input_sha256,d.target);
     const previousSplit=seenAcross.get(d.model_input_sha256);
     if(previousSplit&&previousSplit!==split)throw Error('model input leaks across TRAIN/DEV/HELDOUT');
     seenAcross.set(d.model_input_sha256,split);
     // The old v23 exporter serializes full event/position histories at every step.
     globalStats.raw_observation_bytes_estimate+=fullBytes;
     fs.writeSync(decFd,JSON.stringify(d)+'\n');decisions++;
    });
    fs.writeSync(epFd,JSON.stringify({...header,...summary})+'\n');episodes++;erases+=summary.erases;paints+=summary.paints;
   }
  }}finally{fs.closeSync(epFd);fs.closeSync(decFd);}
  if(firstState.some(n=>n!==episodes/4))throw Error('unbalanced first mission states');
  shards[episodesFile]=fileMeta(epPath,episodes);shards[decisionsFile]=fileMeta(decPath,decisions);
  splits[split]={world_groups:groupCount,episodes,decisions,unique_inputs:seen.size,exact_duplicate_inputs:duplicates,zero_weight_train_duplicates:split==='TRAIN'?duplicates:0,corrective_erases:erases,accepted_paints:paints,first_target_state_balance:firstState};
  globalStats.episodes+=episodes;globalStats.decisions+=decisions;globalStats.compact_bytes+=shards[episodesFile].bytes+shards[decisionsFile].bytes;
 }
 const manifest={schema:'driv3r/corpus_factory/compact-v24/manifest',label:'LOSSLESS synthetic-v23 corpus replay; NOT native CyberSand',generator:'compact_trace_v24.js',parent:'noncommutative_mission_v23.js',generator_sha256:sha(fs.readFileSync(__filename)),parent_sha256:sha(fs.readFileSync(path.join(__dirname,'noncommutative_mission_v23.js'))),quick,world_and_programme_split_protection:true,split_disjoint_programmes:true,reconstruction_sha256_verification_available:true,splits,...globalStats,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={COUNTS,sha,splitObservation,reconstruct,makeEpisode,generate};
if(require.main===module){if(!process.argv[2])throw Error('usage node compact_trace_v24.js NEW_EMPTY_DIRECTORY [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
