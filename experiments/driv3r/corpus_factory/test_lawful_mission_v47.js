'use strict';
/* Independent corpus v47 full-host and label audit. Does NOT invoke v47.teacher,
 * v47.rollout, v47.accepted or v47.scenario. Only shared host + input serializer. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v47=require('./lawful_mission_v47.js'),v46=require('./mission_memory_v46.js'),oldAudit=require('./test_mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(b,m)=>{if(!b)throw Error(m)};
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const mode=i=>{const r=i.durable.filter(e=>e[1]==='TOOL_LAW');ok(r.length>=2,'missing actual tool law');return r.at(-1)[2]};
const flip=x=>x==='NORMAL'?'SWAPPED':'NORMAL';
function independentAction(i){const physical=oldAudit.independentAction(i);return physical.startsWith('PAINT_')&&mode(i)==='SWAPPED'?'PAINT_'+(physical.endsWith('A')?'B':'A'):physical;}
function hostStep(h,a){ok(host.legal(h).includes(a),'proposed command absent from host legal');
 const physical=a.startsWith('PAINT_')&&h.tool_mode==='SWAPPED'?'PAINT_'+(a.endsWith('A')?'B':'A'):a;
 const before=h.stage,prior=h.accepted;ok(host.apply(h,physical).accepted,'physical host rejected action');ok(h.accepted===prior+1,'no host accepted action');
 const ack=h.events.filter(e=>e.kind==='ACTION_ACK').at(-1);ok(ack.at===h.accepted,'no host receipt');ack.action=a;
 if(before!==2&&h.stage===2)h.events.push({kind:'MISSION_COMPLETE',accepted:true,at:h.accepted});
 if(before===0&&h.stage===1){h.tool_mode=h.stage_flip?flip(h.tool_mode):h.tool_mode;h.events.push({kind:'TOOL_LAW',at:h.accepted,accepted:true,mode:h.tool_mode});}
}
function validateObservation(i,h){ok(i.v==='synthetic-v47-historical-tool-law-and-mission','observation identity wrong');
 for(const forbidden of ['target','stage','tool_mode','goal','source_seed','geometry_sha256','oracle','action','weight'])ok(!(forbidden in i),'privileged observation field '+forbidden);
 const asV46=JSON.parse(JSON.stringify(i));asV46.v='synthetic-v46-historical-missions';
 oldAudit.checkObservation(asV46,h);
 const laws=i.durable.filter(x=>x[1]==='TOOL_LAW');ok(laws.length===h.events.filter(x=>x.kind==='TOOL_LAW').length,'tool-law receipts missing');
 ok(laws.every(x=>x[2]==='NORMAL'||x[2]==='SWAPPED'),'bad law code');
 ok(mode(i)===h.tool_mode,'tool-law receipt inconsistent with active synthetic adapter');
 ok(i.recent.length<=32&&i.recent_positions.length<=32,'recent window leaked');
}
function audit(dir,{quick=false}={}){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));ok(mf.quick===quick&&mf.synthetic_not_native_CyberSand,'manifest task mismatch');
 ok(mf.schema==='driv3r/corpus_factory/historical-mission-tool-law-v47/manifest','schema mismatch');
 ok(H(fs.readFileSync(path.join(__dirname,mf.generator)))===mf.generator_sha256,'generator hash mismatch');
 for(const p of mf.parents)ok(H(fs.readFileSync(path.join(__dirname,p.file)))===p.sha256,'parent hash mismatch '+p.file);
 const files={};for(const [f,s] of Object.entries(mf.shards)){const bytes=fs.readFileSync(path.join(dir,f));ok(H(bytes)===s.sha256&&bytes.length===s.bytes,'shard hash mismatch '+f);files[f]=read(path.join(dir,f));ok(files[f].length===s.rows,'row mismatch '+f)}
 let decisions=0,episodes=0,mission=0,gates=0,tool=0,staleMission=0,staleLaw=0,erases=0,paintA=0,paintB=0,stageFlips=0;
 const geometry=new Map(),inputs=new Map();
 const settings=[[false,false,'opposite','NORMAL','NORMAL'],[false,true,'opposite','NORMAL','NORMAL'],
  [true,false,'opposite','NORMAL','NORMAL'],[true,true,'opposite','NORMAL','NORMAL'],
  [false,false,'same','NORMAL','NORMAL'],[false,false,'opposite','SWAPPED','NORMAL'],
  [false,true,'opposite','SWAPPED','NORMAL'],[false,false,'opposite','NORMAL','SWAPPED']];
 for(const [split,s] of Object.entries(mf.splits)){
  const policy=files[split+'_policy.jsonl'],challenges=files[split+'_challenge_EVALUATOR_ONLY.jsonl'];
  ok(challenges.length===s.worlds&&s.host_successful_episodes===s.worlds*8,'challenge or episode count incorrect');
  let pointer=0,zero=0;
  for(let j=0;j<s.worlds;j++){
   const w=v46.candidate(split,j).w,ch=challenges[j],stageFlip=w.index%2===0;
   ok(ch.source_world_index===w.index&&ch.source_seed===w.seed&&ch.geometry_sha256===w.geometry_sha256,'bad source provenance');
   ok(!geometry.has(w.geometry_sha256)||geometry.get(w.geometry_sha256)===split,'cross-split geometry leakage');geometry.set(w.geometry_sha256,split);
   ok(ch.stage_law_flip===stageFlip,'wrong law schedule');if(stageFlip)stageFlips++;
   const hs=settings.map(z=>v47.prep(w,...z,stageFlip));const start=hs.map(v47.observe);
   ok(H(v47.withoutMission(start[0]))===H(v47.withoutMission(start[1])),'causal mission branch leakage');
   ok(H(v47.withoutLaw(start[0]))===H(v47.withoutLaw(start[5])),'causal law branch leakage');
   ok(H(v47.withoutLaw(start[0]))===H(v47.withoutLaw(start[7])),'stale law control leakage');
   const starts=start.map(independentAction);
   ok(starts[0]!==starts[1]&&starts[0]!==starts[2]&&starts[0]===starts[4]&&starts[0]===starts[5]&&starts[0]===starts[7],'causal fork initial actions invalid');
   ok(H(starts)===H(ch.first_actions),'initial challenge answers incorrect');
   ok(ch.historical_law_age_actions>=80&&ch.historical_mission_age_actions>=256,'historical ageing gate incorrect');
   let firstPaint=[];
   for(let arm=0;arm<8;arm++){
    const h=hs[arm],length=ch.episode_steps[arm];ok(length>0&&length<=256,'invalid episode length');
    const thisPaint=[];let doneAt=0;
    for(let turn=0;turn<length;turn++){
     const row=policy[pointer++];ok(row&&Object.keys(row).sort().join(',')==='action,input,weight','bad policy row shape');
     const input=v47.observe(h);ok(H(input)===H(row.input),'host observation does not match dataset');
     validateObservation(row.input,h);
     const expected=independentAction(row.input);ok(row.action===expected,'independent rule/survey planner disagrees '+split+'/'+j+'/'+arm+'/'+turn);
     ok(row.input.legal.includes(row.action),'illegal supervised command');
     const hash=H(row.input),prior=inputs.get(hash);ok(!prior||prior.split===split,'cross-split input contamination');ok(!prior||prior.action===row.action,'conflicting action labels');
     if(prior&&split==='TRAIN'){ok(row.weight===0,'TRAIN duplicate has nonzero loss weight');zero++;}else{ok(row.weight===1,'unexpected weight');inputs.set(hash,{split,action:row.action});}
     if(row.action==='ERASE')erases++;if(row.action==='PAINT_A')paintA++;if(row.action==='PAINT_B')paintB++;
     if(row.action.startsWith('PAINT_'))thisPaint.push({input:row.input,action:row.action,mode:mode(row.input)});
     hostStep(h,row.action);decisions++;doneAt++;
    }
    ok(h.done&&h.success&&doneAt===length,'host terminal unsuccessful');
    ok(thisPaint.length===2,'not exactly two paint operations');ok(thisPaint[0].mode===settings[arm][3],'wrong initial law');
    ok(thisPaint[1].mode===(stageFlip?flip(settings[arm][3]):settings[arm][3]),'stage law transition not reflected');
    ok(ch.material_first_commands[arm]===thisPaint[0].action&&ch.material_second_commands[arm]===thisPaint[1].action,'paint challenge labels incorrect');
    firstPaint.push(thisPaint[0]);episodes++;
   }
   ok(H(v47.withoutLaw(firstPaint[0].input))===H(v47.withoutLaw(firstPaint[5].input)),'first paint branch leaks other than law receipts');
   ok(firstPaint[0].action!==firstPaint[5].action,'causal material mapping failed');
   ok(H(v47.withoutLaw(firstPaint[0].input))===H(v47.withoutLaw(firstPaint[7].input))&&firstPaint[0].action===firstPaint[7].action,'stale-law negative failed');
   mission++;gates++;tool++;staleMission++;staleLaw++;
  }
  ok(pointer===policy.length&&zero===s.duplicate_zero_weight,'split accounting mismatch');
 }
 ok(decisions===mf.total_policy_decisions&&geometry.size===mf.unique_world_geometries,'total accounting mismatch');
 return {verified_decisions:decisions,successful_host_episodes:episodes,worlds:geometry.size,
  causal_mission_pairs:mission,causal_remote_gate_pairs:gates,causal_tool_law_pairs:tool,
  stale_mission_controls:staleMission,stale_law_controls:staleLaw,
  worlds_with_mid_episode_law_flip:stageFlips,erase_actions:erases,paint_A_commands:paintA,paint_B_commands:paintB,
  cross_split_geometry_or_input_overlap:0,contradictory_labels:0};
}
if(require.main===module){const [one,two,quickArg]=process.argv.slice(2);ok(one,'usage node test_lawful_mission_v47.js DATA [SECOND_DATA] [--quick]');
 const quick=two==='--quick'||quickArg==='--quick',report=audit(one,{quick});
 if(two&&two!=='--quick'){
  audit(two,{quick});const f=fs.readdirSync(one).sort(),g=fs.readdirSync(two).sort();ok(H(f)===H(g),'nonrepeatable file list');
  for(const name of f)ok(H(fs.readFileSync(path.join(one,name)))===H(fs.readFileSync(path.join(two,name))),'nonrepeatable bytes '+name);
 }console.log(JSON.stringify(report,null,2));}
module.exports={audit,independentAction,hostStep};