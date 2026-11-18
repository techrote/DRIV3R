'use strict';
/* v48 independent full-host replay and observable-only planner audit.
 * Does NOT call v48.teacher, v48.rollout, v48.accepted, v48.scenario.
 * Audited source preludes are reconstructed via v48.prep, then checked against
 * original host receipts and physical probe results. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v48=require('./empirical_tool_v48.js'),v46=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const original=require('./test_mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error(m)};
const records=(file)=>fs.readFileSync(file,'utf8').trim().split('\n').map(JSON.parse);
const flip=x=>x==='A'?'B':'A';
function independentAction(i){const action=original.independentAction(i);
 if(!action.startsWith('PAINT_'))return action;
 const probes=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[2]==='PAINT_A');
 ok(probes.length===2,'incomplete calibration');
 return probes.at(-1)[3]==='B'?'PAINT_'+flip(action.slice(-1)):action;
}
function replayStep(h,command){ok(host.legal(h).includes(command),'host legal absent');
 const physical=command.startsWith('PAINT_')&&h.tool_mode==='SWAPPED'?'PAINT_'+flip(command.slice(-1)):command;
 const at=h.accepted,was=h.stage;ok(host.apply(h,physical).accepted,'host refused replay action');
 ok(h.accepted===at+1,'did not advance host');
 const ack=h.events.filter(e=>e.kind==='ACTION_ACK').at(-1);ok(ack&&ack.at===h.accepted,'missing receipt');ack.action=command;
 if(was!==2&&h.stage===2)h.events.push({kind:'MISSION_COMPLETE',accepted:true,at:h.accepted});
}
function validateProbeHistory(h){const list=h.events.filter(e=>e.kind==='PAINT_RESULT');ok(list.length===2,'exactly two calibrations');
 for(const e of list){const matching=h.events.find(x=>x.kind==='ACTION_ACK'&&x.at===e.at);ok(matching&&matching.action==='PAINT_A'&&matching.pos===h.w.s,'probe is not host executed');
  const erase=h.events.find(x=>x.kind==='ACTION_ACK'&&x.at===e.at+1);ok(erase&&erase.action==='ERASE'&&erase.pos===h.w.s,'probe not erased');
  ok(e.material==='A'||e.material==='B','invalid measured material');}
 ok(h.painted[h.w.s]===undefined,'prelude probe contaminated world');
 ok(!h.events.some(e=>e.kind==='TOOL_LAW'),'explicit oracle law leaked to host');
}
function validateInput(i,h){ok(i.v==='synthetic-v48-empirical-paint-calibration','wrong contract version');
 for(const x of ['target','stage','tool_mode','goal','source_seed','geometry_sha256','oracle','action','weight'])ok(!(x in i),'privileged field '+x);
 ok(i.durable.every(e=>e[1]!=='TOOL_LAW'),'symbolic TOOL_LAW shortcut');
 const receipts=i.durable.filter(e=>e[1]==='PAINT_RESULT');ok(receipts.length===2,'missing empirical evidence');
 const actual=h.events.filter(e=>e.kind==='PAINT_RESULT').map(e=>[e.at,'PAINT_RESULT',e.issued,e.material]);
 ok(H(receipts)===H(actual),'reported physical outcome differs from host');
 const compact=JSON.parse(JSON.stringify(i));compact.v='synthetic-v46-historical-missions';
 compact.durable=compact.durable.filter(e=>e[1]!=='PAINT_RESULT'&&e[1]!=='TOOL_DIAGNOSTIC');
 original.checkObservation(compact,h);
}
function audit(dir,{quick=false}={}){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 ok(mf.schema==='driv3r/corpus_factory/empirical-tool-calibration-v48/manifest'&&mf.quick===quick&&mf.synthetic_not_native_CyberSand,'wrong manifest');
 ok(H(fs.readFileSync(path.join(__dirname,mf.generator)))===mf.generator_sha256,'generator mismatch');
 for(const p of mf.parents)ok(H(fs.readFileSync(path.join(__dirname,p.file)))===p.sha256,'parent mismatch '+p.file);
 const files={};for(const [f,s] of Object.entries(mf.shards)){
  const b=fs.readFileSync(path.join(dir,f));ok(H(b)===s.sha256&&b.length===s.bytes,'bad shard hash '+f);
  files[f]=records(path.join(dir,f));ok(files[f].length===s.rows,'bad row count '+f);
 }
 let decisions=0,episodes=0,physicalPairs=0,missionPairs=0,gatePairs=0,negativeMission=0,negativeProbe=0,probeOps=0,paintA=0,paintB=0,erases=0;
 const owners=new Map(),labels=new Map();
 for(const [split,s] of Object.entries(mf.splits)){
  const pol=files[split+'_policy.jsonl'],chall=files[split+'_challenge_EVALUATOR_ONLY.jsonl'];
  ok(chall.length===s.worlds&&s.host_successful_episodes===s.worlds*8,'episode count');
  let p=0,zero=0;
  for(let idx=0;idx<s.worlds;idx++){
   const w=v46.candidate(split,idx).w,meta=chall[idx];
   ok(meta.source_seed===w.seed&&meta.source_world_index===w.index&&meta.geometry_sha256===w.geometry_sha256,'provenance drift');
   const old=owners.get(w.geometry_sha256);ok(!old||old===split,'cross split geometry leakage');owners.set(w.geometry_sha256,split);
   const hs=v48.settings.map(z=>v48.prep(w,...z)),start=hs.map(v48.observe),first=start.map(independentAction);
   for(const h of hs){validateProbeHistory(h);probeOps+=2;}
   ok(H(v48.withoutMission(start[0]))===H(v48.withoutMission(start[1])),'mission ablation failed');
   ok(H(v48.withoutProbe(start[0]))===H(v48.withoutProbe(start[5])),'physical outcome ablation failed');
   ok(H(v48.withoutProbe(start[0]))===H(v48.withoutProbe(start[7])),'obsolete control ablation failed');
   ok(first[0]!==first[1]&&first[0]!==first[2]&&first[0]===first[4]&&first[0]===first[5]&&first[0]===first[7],'causal initial actions invalid');
   ok(H(first)===H(meta.initial_actions)&&meta.probe_age_actions>=81&&meta.mission_age_actions>=260,'challenge metadata inconsistent');
   const paint=[];
   for(let arm=0;arm<8;arm++){
    const h=hs[arm],a=[];
    for(let turn=0;turn<meta.episode_steps[arm];turn++){
     const row=pol[p++];ok(row&&Object.keys(row).sort().join(',')==='action,input,weight','row shape incorrect');
     const input=v48.observe(h);ok(H(input)===H(row.input),'replayed observation mismatch');if(turn===0||turn===meta.episode_steps[arm]-1||row.action.startsWith('PAINT_')||turn%16===0)validateInput(row.input,h);
     const expected=independentAction(row.input);ok(expected===row.action,'independent planner or measured law mismatch');
     ok(row.input.legal.includes(row.action),'illegal label');
     const key=H(row.input),prior=labels.get(key);ok(!prior||prior.split===split,'cross split input');ok(!prior||prior.action===row.action,'contradictory label');
     if(prior&&split==='TRAIN'){ok(row.weight===0,'duplicate not zero weighted');zero++;}else{ok(row.weight===1,'unexpected weight');labels.set(key,{split,action:row.action});}
     if(row.action==='PAINT_A')paintA++;if(row.action==='PAINT_B')paintB++;if(row.action==='ERASE')erases++;
     if(row.action.startsWith('PAINT_'))a.push({input:row.input,command:row.action});
     replayStep(h,row.action);decisions++;
    }
    ok(h.done&&h.success&&a.length===2,'host replay failed');
    ok(a[0].command===meta.first_material_commands[arm]&&a[1].command===meta.second_material_commands[arm],'paint checkpoint drift');
    paint.push(a[0]);episodes++;
   }
   ok(H(v48.withoutProbe(paint[0].input))===H(v48.withoutProbe(paint[5].input))&&paint[0].command!==paint[5].command,'causal material result failed');
   ok(H(v48.withoutProbe(paint[0].input))===H(v48.withoutProbe(paint[7].input))&&paint[0].command===paint[7].command,'obsolete probe control failed');
   physicalPairs++;missionPairs++;gatePairs++;negativeMission++;negativeProbe++;
  }
  ok(p===pol.length&&zero===s.train_duplicate_zero_weight,'split row count/duplicate mismatch');
 }
 ok(decisions===mf.total_policy_decisions&&owners.size===mf.unique_world_geometries,'total mismatch');
 return {verified_policy_decisions:decisions,successful_synthetic_host_episodes:episodes,source_worlds:owners.size,
  physically_executed_calibration_paints:probeOps,physically_executed_calibration_erases:probeOps,
  causal_physical_observation_pairs:physicalPairs,causal_mission_pairs:missionPairs,causal_remote_gate_pairs:gatePairs,
  stale_mission_controls:negativeMission,obsolete_physical_probe_controls:negativeProbe,
  labelled_paint_A_commands:paintA,labelled_paint_B_commands:paintB,labelled_erase_commands:erases,
  cross_split_leakage:0,contradictory_labels:0};
}
if(require.main===module){const [dir,other,flag]=process.argv.slice(2);ok(dir,'usage node test_empirical_tool_v48.js DIR [SECOND_DIR] [--quick]');
 const quick=other==='--quick'||flag==='--quick';const r=audit(dir,{quick});
 if(other&&other!=='--quick'){const a=fs.readdirSync(dir).sort(),b=fs.readdirSync(other).sort();ok(H(a)===H(b),'nondeterministic filenames');for(const file of a)ok(H(fs.readFileSync(path.join(dir,file)))===H(fs.readFileSync(path.join(other,file))),'nondeterministic bytes '+file);}
 console.log(JSON.stringify(r,null,2));
}
module.exports={audit,independentAction,replayStep};
