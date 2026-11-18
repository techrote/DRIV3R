'use strict';
/* DRIV3R Corpus Factory v48: empirical paint-outcome calibration, historic goals,
 * partial surveyed topology and remote gate. Actual synthetic host applies each
 * calibration paint/erase. DATA ONLY, NOT native CyberSand or policy training. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v46=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(b,m)=>{if(!b)throw Error(m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const settings=[
 [false,false,'opposite','NORMAL','NORMAL'],
 [false,true,'opposite','NORMAL','NORMAL'],
 [true,false,'opposite','NORMAL','NORMAL'],
 [true,true,'opposite','NORMAL','NORMAL'],
 [false,false,'same','NORMAL','NORMAL'],
 [false,false,'opposite','SWAPPED','NORMAL'],
 [false,true,'opposite','SWAPPED','NORMAL'],
 [false,false,'opposite','NORMAL','SWAPPED']
];
function accepted(h,command){ok(host.legal(h).includes(command),'host legal menu rejects command '+command);
 const physical=command.startsWith('PAINT_')&&h.tool_mode==='SWAPPED'?'PAINT_'+(command.endsWith('A')?'B':'A'):command;
 const before=h.accepted;v46.accepted(h,physical);ok(h.accepted===before+1,'host did not accept');
 const ack=h.events.findLast(e=>e.kind==='ACTION_ACK');ok(ack&&ack.at===h.accepted,'missing ack');ack.action=command;
 return {physical,material:command.startsWith('PAINT_')?h.painted[h.pos]:undefined};
}
function calibration(h){ok(h.painted[h.pos]===undefined,'calibration cell occupied');
 const r=accepted(h,'PAINT_A');ok(r.material==='A'||r.material==='B','missing physical probe outcome');
 h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:'PAINT_A',material:r.material});
 accepted(h,'ERASE');ok(h.painted[h.pos]===undefined,'calibration did not clean probe cell');return r.material;}
function age(h,n){ok(n%2===0,'age requires even movement count');
 const neighbor=h.g.adj[h.pos].find(([,edge])=>edge!==h.gate);ok(neighbor,'missing safe arm');
 const a=h.g.coords[h.pos],b=h.g.coords[neighbor[0]];
 const move=b[0]>a[0]?'MOVE_E':b[0]<a[0]?'MOVE_W':b[1]>a[1]?'MOVE_S':'MOVE_N';
 for(let i=0;i<n;i++)accepted(h,i%2?reverse[move]:move);
}
function prep(w,closed,missionSwapped,staleMission,mode,obsolete){
 const h=v46.prepare(w,closed,missionSwapped,staleMission);h.tool_mode=obsolete;
 const first=calibration(h);age(h,8);
 h.events.push({kind:'TOOL_DIAGNOSTIC',accepted:true,at:h.accepted,code:(w.seed%991)+17});age(h,8);
 h.tool_mode=mode;const second=calibration(h);age(h,80);
 ok(h.pos===w.s&&h.stage===0&&first===(obsolete==='NORMAL'?'A':'B')&&second===(mode==='NORMAL'?'A':'B'),'preflight state/physical result mismatch');
 return h;
}
function observe(h){const i=v46.observe(h);i.v='synthetic-v48-empirical-paint-calibration';
 const receipts=h.events.filter(e=>e.kind==='PAINT_RESULT'||e.kind==='TOOL_DIAGNOSTIC').map(e=>e.kind==='PAINT_RESULT'?[e.at,e.kind,e.issued,e.material]:[e.at,e.kind,e.code]);
 i.durable=[...i.durable,...receipts].sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));
 return i;
}
function withoutProbe(i){const x=cp(i);x.durable=x.durable.filter(z=>z[1]!=='PAINT_RESULT');return x;}
function withoutMission(i){const x=cp(i);x.durable=x.durable.filter(z=>z[1]!=='MISSION_ASSIGN');return x;}
function measuredLaw(i){const receipts=i.durable.filter(z=>z[1]==='PAINT_RESULT'&&z[2]==='PAINT_A');
 ok(receipts.length===2,'requires precisely two accepted historical calibrations');
 const latest=receipts.at(-1);ok(latest[3]==='A'||latest[3]==='B','invalid physical paint result');
 return latest[3]==='A'?'NORMAL':'SWAPPED';}
function teacher(i){const physical=v46.teacher(i);if(!physical.startsWith('PAINT_'))return physical;
 return measuredLaw(i)==='SWAPPED'?'PAINT_'+(physical.endsWith('A')?'B':'A'):physical;
}
function rollout(h){const rows=[],paint=[];for(let t=0;t<256&&!h.done;t++){
  const input=observe(h),action=teacher(input);ok(input.legal.includes(action),'illegal teacher choice');
  if(action.startsWith('PAINT_'))paint.push({input,action});
  rows.push({input,action,weight:1});accepted(h,action);
 }
 ok(h.done&&h.success&&paint.length===2,'rollout terminal failure');return {rows,paint};
}
function scenario(w){const hs=settings.map(s=>prep(w,...s)),inputs=hs.map(observe),first=inputs.map(teacher);
 ok(H(withoutMission(inputs[0]))===H(withoutMission(inputs[1])),'mission ablation leaks');
 ok(H(withoutProbe(inputs[0]))===H(withoutProbe(inputs[5])),'physical positive-pair ablation leaks');
 ok(H(withoutProbe(inputs[0]))===H(withoutProbe(inputs[7])),'physical stale-control ablation leaks');
 ok(first[0]!==first[1]&&first[0]!==first[2]&&first[0]===first[4]&&first[0]===first[5]&&first[0]===first[7],'navigation causal contract failed');
 const ageProbe=hs[0].accepted-hs[0].events.filter(e=>e.kind==='PAINT_RESULT').at(-1).at;
 const ageMission=hs[0].accepted-hs[0].events.filter(e=>e.kind==='MISSION_ASSIGN').at(-1).at;
 ok(ageProbe>=81&&ageMission>=260,'probe/mission memory not old');
 const traces=hs.map(rollout);
 const a=traces[0].paint[0],b=traces[5].paint[0],c=traces[7].paint[0];
 ok(H(withoutProbe(a.input))===H(withoutProbe(b.input))&&a.action!==b.action,'physical paint outcome not causal');
 ok(H(withoutProbe(a.input))===H(withoutProbe(c.input))&&a.action===c.action,'obsolete physical paint affected decision');
 const physical=(arm,paint)=>settings[arm][3]==='SWAPPED'?(paint.endsWith('A')?'B':'A'):paint.slice(-1);
 for(let k=0;k<8;k++){
  ok(physical(k,traces[k].paint[0].action)==='A'&&physical(k,traces[k].paint[1].action)==='B','physical mission material invalid');
  ok(traces[k].rows.length>=2,'no meaningful recovery');
 }
 const coverage=+(new Set(hs[0].positions).size/hs[0].g.coords.length).toFixed(6);
 return {traces,meta:{synthetic_not_native:true,split:w.split,source_world_index:w.index,source_seed:w.seed,
 geometry_sha256:w.geometry_sha256,world_family:'v46 split-preserved; CALIBRATION replaces v47 symbolic law receipt',
 calibration:'host-accepted PAINT_A => physical material, ERASE; two separated calibration episodes',
 arms:['mission0_gate_open_normal','mission1_gate_open_normal','mission0_gate_closed_normal','mission1_gate_closed_normal',
 'stale_mission_control','mission0_gate_open_swapped','mission1_gate_open_swapped','obsolete_probe_control'],
 mission_positive_pair:[0,1],gate_positive_pair:[0,2],physical_probe_positive_pair:[0,5],
 stale_mission_negative_pair:[0,4],obsolete_probe_negative_pair:[0,7],
 initial_actions:first,first_material_commands:traces.map(z=>z.paint[0].action),second_material_commands:traces.map(z=>z.paint[1].action),
 episode_steps:traces.map(z=>z.rows.length),probe_age_actions:ageProbe,mission_age_actions:ageMission,
 initial_coverage_fraction:coverage,physical_ablation_sha256:H(withoutProbe(a.input))}};
}
function save(dir,name,records){const b=Buffer.from(records.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,name),b,{flag:'wx'});return {rows:records.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output already exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},owners=new Map(),inputOwners=new Map();let total=0;
 for(const [split,num] of Object.entries(SPLITS)){
  const count=quick?Math.min(2,num):num,policy=[],challenge=[];let terminal=0,duplicates=0;
  for(let idx=0;idx<count;idx++){
   const w=v46.candidate(split,idx).w,world=scenario(w),old=owners.get(w.geometry_sha256);
   ok(!old||old===split,'cross split geometry leak');owners.set(w.geometry_sha256,split);challenge.push(world.meta);
   for(const trace of world.traces){terminal++;for(const row of trace.rows){
    const hash=H(row.input),oldInput=inputOwners.get(hash);
    ok(!oldInput||oldInput.split===split,'cross split input leakage');ok(!oldInput||oldInput.action===row.action,'contradictory labels');
    if(oldInput&&split==='TRAIN'){row.weight=0;duplicates++;}else inputOwners.set(hash,{split,action:row.action});policy.push(row);
   }}
  }
  shards[split+'_policy.jsonl']=save(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=save(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,host_successful_episodes:terminal,policy_decisions:policy.length,
   causal_mission_pairs:count,causal_remote_gate_pairs:count,causal_physical_probes:count,
   stale_mission_controls:count,obsolete_probe_controls:count,train_duplicate_zero_weight:duplicates};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/empirical-tool-calibration-v48/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'empirical_tool_v48.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contracts:{policy:'ONLY policy.input may be given to SLM; policy.action/weight are supervised targets; NEVER train on evaluator-only shards',
   split:'v46 source worlds preserve original TRAIN/DEV/HELDOUT partitions; across v46/v47/v48 group/deduplicate by split+source_world_index+source_seed',
   probe:'PAINT_A/ERASE are host-accepted operations in PRELUDE, with the physically observed material written to historical PAINT_RESULT; no oracle TOOL_LAW appears in inputs'},
  scope:'Partial survey; stale historical missions; remote gate; physical paint-outcome calibration and obsolete calibration controls; wrong paint recovery.',
  splits,shards,total_policy_decisions:total,unique_world_geometries:owners.size,
  limitations:['Synthetic graph/material mapping adapter, NOT native CyberSand','Preludes including actual paint/erase probes are scripted host-accepted actions but excluded from supervised policy records',
   'Tool law fixed within each rollout; v47 separately covers disclosed mid-mission flips','Explicit physical PAINT_RESULT can still be solved by a simple symbolic last-observation parser',
   'No SLM optimizer step, no native gameplay episodes, no neural latency/RSS, no immutable Actions shard artifact']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage node empirical_tool_v48.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,settings,accepted,calibration,age,prep,observe,withoutProbe,withoutMission,measuredLaw,teacher,rollout,scenario,generate,H};
