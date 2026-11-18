'use strict';
/* Corpus Factory v47: old mission + remote gate + historically revised material-tool law.
 * Data only. Host is synthetic; all commands are accepted by v44 actual host apply(). */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v46=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error(m)};
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const cp=x=>JSON.parse(JSON.stringify(x));
const flip=x=>x==='NORMAL'?'SWAPPED':'NORMAL';
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
function age(h,n){const [next]=h.g.adj[h.pos].find(([,e])=>e!==h.gate)||[];ok(next!==undefined,'no safe arm');
 const a=h.g.coords[h.pos],b=h.g.coords[next];const go=b[0]>a[0]?'MOVE_E':b[0]<a[0]?'MOVE_W':b[1]>a[1]?'MOVE_S':'MOVE_N';
 for(let t=0;t<n;t++)v46.accepted(h,t%2?reverse[go]:go);
 ok(n%2===0,'age must restore start');}
function lawEvent(h,mode){ok(['NORMAL','SWAPPED'].includes(mode),'bad tool mode');h.tool_mode=mode;h.events.push({kind:'TOOL_LAW',at:h.accepted,accepted:true,mode});}
function prep(w,closed,missionSwapped,staleMode,toolMode,obsoleteMode,stageFlip){
 const h=v46.prepare(w,closed,missionSwapped,staleMode);h.stage_flip=stageFlip;
 lawEvent(h,obsoleteMode);age(h,8);
 h.events.push({kind:'TOOL_DIAGNOSTIC',at:h.accepted,accepted:true,code:(w.seed%991)+17});age(h,8);
 lawEvent(h,toolMode);age(h,80);
 ok(h.pos===w.s&&h.stage===0,'preparation changed mission state');return h;}
function observe(h){const x=v46.observe(h);x.v='synthetic-v47-historical-tool-law-and-mission';
 const events=h.events.filter(e=>e.kind==='TOOL_LAW'||e.kind==='TOOL_DIAGNOSTIC').map(e=>e.kind==='TOOL_LAW'?[e.at,e.kind,e.mode]:[e.at,e.kind,e.code]);
 x.durable=[...x.durable,...events].sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));
 return x;}
function withoutLaw(i){const x=cp(i);x.durable=x.durable.filter(z=>z[1]!=='TOOL_LAW'&&z[1]!=='TOOL_DIAGNOSTIC');return x;}
function withoutMission(i){const x=cp(i);x.durable=x.durable.filter(z=>z[1]!=='MISSION_ASSIGN');return x;}
function rule(i){const events=i.durable.filter(e=>e[1]==='TOOL_LAW');ok(events.length>=2,'insufficient law history');return events.at(-1)[2];}
function teacher(i){const physical=v46.teacher(i);if(!physical.startsWith('PAINT_'))return physical;
 return rule(i)==='SWAPPED'?'PAINT_'+(physical.endsWith('A')?'B':'A'):physical;}
function accepted(h,command){ok(host.legal(h).includes(command),'illegal presented action '+command);
 const physical=command.startsWith('PAINT_')&&h.tool_mode==='SWAPPED'?'PAINT_'+(command.endsWith('A')?'B':'A'):command;
 const old=h.stage,previous=h.accepted;v46.accepted(h,physical);ok(h.accepted===previous+1,'host not accepted');
 const ack=h.events.findLast(e=>e.kind==='ACTION_ACK'&&e.at===h.accepted);ok(ack,'missing host acknowledgement');ack.action=command;
 // Material-law transition follows a real successful first-stage PAINT operation.
 // This emits a history receipt; no current law bit enters the observation.
 if(old===0&&h.stage===1){const target=h.stage_flip?flip(h.tool_mode):h.tool_mode;lawEvent(h,target);}
 return physical;}
function rollout(h,limit=256){const rows=[],paints=[],checkpoints=[];
 for(let t=0;t<limit&&!h.done;t++){
  const input=observe(h),action=teacher(input);ok(input.legal.includes(action),'teacher proposed illegal command');
  if(action.startsWith('PAINT_')){paints.push(action);checkpoints.push({input,action,tool_mode:rule(input),stage:h.stage});}
  rows.push({input,action,weight:1});accepted(h,action);
 }ok(h.done&&h.success,'synthetic host did not reach successful terminal: '+JSON.stringify({split:h.w.split,index:h.w.index,stage:h.stage,painted:h.painted,pos:h.pos,goal:h.w.goal,goal2:h.w.goal2,tool_mode:h.tool_mode,steps:rows.length,last:rows.slice(-4).map(x=>x.action)}));
 ok(paints.length===2,'expected exactly two material paints');
 return {rows,paints,checkpoints};}
function scenario(w){const stageFlip=w.index%2===0;
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
 const initial=settings.map(([closed,missionSwapped,stale,mode,obsolete])=>prep(w,closed,missionSwapped,stale,mode,obsolete,stageFlip));
 const inputs=initial.map(observe),first=inputs.map(teacher);
 ok(H(withoutMission(inputs[0]))===H(withoutMission(inputs[1])),'mission receipt ablation leaked');
 ok(H(withoutLaw(inputs[0]))===H(withoutLaw(inputs[5])),'material-law receipt ablation leaked');
 ok(H(withoutLaw(inputs[0]))===H(withoutLaw(inputs[7])),'obsolete material-law receipt ablation leaked');
 ok(first[0]!==first[1]&&first[0]!==first[2]&&first[0]===first[4]&&first[0]===first[5]&&first[0]===first[7],'navigation causal controls failed');
 ok(inputs.every(x=>!('target' in x)&&!('stage' in x)&&!('tool_mode' in x)&&!('source_seed' in x)),'privileged input leak');
 const lawAge=initial[0].accepted-initial[0].events.filter(e=>e.kind==='TOOL_LAW').at(-1).at;
 const missionAge=initial[0].accepted-initial[0].events.filter(e=>e.kind==='MISSION_ASSIGN').at(-1).at;
 ok(lawAge>=80&&missionAge>=256,'historical receipts not aged');
 const traces=initial.map(h=>rollout(h));const a=traces[0].checkpoints[0],b=traces[5].checkpoints[0],c=traces[7].checkpoints[0];
 ok(H(withoutLaw(a.input))===H(withoutLaw(b.input)),'first PAINT snapshots leak tool law');
 ok(a.action!==b.action&&a.tool_mode!==b.tool_mode,'tool law does not change paint label');
 ok(H(withoutLaw(a.input))===H(withoutLaw(c.input))&&a.action===c.action,'obsolete receipt altered material command');
 for(let arm=0;arm<traces.length;arm++){
  const trace=traces[arm],initialMode=settings[arm][3];
  ok(trace.checkpoints[0].stage===0&&trace.checkpoints[1].stage===1,'paint checkpoint stage mismatch');
  ok(trace.checkpoints[0].tool_mode===initialMode,'initial historical law mismatch');
  ok(trace.checkpoints[1].tool_mode===(stageFlip?flip(initialMode):initialMode),'mid-episode law transition mismatch');
 }
 return {traces,meta:{synthetic_not_native:true,split:w.split,source_world_index:w.index,source_seed:w.seed,geometry_sha256:w.geometry_sha256,
  derived_from:'v46 same world split; never repartition source worlds',stage_law_flip:stageFlip,
  arms:['mission0_gate_open_law_normal','mission1_gate_open_law_normal','mission0_gate_closed_law_normal',
   'mission1_gate_closed_law_normal','obsolete_mission_control','mission0_gate_open_law_swapped',
   'mission1_gate_open_law_swapped','obsolete_law_control'],
  first_actions:first,material_first_commands:traces.map(z=>z.paints[0]),material_second_commands:traces.map(z=>z.paints[1]),
  episode_steps:traces.map(z=>z.rows.length),mission_positive_pair:[0,1],gate_positive_pair:[0,2],
  law_positive_pair:[0,5],stale_mission_control:[0,4],stale_law_control:[0,7],
  historical_law_age_actions:lawAge,historical_mission_age_actions:missionAge,
  law_ablated_paint_checkpoint_sha256:H(withoutLaw(a.input)),
  law_first_actions_causal: [a.action,b.action],initial_world_coverage_fraction:+(new Set(initial[0].positions).size/initial[0].g.coords.length).toFixed(6)}};
}
function save(out,name,rows){const data=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),data,{flag:'wx'});return {rows:rows.length,bytes:data.length,sha256:H(data)}}
function generate(out,{quick=false}={}){ok(!fs.existsSync(out),'new output directory required');fs.mkdirSync(out,{recursive:true});
 const shards={},splits={},owners=new Map(),inputs=new Map();let total=0;
 for(const [split,count] of Object.entries(SPLITS)){
  const countActual=quick?Math.min(2,count):count,policy=[],challenges=[];let zero=0,terminal=0;
  for(let index=0;index<countActual;index++){
   const w=v46.candidate(split,index).w,world=scenario(w),geometry=owners.get(w.geometry_sha256);
   ok(!geometry||geometry===split,'cross-split geometry collision');owners.set(w.geometry_sha256,split);
   challenges.push(world.meta);
   for(const trace of world.traces){terminal++;for(const row of trace.rows){
    const hash=H(row.input),prior=inputs.get(hash);ok(!prior||prior.split===split,'cross-split input leakage');ok(!prior||prior.action===row.action,'contradictory label');
    if(prior&&split==='TRAIN'){row.weight=0;zero++;}else inputs.set(hash,{split,action:row.action});policy.push(row);
   }}
  }
  shards[split+'_policy.jsonl']=save(out,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=save(out,split+'_challenge_EVALUATOR_ONLY.jsonl',challenges);
  splits[split]={worlds:countActual,host_successful_episodes:terminal,decisions:policy.length,
   mission_positive_pairs:countActual,gate_positive_pairs:countActual,tool_law_positive_pairs:countActual,
   stale_mission_controls:countActual,stale_law_controls:countActual,duplicate_zero_weight:zero,
   law_flip_worlds:challenges.filter(c=>c.stage_law_flip).length};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/historical-mission-tool-law-v47/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'lawful_mission_v47.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  model_contract:'ONLY the input object from *_policy.jsonl may be fed to the model. action/weight are supervised labels; all *_challenge_EVALUATOR_ONLY.jsonl must remain sealed.',
  split_contract:'Same v46 source worlds and assignments: each world remains in its original TRAIN/DEV/HELDOUT split. Deduplicate and group across v46+v47 by source_world_index+source_seed.',
  task:'Partial survey, old superseded mission, remote obstructed junction, host-executed wrong-paint recovery, old tool-law permutation receipts and mid-episode material mapping transition. Eight matched trajectories per world.',
  splits,shards,total_policy_decisions:total,unique_world_geometries:owners.size,
  known_limitations:['SYNTHETIC graph host/adapter, not native CyberSand ABI','Initial survey and history ageing scripted but host-accepted',
   'Tool-law events are symbolic and explicit, so a parser can solve without an SLM','Post-first-paint law transition disclosed in durable history; synthetic host stage completion remains narrow',
   'No SLM training/inference, no neural latency/RSS, and no immutable Actions artifact publication']};
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;}
if(require.main===module){const [out,quick]=process.argv.slice(2);ok(out,'usage node lawful_mission_v47.js NEW_DATA_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(out),{quick:quick==='--quick'}).splits,null,2));}
module.exports={SPLITS,prep,observe,withoutLaw,withoutMission,rule,teacher,accepted,rollout,scenario,generate,H};
