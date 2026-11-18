'use strict';
/* DRIV3R corpus_factory v46: host-accepted navigation under stale/superseded
 * historical mission receipts and partial-world exploration. SYNTHETIC ONLY. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v45=require('./frontier_memory_v45.js'),host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(p,m)=>{if(!p)throw Error(m)};
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const directions=['N','E','S','W'];
const reverse={MOVE_N:'MOVE_S',MOVE_E:'MOVE_W',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E'};
const coord=(h,n)=>h.g.coords[n];
const k=x=>x.join(',');
function auditTarget(i){const msgs=i.durable.filter(x=>x[1]==='MISSION_ASSIGN'||x[1]==='GOAL_REVISION');ok(msgs.length,'no live mission receipt');const msg=msgs.at(-1);return [msg[2],msg[3]];}
function accepted(h,a){const before=h.stage;ok(host.apply(h,a).accepted,'host refused '+a);
 if(before!==2&&h.stage===2)h.events.push({kind:'MISSION_COMPLETE',accepted:true,at:h.accepted});}
function safeAge(h,steps){const neighbor=h.g.adj[h.pos].find(([,e])=>e!==h.gate);
 ok(neighbor,'safe age arm absent');const a=h.g.coords[h.pos],b=h.g.coords[neighbor[0]];
 const d=directions.find(z=>({N:b[1]===a[1]-1,E:b[0]===a[0]+1,S:b[1]===a[1]+1,W:b[0]===a[0]-1})[z]);
 ok(d,'no cardinal age arm');const out='MOVE_'+d,back=reverse[out];for(let n=0;n<steps;n++)accepted(h,n%2?back:out);}
function receipt(h,goal,mat){h.events.push({kind:'MISSION_ASSIGN',at:h.accepted,accepted:true,goal,material:mat});}
function prepare(w,closed,swapped,staleMode='opposite'){
 const h=v45.prelude(w,'positive',closed);ok(h.stage===0,'bad starting stage');
 // A competing instruction is given before an explicit superseding instruction;
 // only the latest assignment is live. This is a synthetic host external event.
 const final=swapped?w.goal2:w.goal,other=swapped?w.goal:w.goal2;
 const stale=staleMode==='same'?final:other;
 receipt(h,stale,'A');safeAge(h,24);receipt(h,final,'A');
 h.w={...h.w,goal:final,goal2:other};
 safeAge(h,176);ok(h.pos===w.s&&h.stage===0,'initial positioning mismatch');
 return h;
}
function observe(h){const x=v45.observe(h);delete x.stage;delete x.target;
 x.v='synthetic-v46-historical-missions';
 const extra=h.events.filter(e=>e.kind==='MISSION_ASSIGN'||e.kind==='MISSION_COMPLETE').map(e=>e.kind==='MISSION_COMPLETE'?[e.at,e.kind]:[e.at,e.kind,coord(h,e.goal),e.material]);
 x.durable=[...x.durable,...extra].sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));
 return x;}
function ablateMission(i){const x=JSON.parse(JSON.stringify(i));x.durable=x.durable.filter(e=>e[1]!=='MISSION_ASSIGN');return x;}
function teacher(i){if(i.durable.some(x=>x[1]==='MISSION_COMPLETE'))return 'STOP';
 const [goal,mat]=auditTarget(i);if(k(i.p)===k(goal)){
  return i.painted.some(([p])=>k(p)===k(goal))?'ERASE':'PAINT_'+mat;
 }
 const a=v45.pathOnSurvey(i,goal);ok(a&&i.legal.includes(a),'no legal surveyed route to frontier');return a;
}
function replay(h,limit=256){const records=[];for(let step=0;step<limit&&!h.done;step++){
 const i=observe(h),a=teacher(i);ok(i.legal.includes(a),'illegal trainer action '+a);
 records.push({input:i,action:a,weight:1});accepted(h,a);}
 ok(h.done&&h.success,'mission recovery failed: '+h.w.split+':'+h.w.index);return records;}
function fullRank(h,target,blocked){const ranks=host.ranks(h.g,h.pos,target,blocked);ok(ranks.length&&Number.isFinite(ranks[0].cost),'no full host route');return ranks;}
function pair(w){const init=[prepare(w,false,false),prepare(w,false,true),prepare(w,true,false),prepare(w,true,true),prepare(w,false,false,'same')];
 const ins=init.map(observe),first=ins.map(teacher);
 // Mission-positive controls: same legal menu, physical map, recent history and
 // explored edges, different delayed final assignment. Remove assignments and
 // both branches are information-theoretically identical at the first decision.
 ok(H(ablateMission(ins[0]))===H(ablateMission(ins[1])),'mission counterfactual leaked');
 ok(first[0]!==first[1],'mission assignment did not cause action change');
 const firstRank=fullRank(init[0],init[0].w.goal,-1),swapRank=fullRank(init[1],init[1].w.goal,-1);
 ok(first[0]===firstRank[0].action&&first[1]===swapRank[0].action,'mission fork not globally optimal first action');
 const mr1=firstRank.find(z=>z.action===first[1]),mr2=swapRank.find(z=>z.action===first[0]);
 ok(mr1&&mr2&&mr1.cost-firstRank[0].cost>=1&&mr2.cost-swapRank[0].cost>=1,'no strict mission regret');
 ok(first[0]!==first[2],'positive remote gate contrast lost');
 ok(H(ins[0].recent)===H(ins[2].recent)&&H(ins[0].local)===H(ins[2].local),'remote gate current/recent leak');
 ok(first[0]===first[4],'stale superseded receipt changed control decision');
 // Negative stale-message control differs only in an obsolete historical receipt.
 const negative=ablateMission(ins[0]);ok(H(negative)===H(ablateMission(ins[4])),'negative receipt leak beyond mission history');
 const minAge=init[0].accepted - init[0].events.filter(e=>e.kind==='MISSION_ASSIGN').at(-1).at;
 ok(minAge>=176,'final mission message not old enough');
 const initialCoverage=+(new Set(init[0].positions).size/init[0].g.coords.length).toFixed(6);
 const traces=init.map(h=>replay(h));
 return {traces,meta:{synthetic_not_native:true,split:w.split,source_world_index:w.index,source_seed:w.seed,
  geometry_sha256:w.geometry_sha256,first_actions:first,mission_positive_pair:[0,1],gate_positive_pair:[0,2],
  superseded_negative_pair:[0,4],mission_ablation_identical:true,terminal_steps:traces.map(z=>z.length),
  last_mission_age_actions:minAge,first_action_regret_mission:[mr1.cost-firstRank[0].cost,mr2.cost-swapRank[0].cost],
  initial_coverage_fraction:initialCoverage,
  historical_ordinal_mission_count:2}};
}
function candidate(split,index){for(let probe=0;probe<64;probe++){
 const w=v45.candidate(split,800+index*31+probe).w;
 try{const value=pair(w);if(value.meta.initial_coverage_fraction>.82)continue;return {w,...value,probe};}
 catch(e){if(e.message.includes('no qualified topology'))throw e;}
 }throw Error('no strict mission and gate counterfactual '+split+':'+index);}
function write(dir,name,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,name),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)}}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},stats={},ownership=new Map(),inputs=new Map();let total=0;
 for(const [split,num] of Object.entries(SPLITS)){
  const n=quick?Math.min(2,num):num,policy=[],challenge=[];let duplicate=0,finished=0;
  for(let index=0;index<n;index++){
   const {w,traces,meta}=candidate(split,index),owner=ownership.get(w.geometry_sha256);
   ok(!owner||owner===split,'cross-split geometry collision');ownership.set(w.geometry_sha256,split);challenge.push(meta);
   for(const trace of traces){finished++;for(const row of trace){const inputHash=H(row.input),p=inputs.get(inputHash);
    ok(!p||p.split===split,'cross-split input collision');ok(!p||p.action===row.action,'conflicting labels');
    if(p&&split==='TRAIN'){row.weight=0;duplicate++}else inputs.set(inputHash,{split,action:row.action});
    policy.push(row);}}
  }
  shards[split+'_policy.jsonl']=write(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=write(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  stats[split]={worlds:n,terminal_episodes:finished,decisions:policy.length,positive_mission_pairs:n,
   positive_gate_pairs:n,negative_stale_controls:n,duplicate_zero_weight:duplicate};total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/historical-mission-memory-v46/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'mission_memory_v46.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:[{file:'frontier_memory_v45.js',sha256:H(fs.readFileSync(require.resolve('./frontier_memory_v45.js')))},
   {file:'unseen_topology_v44.js',sha256:H(fs.readFileSync(require.resolve('./unseen_topology_v44.js')))}],
  input_contract:'Only policy.input is exposed. policy.action and weight are supervised labels; all *_challenge_EVALUATOR_ONLY.jsonl are sealed evaluation metadata.',
  focus:'Historical superseded-goal receipts without current target/stage; 176 accepted action age, partial map survey, remote obstruction and counterfactual mission vs stale-history control.',
  splits:stats,shards,total_policy_decisions:total,unique_world_geometries:ownership.size,
  caveats:['SYNTHETIC graph host, not native CyberSand','Survey and historical ageing are host-accepted but scripted','Mission external events synthetic, not native gameplay signals','Teacher uses observed survey, not hidden graph, except challenge qualification','No target SLM weights updated or inference measured','Actions artifact retention not implemented']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;}
module.exports={SPLITS,prepare,observe,ablateMission,teacher,replay,pair,candidate,generate,H,accepted,auditTarget};
if(require.main===module){const [dir,mode]=process.argv.slice(2);ok(dir,'usage: node mission_memory_v46.js NEW_OUTPUT [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:mode==='--quick'}).splits,null,2));}
