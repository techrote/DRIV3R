'use strict';
/* DRIV3R Corpus Factory v49 — physical dual-surface calibration and associative
 * memory. DATA ONLY. All actions executed by a synthetic graph/material host;
 * not native CyberSand, no policy training, no GPU/API expenditures. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v46=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js'),v48=require('./empirical_tool_v48.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error(m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:48,DEV:16,HELDOUT:16};
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const arms=[
 {id:'00_baseline',A:'NORMAL',B:'NORMAL',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:false,reverse:false},
 {id:'01_causal_A',A:'SWAPPED',B:'NORMAL',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:false,reverse:false},
 {id:'02_causal_B',A:'NORMAL',B:'SWAPPED',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:false,reverse:false},
 {id:'03_joint_AB',A:'SWAPPED',B:'SWAPPED',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:false,reverse:false},
 {id:'04_gate_closed',A:'NORMAL',B:'NORMAL',oldA:'NORMAL',oldB:'NORMAL',closed:true,mission:false,reverse:false},
 {id:'05_mission_retarget',A:'NORMAL',B:'NORMAL',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:true,reverse:false},
 {id:'06_stale_A',A:'NORMAL',B:'NORMAL',oldA:'SWAPPED',oldB:'NORMAL',closed:false,mission:false,reverse:false},
 {id:'07_stale_B',A:'NORMAL',B:'NORMAL',oldA:'NORMAL',oldB:'SWAPPED',closed:false,mission:false,reverse:false},
 {id:'08_reversed_fresh_probe_order',A:'NORMAL',B:'NORMAL',oldA:'NORMAL',oldB:'NORMAL',closed:false,mission:false,reverse:true}
];
function site(w){const g=host.hydrate(w),n=g.adj[w.s].find(([i,e])=>e!==w.pivotal&&i!==w.goal&&i!==w.goal2);ok(n,'no safe B probe site');return n[0];}
function label(h){return h.surfaces[h.pos]||'A';}
function configure(h,w,arm){const b=site(w);
 h.surfaces={[w.s]:'A',[w.goal]:'A',[w.goal2]:'B',[b]:'B'};
 h.siteB=b;h.laws={A:arm.A,B:arm.B};h.tool_mode='NORMAL';return h;}
function command(h,a){if(a.startsWith('PAINT_'))h.tool_mode=h.laws[label(h)];return v48.accepted(h,a);}
function travel(h,to){const a=h.g.coords[h.pos],b=h.g.coords[to],dx=b[0]-a[0],dy=b[1]-a[1];
 ok(Math.abs(dx)+Math.abs(dy)===1,'probe site must be adjacent');const mv=dx===1?'MOVE_E':dx===-1?'MOVE_W':dy===1?'MOVE_S':'MOVE_N';command(h,mv);return mv;}
function probe(h,surface){const target=surface==='A'?h.w.s:h.siteB;
 if(h.pos!==target)travel(h,target);
 ok(label(h)===surface&&h.painted[h.pos]===undefined,'probe must land on empty labelled surface');
 const r=command(h,'PAINT_A');ok(r.material==='A'||r.material==='B','no observed outcome');
 h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:'PAINT_A',material:r.material,surface,site:h.g.coords[h.pos]});
 command(h,'ERASE');ok(h.painted[h.pos]===undefined,'calibration residue');return r.material;
}
function age(h,count){ok(count%2===0,'safe age must be even');const nb=h.g.adj[h.w.s].find(([v,e])=>v!==h.siteB&&e!==h.gate);
 // v46's safeAge semantics use currently reachable host edges and work at start.
 ok(h.pos===h.w.s,'age requires start');v48.age(h,count);ok(h.pos===h.w.s,'age failed to return');}
function prepare(w,arm){const h=configure(v46.prepare(w,arm.closed,arm.mission),w,arm),initial=cp(h.laws);
 h.laws={A:arm.oldA,B:arm.oldB};probe(h,'A');age(h,8);probe(h,'B');travel(h,w.s);age(h,8);
 h.laws=initial;
 const order=arm.reverse?['B','A']:['A','B'];for(const surf of order){probe(h,surf);if(h.pos!==w.s)travel(h,w.s);age(h,8);}
 age(h,80);ok(h.pos===w.s&&h.stage===0,'prelude moved or advanced mission');
 const receipts=h.events.filter(e=>e.kind==='PAINT_RESULT');ok(receipts.length===4,'missing four host probes');
 ok(receipts.every(e=>e.site.join(',')===h.g.coords[e.surface==='A'?w.s:h.siteB].join(',')),'wrong sensor position');
 return h;}
function observe(h){const x=v46.observe(h);x.v='synthetic-v49-dual-surface-physical-probes';x.surface=label(h);
 x.durable=[...x.durable,...h.events.filter(e=>e.kind==='PAINT_RESULT').map(e=>[e.at,'PAINT_RESULT',e.issued,e.material,e.surface,e.site])].sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));
 return x;}
function lawFor(i){const receipts=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[2]==='PAINT_A'&&e[4]===i.surface);
 ok(receipts.length===2,'two physical calibration receipts must exist for surface '+i.surface);
 return receipts.at(-1)[3]==='A'?'NORMAL':'SWAPPED';}
function teacher(i){const physical=v46.teacher(i);if(!physical.startsWith('PAINT_'))return physical;
 return lawFor(i)==='NORMAL'?physical:'PAINT_'+(physical.endsWith('A')?'B':'A');}
function mask(i,surface,ordinal='latest'){const x=cp(i),q=x.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===surface);
 const at=(ordinal==='latest'?q.at(-1):q[0]);ok(at,'missing selected probe');
 x.durable=x.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[0]===at[0]&&e[4]===surface));return x;}
function ablateAllProbes(i){return {...i,durable:i.durable.filter(e=>e[1]!=='PAINT_RESULT')};}
function ablateMission(i){const x=cp(i);x.durable=x.durable.filter(e=>e[1]!=='MISSION_ASSIGN');return x;}
function ablateGate(i){const x=cp(i);x.durable=x.durable.filter(e=>e[1]!=='EDGE_GATE');return x;}
function rollout(h){const rows=[],paints=[];
 for(let t=0;t<256&&!h.done;t++){
  const input=observe(h),action=teacher(input);ok(input.legal.includes(action),'illegal oracle '+action);
  rows.push({input,action,weight:1});if(action.startsWith('PAINT_'))paints.push({input,action});command(h,action);
 }
 ok(h.done&&h.success&&paints.length===2,'synthetic goal completion failed');return {rows,paints};}
function world(w){const hs=arms.map(a=>prepare(w,a)),ins=hs.map(observe),first=ins.map(teacher);
 // At initial fork only historical gate/mission records matter. A/B probe outcomes
 // differ but must NOT affect navigation. Obsolete receipts and order are controls.
 ok(first[0]===first[1]&&first[0]===first[2]&&first[0]===first[3]&&first[0]!==first[4]&&first[0]!==first[5], 'navigation controls failed');
 ok(first[0]===first[6]&&first[0]===first[7]&&first[0]===first[8],'probe nuisance changed navigation');
 ok(H(ablateMission(ins[0]))===H(ablateMission(ins[5])),'mission ablation mismatched');
 ok(H(ablateGate(ins[0]))===H(ablateGate(ins[4])),'gate ablation mismatched');
 const traces=hs.map(rollout),p=traces.map(x=>x.paints);
 ok(p.every((x,k)=>x.length===2&&(k===5?x[0].input.surface==='B'&&x[1].input.surface==='A':x[0].input.surface==='A'&&x[1].input.surface==='B')), 'goals not assigned to distinct surfaces');
 ok(p[0][0].action!==p[1][0].action&&p[0][0].action===p[2][0].action&&p[1][0].action===p[3][0].action,'surface A causality failed');
 ok(p[0][1].action!==p[2][1].action&&p[0][1].action===p[1][1].action&&p[2][1].action===p[3][1].action,'surface B causality failed');
 ok(H(mask(p[0][0].input,'A'))===H(mask(p[1][0].input,'A')),'A positive pair leaks without A last probe');
 ok(H(mask(p[0][1].input,'B'))===H(mask(p[2][1].input,'B')),'B positive pair leaks without B last probe');
 ok(H(mask(p[0][0].input,'A','oldest'))===H(mask(p[6][0].input,'A','oldest')),'stale A control leakage');
 ok(H(mask(p[0][1].input,'B','oldest'))===H(mask(p[7][1].input,'B','oldest')),'stale B control leakage');
 ok(p[0][0].action===p[6][0].action&&p[0][1].action===p[7][1].action,'obsolete result changed policy');
 ok(p[0][0].action===p[8][0].action&&p[0][1].action===p[8][1].action,'reversed receipt order changed policy');
 const ageProbes=hs[0].accepted-hs[0].events.filter(e=>e.kind==='PAINT_RESULT').at(-1).at;
 ok(ageProbes>=81,'calibration history is recent');
 const metas={synthetic_not_native:true,source_split:w.split,source_world_index:w.index,source_seed:w.seed,
 geometry_sha256:w.geometry_sha256,split:w.split,arms:arms.map(a=>a.id),first_actions:first,
 first_paint_commands:p.map(x=>x[0].action),second_paint_commands:p.map(x=>x[1].action),
 episode_steps:traces.map(t=>t.rows.length),probe_age_actions:ageProbes,
 causal_first_surface_pair:[0,1],causal_second_surface_pair:[0,2],
 joint_interaction:[0,3],gate_positive_pair:[0,4],mission_positive_pair:[0,5],
 stale_A_negative:[0,6],stale_B_negative:[0,7],reverse_order_negative:[0,8],
 old_A_probe_at_site:'start',old_B_probe_at_site:'adjacent safe cell',
 trace_synthetic_terminal_success:traces.every(t=>t.rows.at(-1).action==='STOP')};
 return {traces,meta:metas};}
function save(dir,f,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,f),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){ok(!fs.existsSync(dir),'output dir exists');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},geometry=new Map(),inputs=new Map(),nearNavigation=new Map();let total=0;
 for(const [split,n] of Object.entries(SPLITS)){
  const count=quick?Math.min(2,n):n,policy=[],challenge=[];let zero=0,nearZero=0,terminal=0;
  for(let ix=0;ix<count;ix++){
   const w=v46.candidate(split,ix).w,old=geometry.get(w.geometry_sha256);ok(!old||old===split,'geometry cross split');
   geometry.set(w.geometry_sha256,split);const x=world(w);challenge.push(x.meta);
   for(const trace of x.traces){terminal++;for(const row of trace.rows){const key=H(row.input),prev=inputs.get(key);
    ok(!prev||prev.split===split,'cross split input leak');ok(!prev||prev.action===row.action,'conflicting label');
    if(prev&&split==='TRAIN'){row.weight=0;zero++;}else inputs.set(key,{split,action:row.action});
    if(split==='TRAIN'&&row.action.startsWith('MOVE_')){const sig=H(ablateAllProbes(row.input)),other=nearNavigation.get(sig);
     ok(!other||other===row.action,'near-identical navigation has conflicting labels');
     if(other){if(row.weight!==0)nearZero++;row.weight=0;}else nearNavigation.set(sig,row.action);}
    policy.push(row);}}
  }
  shards[split+'_policy.jsonl']=save(dir,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=save(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,successful_synthetic_episodes:terminal,policy_decisions:policy.length,
   causal_surface_A_pairs:count,causal_surface_B_pairs:count,stale_A_controls:count,
   stale_B_controls:count,probe_order_controls:count,gate_pairs:count,mission_pairs:count,duplicate_train_zero_weight:zero,train_near_duplicate_navigation_zero_weight:nearZero};
  total+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/contextual-probe-v49/manifest',date:'2026-11-18',quick,
  synthetic_not_native_CyberSand:true,generator:'contextual_probe_v49.js',generator_sha256:H(fs.readFileSync(__filename)),
  parents:['empirical_tool_v48.js','mission_memory_v46.js','frontier_memory_v45.js','unseen_topology_v44.js'].map(file=>({file,sha256:H(fs.readFileSync(path.join(__dirname,file)))})),
  contract:{near_duplicate:'TRAIN movement records are zero-weighted if a previous movement has identical input after stripping all physically irrelevant calibration receipts; all trajectories retained for honest replay. Positive/negative paint contrasts remain full-weight.',input:'Only policy.input can be passed to model. policy.action and policy.weight are supervised targets; no evaluator metadata in training.',
   group:'Group by source_split+source_world_index+source_seed across v45-v49 to avoid training on the same geometry and mission in a consumer HELDOUT.',
   calibration:'Every context A/B probe is an accepted PAINT_A+ERASE at a real synthetic-host cell. Model observes historical physical material and surface, not hidden tool mapping.',
   strict_counterfactual:'At paint decision, identical input after removing specifically the corresponding current surface receipt; distinct legal optimal paint commands.'},
  splits,shards,total_policy_decisions:total,source_worlds:geometry.size,
  limitations:['Synthetic graph/material host, not native CyberSand ABI','Oracle prelude includes scripted host-accepted calibration, not a learned probing policy',
   'Surface mapping is a fixed binary A/B toy law for the episode; model can solve with a per-surface last-receipt parser',
   'No root SLM training, neural inference latency/RSS, native game acceptance or Actions shard retention']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;}
if(require.main===module){const [dir,flag]=process.argv.slice(2);ok(dir,'usage node contextual_probe_v49.js OUTPUT_NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:flag==='--quick'}).splits,null,2));}
module.exports={SPLITS,arms,site,label,configure,command,probe,prepare,observe,lawFor,teacher,mask,ablateMission,ablateGate,rollout,world,ablateAllProbes,generate,H};
