#!/usr/bin/env node
'use strict';
// DRIV3R DATA LANE only. Synthetic game-like host, NOT the native CyberSand ABI.
// Oracle labels use source-time remote receipts; hidden changes are evaluated physically.
const fs=require('node:fs'), path=require('node:path'), crypto=require('node:crypto');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,msg)=>{if(!x)throw Error('v80: '+msg)};
const DIR=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const LANES=['TOP','CENTER','BOTTOM'],Y={TOP:0,CENTER:2,BOTTOM:4};
const SPLITS={TRAIN:6,DEV:3,HELDOUT:3};const pos=(x,y)=>`${x},${y}`;
function makeWorld(split,i){
 const width={TRAIN:46,DEV:74,HELDOUT:103}[split]+3*i;
 const j1=8+i%3, g1=j1+2, mid=22+i%3, j2=mid+8+i%2,g2=j2+2;
 ok(g1<mid-5 && g2<width-5,'geometry bounds');
 const cells=new Set([pos(-1,2)]);
 for(let x=0;x<=width;x++)for(let y of [0,2,4])cells.add(pos(x,y));
 const bridges=[j1,mid,j2,g1+2,g2+3,width];
 for(const x of bridges)for(let y=0;y<=4;y++)cells.add(pos(x,y));
 const geometry={width,j1,g1,mid,j2,g2,bridges,protected:[pos(width-3,0),pos(width-3,4)],cells:[...cells].sort()};
 return {...geometry,cells,geometry_sha256:hash(geometry),split,index:i};
}
function newHost(w){return {w,loc:[0,2],tick:0,gates:{S1:'CENTER',S2:'CENTER'},goal:'A',material:null,terminal:false,reports:[],discoveries:0};}
function gateAt(h,x,y){if(y!==0&&y!==2&&y!==4)return null;
 if(x===h.w.g1)return {s:'S1',lane:LANES.find(l=>Y[l]===y)};
 if(x===h.w.g2)return {s:'S2',lane:LANES.find(l=>Y[l]===y)};
 return null;}
function legal(h){const cmds=[];for(const [c,dx,dy] of DIR){const x=h.loc[0]+dx,y=h.loc[1]+dy;if(!h.w.cells.has(pos(x,y)))continue;
 const g=gateAt(h,x,y);if(g&&h.gates[g.s]!==g.lane)continue;cmds.push(c);}
 if(pos(...h.loc)===pos(h.w.width,2)){
  if(h.material===null){cmds.push('PAINT_A','PAINT_B');}
  else{cmds.push('ERASE');if(h.material===h.goal)cmds.push('STOP');}
 }
 return cmds;}
function observation(h){const visible={};for(const s of ['S1','S2'])for(const l of LANES){const x=h.w[s==='S1'?'g1':'g2'];
 visible[s+'_'+l]=h.loc[0]>=x-1 && h.loc[0]<=x && h.loc[1]===Y[l] ? (h.gates[s]===l?'OPEN':'BLOCKED'):null;}
 return {schema:'driv3r/synthetic-three-lane-host/1',tick:h.tick,position:h.loc.slice(),goal_site:[h.w.width,2],legal:legal(h),local_gate_visible:visible,painted_material:h.material,reports:h.reports.map(r=>({...r}))};}
function take(h,c){ok(legal(h).includes(c),'illegal '+c+' @ '+pos(...h.loc));const d=DIR.find(v=>v[0]===c);
 if(d){h.loc=[h.loc[0]+d[1],h.loc[1]+d[2]];}else if(c.startsWith('PAINT_'))h.material=c.slice(-1);else if(c==='ERASE')h.material=null;else if(c==='STOP')h.terminal=true;
 h.tick++;h.reports=[];}
function bfs(h,target){const q=[[...h.loc,[]]],seen=new Set([pos(...h.loc)]);
 for(let i=0;i<q.length;i++){const [x,y,p]=q[i];if(pos(x,y)===pos(...target))return p;
 for(const [c,dx,dy] of DIR){const X=x+dx,Y=y+dy,k=pos(X,Y);if(seen.has(k)||!h.w.cells.has(k))continue;const gate=gateAt(h,X,Y);
 if(gate&&h.gates[gate.s]!==gate.lane)continue;seen.add(k);q.push([X,Y,[...p,c]]);}}
 throw Error('no physical path');}
function receipts(h){return [{type:'REMOTE_GATE',section:'S1',sample_tick:h.tick,open:h.gates.S1},
 {type:'REMOTE_GATE',section:'S2',sample_tick:h.tick,open:h.gates.S2},
 {type:'REMOTE_OBJECTIVE',sample_tick:h.tick,material:h.goal}];}
function latest(rows,type,section){let result=null;for(const row of rows)for(const r of row.input.reports){if(r.type!==type||(section&&r.section!==section))continue;
 if(result===null||result.sample_tick<r.sample_tick)result=r;}ok(result,'missing history '+type+'/'+section);return result;}
const moveFor={TOP:'MOVE_N',CENTER:'MOVE_E',BOTTOM:'MOVE_S'};
function runEpisode(w,s1,s2,goal,order,surprise){const h=newHost(w),steps=[],events=[];
 const event=(type,details)=>events.push({at_step:steps.length,type,...details});
 const record=(c,weight=0,role='trajectory')=>{const input=observation(h);ok(input.legal.includes(c),'not legal '+c);steps.push({input,action:c,loss_weight:weight,role});take(h,c);};
 const stale=receipts(h),fresh={};
 for(let t=0;t<250;t++){
  if(t===20){h.gates.S1=s1;event('GATE_SET',{section:'S1',open:s1});fresh.S1={...receipts(h)[0]};}
  if(t===40){h.gates.S2=s2;event('GATE_SET',{section:'S2',open:s2});fresh.S2={...receipts(h)[1]};}
  if(t===44){h.goal=goal;event('GOAL_SET',{material:goal});fresh.GOAL={...receipts(h)[2]};}
  if(t===60||t===90){const recent=order==='FRESH_FIRST'?t===60:t===90;
   h.reports.push(...(recent?[fresh.S1,fresh.S2,fresh.GOAL]:stale).map(r=>({...r})));
   event('REPORT_RECEIVED',{sample_ticks:h.reports.map(r=>r.sample_tick),count:h.reports.length});}
  record(t%2===0?'MOVE_W':'MOVE_E');
 }
 ok(pos(...h.loc)==='0,2','patrol end');
 const chooseS1=latest(steps,'REMOTE_GATE','S1').open,chooseS2=latest(steps,'REMOTE_GATE','S2').open,chooseGoal=latest(steps,'REMOTE_OBJECTIVE').material;
 ok(chooseS1===s1&&chooseS2===s2&&chooseGoal===goal,'chronology failed');
 for(const c of bfs(h,[w.j1,2]))record(c,0);
 const first_step=steps.length,first_input=hash(observation(h)),first_action=moveFor[s1];
 const firstPath=bfs(h,[w.mid,2]);ok(firstPath[0]===first_action,'first route unexpected '+firstPath[0]+'/'+first_action);
 firstPath.forEach((c,i)=>record(c,!surprise&&i===0?1:0,i===0?'first_route':'stage1_travel'));
 for(const c of bfs(h,[w.j2,2]))record(c,0);
 const second_step=steps.length,second_input=hash(observation(h)),second_action=moveFor[s2];
 const physicalS2=surprise?LANES[(LANES.indexOf(s2)+1)%3]:s2;
 if(surprise){h.gates.S2=physicalS2;event('UNREPORTED_GATE_SET',{section:'S2',open:physicalS2});}
 ok(legal(h).includes(second_action),'second choice not legal');
 record(second_action,!surprise?1:0,'second_route');
 // Follow the earlier remote report until a physical obstruction is actually observed.
 if(s2!=='CENTER')record(second_action,0,'branch_entry');
 while(h.loc[0]<w.g2-1)record('MOVE_E',0,'approach_reported_gate');
 let discovered=false;
 if(!legal(h).includes('MOVE_E')){discovered=true;h.discoveries++;event('PHYSICAL_OBSTRUCTION_DISCOVERED',{at:[...h.loc],attempted:'MOVE_E'});}
 else record('MOVE_E',0,'cross_reported_gate');
 // After an observed obstruction, the deterministic teacher can plan a legal recovery.
 // This future outcome never appears in the earlier first/second-decision model input.
 for(const c of bfs(h,[w.width,2]))record(c,1,discovered?'physical_recovery':'terminal_travel');
 const goal_step=steps.length,goal_input=hash(observation(h)),goal_action='PAINT_'+goal;
 const wrongPaint=surprise&&order==='STALE_FIRST';
 if(wrongPaint){record('PAINT_'+(goal==='A'?'B':'A'),0,'masked_bad_paint');record('ERASE',1,'paint_recovery');}
 record(goal_action,1,'goal_memory');record('STOP',1,'terminal');
 ok(h.terminal&&h.material===h.goal&&pos(...h.loc)===pos(w.width,2),'terminal fail');
 const cert={schema:'driv3r/corpus_factory/three-lane-loop-v80-cert/1',split:w.split,world_index:w.index,geometry_sha256:w.geometry_sha256,
  stage1_open:s1,stage2_reported_open:s2,stage2_physical_open:physicalS2,goal_material:goal,report_order:order,hidden_stage2_shift:surprise,
  first_step,second_step,goal_step,first_action,second_action,goal_action,first_observation_sha256:first_input,second_observation_sha256:second_input,goal_observation_sha256:goal_input,
  physical_obstruction_discovered:discovered,wrong_paint_recovered:wrongPaint,events,steps:steps.length,episode_sha256:hash(steps),terminal_success:true,synthetic_not_native_CyberSand:true};
 return {steps,cert};}
function generate(dir,quick=false){ok(!fs.existsSync(dir),'output exists');fs.mkdirSync(dir,{recursive:true});
 const m={schema:'driv3r/corpus_factory/three-lane-loop-v80/1',synthetic_not_native_CyberSand:true,source_sha256:hash(fs.readFileSync(__filename,'utf8')),
   split_policy:'geometry and trajectory splits disjoint by width; DEV/HELDOUT evaluator only',files:{},splits:{}};
 const worlds=new Set();for(const split of ['TRAIN','DEV','HELDOUT']){
  const streams=[],certs=[],stat={worlds:[],episodes:0,legal_actions:0,success:0,physical_recoveries:0,wrong_paint_recoveries:0,ambiguity_first:0,ambiguity_goal:0,
   first_legal_labeled:{MOVE_N:0,MOVE_E:0,MOVE_S:0},goal_labels:{PAINT_A:0,PAINT_B:0},supervised_landmarks:0};
  for(let i=0;i<(quick?1:SPLITS[split]);i++){const w=makeWorld(split,i);ok(!worlds.has(w.geometry_sha256),'split geometry leak');worlds.add(w.geometry_sha256);
   stat.worlds.push({index:i,width:w.width,j1:w.j1,g1:w.g1,mid:w.mid,j2:w.j2,g2:w.g2,bridges:w.bridges,geometry_sha256:w.geometry_sha256});
   const groups=new Map(),goals=new Map();
   for(const s1 of LANES)for(const s2 of LANES)for(const goal of ['A','B'])for(const order of ['FRESH_FIRST','STALE_FIRST'])for(const surprise of [false,true]){
    const {steps,cert}=runEpisode(w,s1,s2,goal,order,surprise);cert.episode=stat.episodes;streams.push({steps});certs.push(cert);stat.episodes++;
    stat.legal_actions+=steps.length;stat.success++;if(cert.physical_obstruction_discovered)stat.physical_recoveries++;
    if(cert.wrong_paint_recovered)stat.wrong_paint_recoveries++;
    if(!surprise){stat.first_legal_labeled[cert.first_action]++;stat.goal_labels[cert.goal_action]++;stat.supervised_landmarks+=2;}
    const key=`${s2}/${goal}/${order}/${surprise}`;if(!groups.has(key))groups.set(key,{inputs:new Set(),actions:new Set()});groups.get(key).inputs.add(cert.first_observation_sha256);groups.get(key).actions.add(cert.first_action);
    const key2=`${s1}/${s2}/${order}/${surprise}`;if(!goals.has(key2))goals.set(key2,{labels:new Set()});goals.get(key2).labels.add(cert.goal_action);
   }
   for(const v of groups.values()){ok(v.inputs.size===1&&v.actions.size===3,'no three-way history contrast');stat.ambiguity_first++;}
   for(const v of goals.values()){ok(v.labels.size===2,'no goal-memory contrast');stat.ambiguity_goal++;}
  }
  const pre=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  for(const [name,items] of [[pre+'_three_lane_stream.jsonl',streams],[pre+'_three_lane_certificates_EVALUATOR_ONLY.jsonl',certs]]){
   const out=items.map(x=>JSON.stringify(x)+'\n').join('');fs.writeFileSync(path.join(dir,name),out,{flag:'wx'});m.files[name]={rows:items.length,bytes:Buffer.byteLength(out),sha256:hash(out)};}
  m.splits[split]=stat;
 }
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(m,null,2)+'\n',{flag:'wx'});return m;}
if(require.main===module){ok(process.argv[2],'usage node three_lane_loop_v80.js OUT [--quick]');let m=generate(path.resolve(process.argv[2]),process.argv.includes('--quick'));console.log(JSON.stringify(Object.fromEntries(Object.entries(m.splits).map(([k,v])=>[k,{worlds:v.worlds.length,episodes:v.episodes,actions:v.legal_actions,recoveries:v.physical_recoveries,contrasts:v.ambiguity_first}]))));}
module.exports={makeWorld,newHost,legal,observation,take,bfs,receipts,runEpisode,generate,hash};
