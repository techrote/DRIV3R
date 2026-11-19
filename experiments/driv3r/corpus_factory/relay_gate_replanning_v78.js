#!/usr/bin/env node
'use strict';
// DATA ONLY: simulated host, not CyberSand or a native-ABI compatibility claim.
// Teaches observable action choices across two decisions and genuine physical replanning.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const assert=(p,m)=>{if(!p)throw Error('v78 '+m)};
const DIRECTIONS=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const SIZES={TRAIN:12,DEV:6,HELDOUT:6};
const K=(x,y)=>`${x},${y}`;
function world(split,index){
 const w=26+({TRAIN:0,DEV:28,HELDOUT:42}[split])+index*2, mid=Math.floor(w*.48);
 const s1top=3+(index*3+1)%(mid-6),s1bottom=3+(index*5+2)%(mid-6);
 const s2top=mid+3+(index*3+1)%(w-mid-6),s2bottom=mid+3+(index*5+2)%(w-mid-6);
 const cells=new Set([K(-1,1),K(0,1),K(mid,1),K(w,1)]);
 for(let x=0;x<=w;x++)for(let y of [0,2])cells.add(K(x,y));
 // One-cell side pockets increase topology diversity without bypassing gates.
 for(let x=2;x<w-1;x+=3){const y=(Math.floor(x/3)+index)%2===0?-1:3;cells.add(K(x,y));}
 const geo={width:w,mid,stage1_gates:[[s1top,0],[s1bottom,2]],stage2_gates:[[s2top,0],[s2bottom,2]],cells:[...cells].sort()};
 return {split,index,width:w,mid,cells,stage1:{TOP:s1top,BOTTOM:s1bottom},stage2:{TOP:s2top,BOTTOM:s2bottom},geometry_sha256:sha(geo)};
}
function host(w){return {w,pos:[0,1],tick:0,gate:{S1_TOP:true,S1_BOTTOM:true,S2_TOP:true,S2_BOTTOM:true},inbox:[],painted:false,done:false,switched:false};}
function gateFor(h,x,y){if(y!==0&&y!==2)return null;for(const section of ['S1','S2']){
 const gx=h.w[section==='S1'?'stage1':'stage2'][y===0?'TOP':'BOTTOM'];
 if(x===gx)return section+'_'+(y===0?'TOP':'BOTTOM');
 }return null;}
function legal(h){const out=[];for(const [cmd,dx,dy] of DIRECTIONS){let x=h.pos[0]+dx,y=h.pos[1]+dy;
 if(!h.w.cells.has(K(x,y)))continue;const g=gateFor(h,x,y);if(g&&!h.gate[g])continue;out.push(cmd);}
 if(K(...h.pos)===K(h.w.width,1))out.push(h.painted?'STOP':'PAINT_A');return out;}
function observe(h){const [x,y]=h.pos,local={};for(const sec of ['S1','S2'])for(const lane of ['TOP','BOTTOM']){
 const g=h.w[sec==='S1'?'stage1':'stage2'][lane];local[sec+'_'+lane]=(y===(lane==='TOP'?0:2)&&Math.abs(x-g)<=1)?h.gate[sec+'_'+lane]:null;
 }return {schema:'driv3r/synthetic-two-relay-host/1',tick:h.tick,position:h.pos.slice(),goal:[h.w.width,1],midpoint:[h.w.mid,1],legal:legal(h),local_gate_visible:local,reports:h.inbox.map(r=>({...r}))};}
function act(h,cmd){assert(legal(h).includes(cmd),'illegal '+cmd+' '+K(...h.pos));
 const move=DIRECTIONS.find(a=>a[0]===cmd);if(move){h.pos[0]+=move[1];h.pos[1]+=move[2];}
 else if(cmd==='PAINT_A')h.painted=true;else if(cmd==='STOP')h.done=true;
 h.tick++;h.inbox=[];
 let event=null;if(K(...h.pos)===K(h.w.mid,1)&&!h.switched){h.switched=true;event={type:'MIDPOINT_ENTERED',tick:h.tick};}
 return event;
}
function physicalPlan(h,destination){const q=[[...h.pos,[]]],seen=new Set([K(...h.pos)]);
 for(let i=0;i<q.length;i++){const [x,y,steps]=q[i];if(K(x,y)===K(...destination))return steps;
  for(const [cmd,dx,dy] of DIRECTIONS){const xx=x+dx,yy=y+dy,key=K(xx,yy);if(seen.has(key)||!h.w.cells.has(key))continue;
   const gate=gateFor(h,xx,yy);if(gate&&!h.gate[gate])continue;
   seen.add(key);q.push([xx,yy,[...steps,cmd]]);
  }}throw Error('destination unreachable');}
function knownLane(steps,section){const observed={};for(const row of steps)for(const r of row.input.reports){if(r.section!==section)continue;
 const previous=observed[r.lane];if(!previous||r.measure_tick>previous.measure_tick)observed[r.lane]=r;}
 assert(observed.TOP&&observed.BOTTOM&&observed.TOP.measure_tick===observed.BOTTOM.measure_tick,'missing latest report '+section);
 assert(observed.TOP.status!==observed.BOTTOM.status,'ambiguous report '+section);return observed.TOP.status==='OPEN'?'TOP':'BOTTOM';}
function emitReport(h,section,snapshot,events){for(const lane of ['TOP','BOTTOM'])h.inbox.push({type:'REMOTE_GATE_REPORT',section,lane,measure_tick:snapshot.measure_tick,status:snapshot[lane]});
 events.push({type:'REPORT_DELIVERED',section,delivered_tick:h.tick,measure_tick:snapshot.measure_tick});}
function snap(h,section){return {measure_tick:h.tick,TOP:h.gate[section+'_TOP']?'OPEN':'CLOSED',BOTTOM:h.gate[section+'_BOTTOM']?'OPEN':'CLOSED'};}
function makeEpisode(w,choice1,choice2,order,flip,offpolicy){const h=host(w),rows=[],events=[];
 const recording=(command,weight=0)=>{const input=observe(h);assert(input.legal.includes(command),'not legal '+command);rows.push({input,action:command,weight,focus_loss_weight:weight});const event=act(h,command);if(event)events.push(event);return event;};
 const initial1=snap(h,'S1'),initial2=snap(h,'S2');let fresh1,fresh2;
 for(let i=0;i<200;i++){
  if(i===10){h.gate.S1_TOP=choice1==='TOP';h.gate.S1_BOTTOM=choice1==='BOTTOM';fresh1=snap(h,'S1');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S1',tick:h.tick,gate:{TOP:h.gate.S1_TOP,BOTTOM:h.gate.S1_BOTTOM}});}
  if(i===50){h.gate.S2_TOP=choice2==='TOP';h.gate.S2_BOTTOM=choice2==='BOTTOM';fresh2=snap(h,'S2');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,gate:{TOP:h.gate.S2_TOP,BOTTOM:h.gate.S2_BOTTOM}});}
  for(const [section,at,earlier,later] of [['S1',[12,22],initial1,fresh1],['S2',[52,62],initial2,fresh2]])if(at.includes(i)){
   const s=order==='FRESH_FIRST'?(i===at[0]?later:earlier):(i===at[0]?earlier:later);emitReport(h,section,s,events);
  }
  recording(i%2===0?'MOVE_W':'MOVE_E');
 }
 assert(K(...h.pos)==='0,1'&&h.tick===200,'start mismatch');
 const firstInput=observe(h),first=rows.length;
 const report1=knownLane(rows,'S1'),report2=knownLane(rows,'S2');assert(report1===choice1&&report2===choice2,'non-privileged reports disagree');
 const good1=report1==='TOP'?'MOVE_N':'MOVE_S';let wrongRecovery=-1;
 if(offpolicy){recording(good1==='MOVE_N'?'MOVE_S':'MOVE_N',0);
  while(observe(h).legal.includes('MOVE_E'))recording('MOVE_E',0);
  assert(h.pos[0]===w.stage1[report1==='TOP'?'BOTTOM':'TOP']-1,'wrong route not at physical barrier');
  wrongRecovery=rows.length;
 }
 const path1=physicalPlan(h,[w.mid,1]);assert(path1.length>0,'stage1 path empty');for(const move of path1)recording(move,1);
 assert(h.switched&&K(...h.pos)===K(w.mid,1),'midpoint not reached');
 if(flip){h.gate.S2_TOP=!h.gate.S2_TOP;h.gate.S2_BOTTOM=!h.gate.S2_BOTTOM;
  events.push({type:'UNREPORTED_PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,gate:{TOP:h.gate.S2_TOP,BOTTOM:h.gate.S2_BOTTOM}});}
 const second=rows.length,secondInput=observe(h);
 const first2=report2==='TOP'?'MOVE_N':'MOVE_S';recording(first2,1);
 let surpriseRecovery=-1;
 while(observe(h).legal.includes('MOVE_E'))recording('MOVE_E',1);
 if(K(...h.pos)!==K(w.width,0)&&K(...h.pos)!==K(w.width,2)){
  assert(flip,'unexpected stale blockage');surpriseRecovery=rows.length;
  assert(h.pos[0]===w.stage2[report2]-1,'surprise gate not contacted');
 }
 for(const move of physicalPlan(h,[w.width,1]))recording(move,1);
 recording('PAINT_A',1);recording('STOP',1);
 assert(h.done&&h.painted&&K(...h.pos)===K(w.width,1),'terminal incomplete');
 assert(wrongRecovery===-1||rows[wrongRecovery].action==='MOVE_W','no wrong-route recovery');
 assert(surpriseRecovery===-1||rows[surpriseRecovery].action==='MOVE_W','no surprise recovery');
 const cert={schema:'driv3r/corpus_factory/relay-gate-v78-cert/1',split:w.split,world_index:w.index,geometry_sha256:w.geometry_sha256,
  stage1_open:choice1,stage2_reported_open:choice2,receipt_order:order,midpoint_flip:flip,offpolicy_stage1:offpolicy,
  first_decision_step:first,second_decision_step:second,wrong_recovery_step:wrongRecovery,surprise_recovery_step:surpriseRecovery,
  first_decision_input_sha256:sha(firstInput),second_decision_input_sha256:sha(secondInput),
  first_action_label:good1,second_action_label:first2,events,total_actions:rows.length,episode_sha256:sha(rows),terminal_success:true,
  synthetic_not_native_CyberSand:true};return {episode:{steps:rows},certificate:cert};
}
function write(dir,quick=false){assert(!fs.existsSync(dir),'directory exists');fs.mkdirSync(dir,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/relay-gate-replanning-v78/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,
  source_sha256:sha(fs.readFileSync(__filename,'utf8')),
  training_contract:'Only positive-weight TRAIN actions; all receipts are delivered historical observations; no evaluator certificate in model input; wrong-route actions loss-masked',files:{},splits:{}};
 const seen=new Set();
 for(const split of ['TRAIN','DEV','HELDOUT']){const n=quick?1:SIZES[split],stream=[],certs=[],stat={worlds:[],episodes:0,legal_actions:0,terminal_successes:0,offpolicy_wrong_corridor_recoveries:0,unannounced_gate_replans:0,first_choice_memory_conflicts:0,second_choice_memory_conflicts:0};
  for(let idx=0;idx<n;idx++){const w=world(split,idx);assert(!seen.has(w.geometry_sha256),'duplicate geometry');seen.add(w.geometry_sha256);
   stat.worlds.push({index:idx,width:w.width,mid:w.mid,geometry_sha256:w.geometry_sha256,stage1_gates:w.stage1,stage2_gates:w.stage2});
   const firstMap=new Map(),secondMap=new Map();
   for(const a of ['TOP','BOTTOM'])for(const b of ['TOP','BOTTOM'])for(const order of ['FRESH_FIRST','STALE_FIRST'])for(const flip of [false,true])for(const wrong of [false,true]){
    const {episode,certificate}=makeEpisode(w,a,b,order,flip,wrong);
    certificate.episode=stat.episodes;stream.push(episode);certs.push(certificate);
    stat.episodes++;stat.legal_actions+=certificate.total_actions;stat.terminal_successes++;
    if(wrong)stat.offpolicy_wrong_corridor_recoveries++;if(flip)stat.unannounced_gate_replans++;
    const firstHash=certificate.first_decision_input_sha256,secondHash=certificate.second_decision_input_sha256;
    if(!firstMap.has(firstHash))firstMap.set(firstHash,new Set());firstMap.get(firstHash).add(certificate.first_action_label);
    if(!secondMap.has(secondHash))secondMap.set(secondHash,new Set());secondMap.get(secondHash).add(certificate.second_action_label);
   }
   assert([...firstMap.values()].some(s=>s.size===2),'no first matched ambiguity');
   assert([...secondMap.values()].some(s=>s.size===2),'no second matched ambiguity');
   stat.first_choice_memory_conflicts++;stat.second_choice_memory_conflicts++;
  }
  const prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  for(const [suffix,objects] of [['relay_stream.jsonl',stream],['relay_certificates_EVALUATOR_ONLY.jsonl',certs]]){
   const filename=prefix+'_'+suffix,body=objects.map(o=>JSON.stringify(o)+'\n').join('');
   fs.writeFileSync(path.join(dir,filename),body,{flag:'wx'});manifest.files[filename]={rows:objects.length,bytes:Buffer.byteLength(body),sha256:sha(body)};
  }
  manifest.splits[split]=stat;
 }
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){assert(process.argv[2],'usage node relay_gate_replanning_v78.js OUTPUT_DIR [--quick]');
 const m=write(path.resolve(process.argv[2]),process.argv.includes('--quick'));
 console.log(JSON.stringify(Object.fromEntries(Object.entries(m.splits).map(([k,v])=>[k,{worlds:v.worlds.length,episodes:v.episodes,legal_actions:v.legal_actions,replans:v.unannounced_gate_replans}]))));}
module.exports={world,host,observe,act,makeEpisode,write,sha};
