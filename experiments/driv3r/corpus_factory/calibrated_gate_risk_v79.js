#!/usr/bin/env node
'use strict';
// Corpus data only: a synthetic physical graph host, NEVER a native CyberSand ABI claim.
// Historical physical actuator trials shift the ex-ante probability of an unreported gate flip.
// The labelled action is optimal *before* the hidden flip is resolved; never clairvoyant.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ensure=(yes,msg)=>{if(!yes)throw Error('v79: '+msg)};
const DIRECTIONS=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const SIZES={TRAIN:6,DEV:3,HELDOUT:3};
const PAST_TRIALS=4;
const K=(x,y)=>`${x},${y}`;
function world(split,i){
 const width=72+({TRAIN:0,DEV:27,HELDOUT:59}[split])+i*3,mid=Math.floor(width*0.44);
 const s1top=4+(i*7+3)%(mid-8),s1bottom=4+(i*5+1)%(mid-8);
 const s2=mid+5+(i*11+3)%(width-mid-10);
 const cells=new Set([K(-1,1),K(0,1),K(mid,1),K(width,1)]);
 for(let x=0;x<=width;x++)for(const y of [0,2])cells.add(K(x,y));
 for(let x=4;x<width-2;x+=4)cells.add(K(x,(Math.floor(x/4)+i)%2===0?-1:3));
 const geo={width,mid,stage1_gates:[[s1top,0],[s1bottom,2]],stage2_gates:[[s2,0],[s2,2]],cells:[...cells].sort()};
 return {split,index:i,width,mid,cells,stage1:{TOP:s1top,BOTTOM:s1bottom},stage2:{TOP:s2,BOTTOM:s2},geometry_sha256:sha(geo)};
}
function host(w){return {w,pos:[0,1],tick:0,gate:{S1_TOP:true,S1_BOTTOM:true,S2_TOP:true,S2_BOTTOM:true},inbox:[],painted:false,done:false,switched:false};}
function gateFor(h,x,y){if(y!==0&&y!==2)return null;const lane=y===0?'TOP':'BOTTOM';for(const s of ['S1','S2'])if(h.w[s==='S1'?'stage1':'stage2'][lane]===x)return s+'_'+lane;return null;}
function legal(h){const commands=[];for(const [cmd,dx,dy] of DIRECTIONS){const x=h.pos[0]+dx,y=h.pos[1]+dy;if(!h.w.cells.has(K(x,y)))continue;const gate=gateFor(h,x,y);if(gate&&!h.gate[gate])continue;commands.push(cmd);}if(K(...h.pos)===K(h.w.width,1))commands.push(h.painted?'STOP':'PAINT_A');return commands;}
function observe(h){const local={};for(const s of ['S1','S2'])for(const lane of ['TOP','BOTTOM']){const x=h.w[s==='S1'?'stage1':'stage2'][lane];local[s+'_'+lane]=(h.pos[1]===(lane==='TOP'?0:2)&&Math.abs(h.pos[0]-x)<=1)?h.gate[s+'_'+lane]:null;}
 return {schema:'driv3r/synthetic-calibrated-gate-host/1',tick:h.tick,position:h.pos.slice(),goal:[h.w.width,1],midpoint:[h.w.mid,1],legal:legal(h),local_gate_visible:local,reports:h.inbox.map(x=>({...x}))};}
function act(h,cmd){ensure(legal(h).includes(cmd),'illegal '+cmd+' @ '+K(...h.pos));const d=DIRECTIONS.find(x=>x[0]===cmd);if(d){h.pos[0]+=d[1];h.pos[1]+=d[2];}else if(cmd==='PAINT_A')h.painted=true;else if(cmd==='STOP')h.done=true;h.tick++;h.inbox=[];if(!h.switched&&K(...h.pos)===K(h.w.mid,1)){h.switched=true;return {type:'MIDPOINT_ENTERED',tick:h.tick};}return null;}
function physicalPlan(h,target){const todo=[[...h.pos,[]]],seen=new Set([K(...h.pos)]);for(let j=0;j<todo.length;j++){const [x,y,steps]=todo[j];if(K(x,y)===K(...target))return steps;for(const [cmd,dx,dy] of DIRECTIONS){const xx=x+dx,yy=y+dy,key=K(xx,yy);if(seen.has(key)||!h.w.cells.has(key))continue;const g=gateFor(h,xx,yy);if(g&&!h.gate[g])continue;seen.add(key);todo.push([xx,yy,[...steps,cmd]]);}}throw Error('unreachable goal');}
function measuredReports(h,s){return ['TOP','BOTTOM'].map(lane=>({type:'REMOTE_GATE_REPORT',section:s,lane,measure_tick:h.tick,status:h.gate[s+'_'+lane]?'OPEN':'CLOSED'}));}
function receiptChoice(rows,s){const latest={};for(const row of rows)for(const r of row.input.reports){if(r.type!=='REMOTE_GATE_REPORT'||r.section!==s)continue;const old=latest[r.lane];if(!old||old.measure_tick<r.measure_tick)latest[r.lane]=r;}
 ensure(latest.TOP&&latest.BOTTOM&&latest.TOP.measure_tick===latest.BOTTOM.measure_tick,'missing gate reports');ensure(latest.TOP.status!==latest.BOTTOM.status,'bad gate state');return latest.TOP.status==='OPEN'?'TOP':'BOTTOM';}
function setOpen(h,s,lane){h.gate[s+'_TOP']=lane==='TOP';h.gate[s+'_BOTTOM']=lane==='BOTTOM';}
function trialPattern(riskCount,rotation){const mask=Array.from({length:PAST_TRIALS},(_,i)=>i<riskCount);return Array.from({length:PAST_TRIALS},(_,i)=>mask[(i+rotation)%PAST_TRIALS]);}
function routeCost(w,reported,initial,flipped){const h=host(w);h.pos=[w.mid,1];setOpen(h,'S2',flipped?(reported==='TOP'?'BOTTOM':'TOP'):reported);
 const initialCmd=initial==='TOP'?'MOVE_N':'MOVE_S';act(h,initialCmd);let n=1;while(legal(h).includes('MOVE_E')){act(h,'MOVE_E');n++;}for(const cmd of physicalPlan(h,[w.width,1])){act(h,cmd);n++;}return n+2;}
function expectedDecision(w,reported,riskCount){const candidates={};for(const lane of ['TOP','BOTTOM']){const stable=routeCost(w,reported,lane,false),flipped=routeCost(w,reported,lane,true);candidates[lane]={stable,flipped,expected_quarter_actions:stable*(4-riskCount)+flipped*riskCount};}
 ensure(candidates.TOP.expected_quarter_actions!==candidates.BOTTOM.expected_quarter_actions,'ambiguous optimal label');
 return {lane:candidates.TOP.expected_quarter_actions<candidates.BOTTOM.expected_quarter_actions?'TOP':'BOTTOM',costs:candidates};}
function makeEpisode(w,s1,s2,receiptOrder,riskCount,flipIndex){ensure([1,3].includes(riskCount)&&flipIndex>=0&&flipIndex<4,'risk arm');const h=host(w),rows=[],events=[];
 function record(cmd,weight=0){const input=observe(h);ensure(input.legal.includes(cmd),'unavailable '+cmd);rows.push({input,action:cmd,weight,focus_loss_weight:weight});const event=act(h,cmd);if(event)events.push(event);return event;}
 const firstReports={S1:measuredReports(h,'S1'),S2:measuredReports(h,'S2')};let fresh1=null,fresh2=null;
 const rotated=trialPattern(riskCount,(w.index+Number(s1==='BOTTOM')+Number(s2==='BOTTOM'))%4);
 for(let t=0;t<220;t++){
  if(t===10){setOpen(h,'S1',s1);fresh1=measuredReports(h,'S1');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S1',tick:h.tick,open:s1});}
  if(t===29){setOpen(h,'S2','TOP');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:'TOP'});}
  if(t>=30&&t<34){const before=h.gate.S2_TOP?'TOP':'BOTTOM';const changed=rotated[t-30];if(changed)setOpen(h,'S2',before==='TOP'?'BOTTOM':'TOP');const after=h.gate.S2_TOP?'TOP':'BOTTOM';
   const receipt={type:'REMOTE_ACTUATOR_TRIAL',section:'S2',trial:t-30,measure_tick:h.tick,before_open:before,after_open:after};h.inbox.push(receipt);events.push({type:'PHYSICAL_ACTUATOR_TRIAL',tick:h.tick,trial:t-30,before_open:before,after_open:after});}
  if(t===50){setOpen(h,'S2',s2);fresh2=measuredReports(h,'S2');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:s2});}
  for(const [s,at,early,late] of [['S1',[12,22],firstReports.S1,fresh1],['S2',[52,62],firstReports.S2,fresh2]])if(at.includes(t)){
   const moreRecent=receiptOrder==='FRESH_FIRST'?t===at[0]:t===at[1];const delivered=(moreRecent?late:early).map(x=>({...x}));h.inbox.push(...delivered);events.push({type:'REPORT_DELIVERED',section:s,tick:h.tick,measure_tick:delivered[0].measure_tick});}
  record(t%2===0?'MOVE_W':'MOVE_E');
 }
 ensure(K(...h.pos)==='0,1'&&h.tick===220,'patrol');const firstIndex=rows.length,firstInput=observe(h);
 const reported1=receiptChoice(rows,'S1'),reported2=receiptChoice(rows,'S2');ensure(reported1===s1&&reported2===s2,'chronological report wrong');
 const firstLabel=s1==='TOP'?'MOVE_N':'MOVE_S';for(const cmd of physicalPlan(h,[w.mid,1]))record(cmd,flipIndex===0?1:0);
 ensure(h.switched&&K(...h.pos)===K(w.mid,1),'midpoint');
 const flip=flipIndex<riskCount;if(flip){setOpen(h,'S2',s2==='TOP'?'BOTTOM':'TOP');events.push({type:'UNREPORTED_PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:s2==='TOP'?'BOTTOM':'TOP'});}
 const secondIndex=rows.length,secondInput=observe(h),decision=expectedDecision(w,reported2,riskCount),chosen=decision.lane;
 record(chosen==='TOP'?'MOVE_N':'MOVE_S',flipIndex===0?1:0);while(legal(h).includes('MOVE_E'))record('MOVE_E',1);
 const blocked=chosen!==(flip?(s2==='TOP'?'BOTTOM':'TOP'):s2),recoveryIndex=blocked?rows.length:-1;
 if(blocked){ensure(h.pos[0]===w.stage2[chosen]-1,'no physical block');}
 for(const cmd of physicalPlan(h,[w.width,1]))record(cmd,1);
 record('PAINT_A',1);record('STOP',1);ensure(h.done&&h.painted&&K(...h.pos)===K(w.width,1),'terminal');
 if(blocked)ensure(rows[recoveryIndex].action==='MOVE_W','no legal recovery');
 const pilot=rows.slice(0,220).flatMap(row=>row.input.reports.filter(r=>r.type==='REMOTE_ACTUATOR_TRIAL'));
 ensure(pilot.length===4&&pilot.filter(p=>p.before_open!==p.after_open).length===riskCount,'calibration evidence');
 const certificate={schema:'driv3r/corpus_factory/calibrated-gate-risk-v79-cert/1',split:w.split,world_index:w.index,geometry_sha256:w.geometry_sha256,
  stage1_open:s1,stage2_reported_open:s2,receipt_order:receiptOrder,pilot_change_count:riskCount,trial_rotation:(w.index+Number(s1==='BOTTOM')+Number(s2==='BOTTOM'))%4,flip_index:flipIndex,hidden_midpoint_flip:flip,
  first_decision_step:firstIndex,second_decision_step:secondIndex,recovery_step:recoveryIndex,first_decision_input_sha256:sha(firstInput),second_decision_input_sha256:sha(secondInput),
  first_action_label:firstLabel,second_action_label:chosen==='TOP'?'MOVE_N':'MOVE_S',expected_quarter_costs:decision.costs,
  actual_second_lane_open:flip?(s2==='TOP'?'BOTTOM':'TOP'):s2,first_choice_physically_open:!blocked,
  events,total_actions:rows.length,episode_sha256:sha(rows),terminal_success:true,synthetic_not_native_CyberSand:true};
 return {episode:{steps:rows},certificate};
}
function write(dir,quick=false){ensure(!fs.existsSync(dir),'output already exists');fs.mkdirSync(dir,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/calibrated-gate-risk-v79/1',date:'2026-11-19',source_sha256:sha(fs.readFileSync(__filename,'utf8')),synthetic_not_native_CyberSand:true,
  contract:'Only positive-weight TRAIN host legal actions. First-route prefix and second-junction decision are loss-masked on flip_index 1..3 to prevent repeated identical observed prefixes dominating. Calibration is measured prior to decision; no hidden midpoint transition or certificate in model input. Labels minimise expected physical travel cost under observed trials, never future-known flip.',files:{},splits:{}};
 const known=new Set();for(const split of ['TRAIN','DEV','HELDOUT']){
  const size=quick?1:SIZES[split],streams=[],certs=[],stat={worlds:[],episodes:0,legal_actions:0,terminal_successes:0,physical_wrong_route_recoveries:0,calibration_sensitive_groups:0,expected_optimal_action_correct:0,observed_pilot_trials:0};
  for(let i=0;i<size;i++){const w=world(split,i);ensure(!known.has(w.geometry_sha256),'duplicate world');known.add(w.geometry_sha256);stat.worlds.push({index:i,width:w.width,mid:w.mid,geometry_sha256:w.geometry_sha256,stage1_gates:w.stage1,stage2_gates:w.stage2});
   const groups=new Map();for(const a of ['TOP','BOTTOM'])for(const b of ['TOP','BOTTOM'])for(const order of ['FRESH_FIRST','STALE_FIRST'])for(const risk of [1,3])for(let fi=0;fi<4;fi++){
    const {episode,certificate}=makeEpisode(w,a,b,order,risk,fi);certificate.episode=stat.episodes;streams.push(episode);certs.push(certificate);stat.episodes++;stat.legal_actions+=certificate.total_actions;stat.terminal_successes++;stat.observed_pilot_trials+=4;
    if(certificate.recovery_step>=0)stat.physical_wrong_route_recoveries++;if(certificate.first_choice_physically_open)stat.expected_optimal_action_correct++;
    const key=`${a}/${b}/${order}`;if(!groups.has(key))groups.set(key,{labels:new Set(),inputs:new Set()});groups.get(key).labels.add(certificate.second_action_label);groups.get(key).inputs.add(certificate.second_decision_input_sha256);
   }
   for(const g of groups.values()){ensure(g.labels.size===2&&g.inputs.size===1,'no causally shifted action for same current input');stat.calibration_sensitive_groups++;}
  }
  const prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';for(const [suffix,items] of [['calibrated_stream.jsonl',streams],['calibrated_certificates_EVALUATOR_ONLY.jsonl',certs]]){
   const name=prefix+'_'+suffix,body=items.map(x=>JSON.stringify(x)+'\n').join('');fs.writeFileSync(path.join(dir,name),body,{flag:'wx'});manifest.files[name]={rows:items.length,bytes:Buffer.byteLength(body),sha256:sha(body)};
  }manifest.splits[split]=stat;
 }
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){ensure(process.argv[2],'usage node calibrated_gate_risk_v79.js OUTPUT [--quick]');const m=write(path.resolve(process.argv[2]),process.argv.includes('--quick'));console.log(JSON.stringify(Object.fromEntries(Object.entries(m.splits).map(([k,v])=>[k,{worlds:v.worlds.length,episodes:v.episodes,actions:v.legal_actions,optimal_realization:v.expected_optimal_action_correct}]))));}
module.exports={world,host,observe,act,makeEpisode,expectedDecision,routeCost,write,sha};
