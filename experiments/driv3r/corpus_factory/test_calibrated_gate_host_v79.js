#!/usr/bin/env node
'use strict';
// Independently reconstruct physical gate transitions and every legal action.
// No generator import or shared planner; seals training/evaluator boundaries.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const dig=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const chk=(x,s)=>{if(!x)throw Error('v79 independent AUDIT: '+s)};
const SIZES={TRAIN:6,DEV:3,HELDOUT:3};
const D=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const pt=(x,y)=>x+','+y;
function geometry(split,i){
 const width=72+({TRAIN:0,DEV:27,HELDOUT:59}[split])+3*i,mid=Math.floor(width*.44);
 const gates={S1:{TOP:4+(7*i+3)%(mid-8),BOTTOM:4+(5*i+1)%(mid-8)},S2:{}};
 const x=mid+5+(11*i+3)%(width-mid-10);gates.S2={TOP:x,BOTTOM:x};
 const cells=new Set([pt(-1,1),pt(0,1),pt(mid,1),pt(width,1)]);
 for(let x=0;x<=width;x++){cells.add(pt(x,0));cells.add(pt(x,2));}
 for(let x=4;x<width-2;x+=4)cells.add(pt(x,(Math.floor(x/4)+i)%2===0?-1:3));
 const geometryHash=dig({width,mid,stage1_gates:[[gates.S1.TOP,0],[gates.S1.BOTTOM,2]],stage2_gates:[[x,0],[x,2]],cells:[...cells].sort()});
 return {width,mid,cells,gates,geometryHash};
}
function init(g){return {g,pos:[0,1],tick:0,open:{S1_TOP:true,S1_BOTTOM:true,S2_TOP:true,S2_BOTTOM:true},painted:false,done:false,midEntered:false};}
function gateAt(h,x,y){if(y!==0&&y!==2)return null;for(const section of ['S1','S2']){const lane=y===0?'TOP':'BOTTOM';if(h.g.gates[section][lane]===x)return section+'_'+lane;}return null;}
function actions(h){const ans=[];for(const [cmd,dx,dy] of D){const x=h.pos[0]+dx,y=h.pos[1]+dy;if(!h.g.cells.has(pt(x,y)))continue;const gate=gateAt(h,x,y);if(gate&&!h.open[gate])continue;ans.push(cmd);}if(pt(...h.pos)===pt(h.g.width,1))ans.push(h.painted?'STOP':'PAINT_A');return ans;}
function obs(h,reports){const visible={};for(const sec of ['S1','S2'])for(const lane of ['TOP','BOTTOM']){const x=h.g.gates[sec][lane];visible[sec+'_'+lane]=(h.pos[1]===(lane==='TOP'?0:2)&&Math.abs(h.pos[0]-x)<=1)?h.open[sec+'_'+lane]:null;}
 return {schema:'driv3r/synthetic-calibrated-gate-host/1',tick:h.tick,position:h.pos.slice(),goal:[h.g.width,1],midpoint:[h.g.mid,1],legal:actions(h),local_gate_visible:visible,reports};}
function setGate(h,section,opened){h.open[section+'_TOP']=opened==='TOP';h.open[section+'_BOTTOM']=opened==='BOTTOM';}
function snapshot(h,section){return ['TOP','BOTTOM'].map(lane=>({type:'REMOTE_GATE_REPORT',section,lane,measure_tick:h.tick,status:h.open[section+'_'+lane]?'OPEN':'CLOSED'}));}
function routeDistance(g,reported,choice,physicalFlip){const h=init(g);h.pos=[g.mid,1];setGate(h,'S2',physicalFlip?(reported==='TOP'?'BOTTOM':'TOP'):reported);
 h.pos[1]=choice==='TOP'?0:2;let length=1;
 while(actions(h).includes('MOVE_E')){h.pos[0]++;length++;}
 // BFS from physical endpoint to terminal; do not depend on generator's solver.
 const q=[[h.pos[0],h.pos[1],0]],seen=new Set([pt(...h.pos)]);let found=-1;
 for(let n=0;n<q.length;n++){const [x,y,d]=q[n];if(x===g.width&&y===1){found=d;break;}
  for(const [,dx,dy] of D){const xx=x+dx,yy=y+dy,key=pt(xx,yy),bar=gateAt(h,xx,yy);if(seen.has(key)||!g.cells.has(key)||(bar&&!h.open[bar]))continue;seen.add(key);q.push([xx,yy,d+1]);}}
 chk(found>=0,'unreachable expected route');return length+found+2;
}
function optimal(g,reported,k){const c={};for(const lane of ['TOP','BOTTOM']){const a=routeDistance(g,reported,lane,false),b=routeDistance(g,reported,lane,true);c[lane]={stable:a,flipped:b,expected_quarter_actions:(4-k)*a+k*b};}
 chk(c.TOP.expected_quarter_actions!==c.BOTTOM.expected_quarter_actions,'ambiguous expected action');return {lane:c.TOP.expected_quarter_actions<c.BOTTOM.expected_quarter_actions?'TOP':'BOTTOM',costs:c};}
function checkEpisode(ep,c,split,index){
 const g=geometry(split,c.world_index),h=init(g),events=[];
 chk(c.schema==='driv3r/corpus_factory/calibrated-gate-risk-v79-cert/1'&&c.split===split&&c.episode===index,'certificate identity');
 chk(c.geometry_sha256===g.geometryHash,'geometry hash');chk(['TOP','BOTTOM'].includes(c.stage1_open)&&['TOP','BOTTOM'].includes(c.stage2_reported_open),'lane arm');
 chk(['FRESH_FIRST','STALE_FIRST'].includes(c.receipt_order)&&[1,3].includes(c.pilot_change_count)&&c.flip_index>=0&&c.flip_index<4,'risk/receipt arms');
 chk(c.hidden_midpoint_flip===(c.flip_index<c.pilot_change_count),'future arm probability contract');
 chk(c.trial_rotation===(c.world_index+Number(c.stage1_open==='BOTTOM')+Number(c.stage2_reported_open==='BOTTOM'))%4,'trial rotation');
 chk(c.first_decision_step===220&&ep.steps.length===c.total_actions&&dig(ep.steps)===c.episode_sha256,'episode hash');
 const pattern=Array.from({length:4},(_,i)=>((i+c.trial_rotation)%4)<c.pilot_change_count);
 let initial1=snapshot(h,'S1'),initial2=snapshot(h,'S2'),fresh1=null,fresh2=null,pilotCount=0,wrong=0,midCount=0,positive=0;
 const reportHist={S1:[],S2:[]};
 for(let j=0;j<ep.steps.length;j++){
  if(j===10){setGate(h,'S1',c.stage1_open);fresh1=snapshot(h,'S1');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S1',tick:h.tick,open:c.stage1_open});}
  let reports=[];
  if(j===29){setGate(h,'S2','TOP');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:'TOP'});}
  if(j>=30&&j<34){const prev=h.open.S2_TOP?'TOP':'BOTTOM';if(pattern[j-30])setGate(h,'S2',prev==='TOP'?'BOTTOM':'TOP');const after=h.open.S2_TOP?'TOP':'BOTTOM';
   reports.push({type:'REMOTE_ACTUATOR_TRIAL',section:'S2',trial:j-30,measure_tick:h.tick,before_open:prev,after_open:after});
   events.push({type:'PHYSICAL_ACTUATOR_TRIAL',tick:h.tick,trial:j-30,before_open:prev,after_open:after});pilotCount++;}
  if(j===50){setGate(h,'S2',c.stage2_reported_open);fresh2=snapshot(h,'S2');events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:c.stage2_reported_open});}
  for(const [section,times,early,late] of [['S1',[12,22],initial1,fresh1],['S2',[52,62],initial2,fresh2]])if(times.includes(j)){
   const isRecent=c.receipt_order==='FRESH_FIRST'?j===times[0]:j===times[1];const chosen=(isRecent?late:early).map(x=>({...x}));reports.push(...chosen);events.push({type:'REPORT_DELIVERED',section,tick:h.tick,measure_tick:chosen[0].measure_tick});reportHist[section].push(...chosen);}
  const expected=obs(h,reports),row=ep.steps[j];chk(JSON.stringify(expected)===JSON.stringify(row.input),`observation mismatch ${split}/${index}/${j}`);
  chk(expected.legal.includes(row.action),'illegal '+j+' '+row.action);chk((row.weight===0||row.weight===1)&&row.focus_loss_weight===row.weight,'loss weight');
  if(j<220)chk(row.weight===0&&row.action===(j%2===0?'MOVE_W':'MOVE_E'),'bad hidden-history patrol');else if(row.weight===1)positive++;
  if(j===220){chk(dig(expected)===c.first_decision_input_sha256,'first history input hash');chk(row.action===(c.stage1_open==='TOP'?'MOVE_N':'MOVE_S')&&row.weight===(c.flip_index===0?1:0),'first action target/mask');}
  if(j===c.second_decision_step){chk(h.pos[0]===g.mid&&h.pos[1]===1&&dig(expected)===c.second_decision_input_sha256,'second decision observation');
   const decision=optimal(g,c.stage2_reported_open,c.pilot_change_count);chk(row.action===(decision.lane==='TOP'?'MOVE_N':'MOVE_S')&&row.weight===(c.flip_index===0?1:0),'ex-ante action label/mask');
   chk(JSON.stringify(c.expected_quarter_costs)===JSON.stringify(decision.costs),'risk cost certificate');
   chk(c.second_action_label===row.action,'second label certificate');}
  if(j>220&&j<c.second_decision_step)chk(row.weight===(c.flip_index===0?1:0),'repeated route prefix must be masked');
  if(j===c.recovery_step){chk(row.action==='MOVE_W'&&row.weight===1,'recovery move');chk(!expected.legal.includes('MOVE_E'),'recovery does not follow a real block');wrong++;}
  const d=D.find(([command])=>command===row.action);if(d){h.pos[0]+=d[1];h.pos[1]+=d[2];}else if(row.action==='PAINT_A'){chk(!h.painted,'double paint');h.painted=true;}else if(row.action==='STOP'){chk(!h.done&&h.painted&&pt(...h.pos)===pt(g.width,1),'illegal terminal');h.done=true;}else throw Error('unknown action');
  h.tick++;if(!h.midEntered&&pt(...h.pos)===pt(g.mid,1)){h.midEntered=true;midCount++;events.push({type:'MIDPOINT_ENTERED',tick:h.tick});
   if(c.hidden_midpoint_flip){setGate(h,'S2',c.stage2_reported_open==='TOP'?'BOTTOM':'TOP');events.push({type:'UNREPORTED_PHYSICAL_GATE_CHANGE',section:'S2',tick:h.tick,open:c.stage2_reported_open==='TOP'?'BOTTOM':'TOP'});}}
 }
 chk(pilotCount===4&&midCount===1&&h.done&&h.painted,'missing terminal or pilot physical transitions');
 chk(dig(events)===dig(c.events),'event sequence mismatch');chk(c.terminal_success&&c.synthetic_not_native_CyberSand&&positive>0,'certificate completion');
 chk(c.first_action_label===(c.stage1_open==='TOP'?'MOVE_N':'MOVE_S'),'first label');
 const actual=c.hidden_midpoint_flip?(c.stage2_reported_open==='TOP'?'BOTTOM':'TOP'):c.stage2_reported_open,chosen=c.second_action_label==='MOVE_N'?'TOP':'BOTTOM';
 chk(c.actual_second_lane_open===actual&&c.first_choice_physically_open===(chosen===actual),'physical gate outcome mismatch');
 chk((c.recovery_step>=0)===(chosen!==actual)&&wrong===(chosen!==actual?1:0),'physical recovery certificate');
 chk(c.second_decision_step>220&&c.second_decision_step<ep.steps.length-2,'second step');
 for(const [section,hist,tick,open] of [['S1',reportHist.S1,10,c.stage1_open],['S2',reportHist.S2,50,c.stage2_reported_open]]){
  chk(hist.length===4,'reports cardinality');for(const lane of ['TOP','BOTTOM']){const latest=hist.filter(x=>x.lane===lane).sort((a,b)=>b.measure_tick-a.measure_tick)[0];chk(latest.measure_tick===tick&&latest.status===(lane===open?'OPEN':'CLOSED'),'measured state '+section);}}
 return c.total_actions;
}
function load(dir,name,m,rows){const raw=fs.readFileSync(path.join(dir,name),'utf8'),record=m.files[name];chk(record&&Buffer.byteLength(raw)===record.bytes&&dig(raw)===record.sha256,'shard integrity '+name);const lines=raw.trimEnd().split('\n');chk(lines.length===rows&&lines.length===record.rows,'shard count');return lines.map(JSON.parse);}
function audit(dir,quick=false){const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));chk(m.schema==='driv3r/corpus_factory/calibrated-gate-risk-v79/1'&&m.synthetic_not_native_CyberSand,'manifest');
 const seen=new Set(),result={schema:'driv3r/corpus_factory/v79-independent-physical-audit/1',synthetic_not_native_CyberSand:true,neural_model_evaluated:false,splits:{},totals:{worlds:0,episodes:0,legal_actions:0,recoveries:0,physically_correct_first_lanes:0,matched_risk_sensitive_groups:0}};
 for(const split of ['TRAIN','DEV','HELDOUT']){const size=quick?1:SIZES[split],prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  const ep=load(dir,prefix+'_calibrated_stream.jsonl',m,size*64),certs=load(dir,prefix+'_calibrated_certificates_EVALUATOR_ONLY.jsonl',m,size*64);
  let actions=0,recoveries=0,correct=0,groups=0;for(let i=0;i<ep.length;i++){
   const c=certs[i],arm=i%64,stage1=['TOP','BOTTOM'][Math.floor(arm/32)],stage2=['TOP','BOTTOM'][Math.floor((arm%32)/16)],order=['FRESH_FIRST','STALE_FIRST'][Math.floor((arm%16)/8)],risk=[1,3][Math.floor((arm%8)/4)],fi=arm%4;
   chk(c.world_index===Math.floor(i/64)&&c.stage1_open===stage1&&c.stage2_reported_open===stage2&&c.receipt_order===order&&c.pilot_change_count===risk&&c.flip_index===fi,'counterfactual order');
   actions+=checkEpisode(ep[i],c,split,i);recoveries+=Number(c.recovery_step>=0);correct+=Number(c.first_choice_physically_open);
  }
  for(let i=0;i<size;i++){const g=geometry(split,i);chk(!seen.has(g.geometryHash),'geometry leakage');seen.add(g.geometryHash);chk(m.splits[split].worlds[i].geometry_sha256===g.geometryHash,'geometry manifest');
   const subset=certs.slice(i*64,i*64+64),byDecision=new Map();for(const c of subset){const key=c.stage1_open+'/'+c.stage2_reported_open+'/'+c.receipt_order;
    if(!byDecision.has(key))byDecision.set(key,{current:new Set(),label:new Set(),physical:new Map(),outcomes:new Map()});const x=byDecision.get(key);x.current.add(c.second_decision_input_sha256);x.label.add(c.second_action_label);
    if(!x.physical.has(c.pilot_change_count))x.physical.set(c.pilot_change_count,[]);x.physical.get(c.pilot_change_count).push(c.hidden_midpoint_flip);
   }
   chk(byDecision.size===8,'missing paired groups');for(const g of byDecision.values()){chk(g.current.size===1&&g.label.size===2,'risk-dependent label absent while current observation identical');
    chk(g.physical.get(1).filter(Boolean).length===1&&g.physical.get(3).filter(Boolean).length===3,'physical flip probability not calibrated');groups++;}
  }
  const stat=m.splits[split];chk(stat.worlds.length===size&&stat.episodes===size*64&&stat.legal_actions===actions&&stat.terminal_successes===size*64,'manifest action counts');
  chk(stat.physical_wrong_route_recoveries===recoveries&&stat.expected_optimal_action_correct===correct&&stat.calibration_sensitive_groups===groups&&stat.observed_pilot_trials===size*64*4,'manifest derived aggregates');
  result.splits[split]={worlds:size,episodes:ep.length,legal_actions:actions,recoveries,physically_correct_first_lanes:correct,matched_risk_sensitive_groups:groups};
  const t=result.totals;t.worlds+=size;t.episodes+=ep.length;t.legal_actions+=actions;t.recoveries+=recoveries;t.physically_correct_first_lanes+=correct;t.matched_risk_sensitive_groups+=groups;
 }
 chk(result.totals.physically_correct_first_lanes*4===result.totals.episodes*3,'posterior-matched first route not 75%');return result;
}
if(require.main===module){chk(process.argv[2],'usage test_calibrated_gate_host_v79.js CORPUS [--quick]');console.log(JSON.stringify(audit(path.resolve(process.argv[2]),process.argv.includes('--quick')),null,2));}
module.exports={audit};
