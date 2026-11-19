#!/usr/bin/env node
'use strict';
// Independent replay, not a trained neural policy. Reconstruct actual synthetic
// physical movements, remote gate states, observable reports and final paint.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
function ok(p,m){if(!p)throw Error('v77 AUDIT '+m);}
const MOVES=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const sizes={TRAIN:12,DEV:6,HELDOUT:6};
function start(split,i){const width=10+({TRAIN:0,DEV:12,HELDOUT:18}[split])+i;
 const top=3+(i*3)%Math.max(1,width-5),bottom=3+(i*7+1)%Math.max(1,width-5);
 const positions=new Set(['0,1','-1,1',width+',1']);
 for(let x=0;x<=width;x++)for(const y of [0,2])positions.add(x+','+y);
 const h={width,top,bottom,positions,pos:[0,1],gate:{TOP:true,BOTTOM:true},clock:0,painted:false,done:false};
 const geometry={width,top_gate:[top,0],bottom_gate:[bottom,2],cells:[...positions].sort()};
 return {h,geometryHash:hash(geometry)};}
function current(h,reports){const [x,y]=h.pos,legal=[];
 for(const [cmd,dx,dy] of MOVES){const xx=x+dx,yy=y+dy;
  if(!h.positions.has(xx+','+yy))continue;
  if((xx===h.top&&yy===0&&!h.gate.TOP)||(xx===h.bottom&&yy===2&&!h.gate.BOTTOM))continue;
  legal.push(cmd);
 }
 if(x===h.width&&y===1)legal.push(h.painted?'STOP':'PAINT_A');
 return {schema:'driv3r/synthetic-gate-host/1',tick:h.clock,position:[x,y],goal:[h.width,1],map_width:h.width,legal,
  local_gate_visible:{TOP:y===0&&Math.abs(x-h.top)<=1?h.gate.TOP:null,
   BOTTOM:y===2&&Math.abs(x-h.bottom)<=1?h.gate.BOTTOM:null},reports};}
function advance(h,cmd){const mv=MOVES.find(x=>x[0]===cmd);
 if(mv){h.pos[0]+=mv[1];h.pos[1]+=mv[2];}
 else if(cmd==='PAINT_A')h.painted=true;
 else if(cmd==='STOP')h.done=true;
 else throw Error('bad action');h.clock++;
}
function verifyEpisode(c,ep,split,index){
 const {h,geometryHash}=start(split,c.world_index);
 ok(c.schema==='driv3r/corpus_factory/dynamic-gate-v77-cert/1','certificate schema');
 ok(c.split===split&&c.episode===index&&c.geometry_sha256===geometryHash,'source geometry identity');
 ok(c.open_lane==='TOP'||c.open_lane==='BOTTOM','open lane');
 ok(['FRESH_FIRST','STALE_FIRST'].includes(c.receipt_order),'receipt order');
 ok(typeof c.off_policy==='boolean','offpolicy');
 const initial={measure_tick:0,TOP:'OPEN',BOTTOM:'OPEN'},target=c.open_lane==='TOP'?'MOVE_N':'MOVE_S';
 let updated=null,nonzero=0;const reportsSeen=[];
 ok(c.decision_step===160,'critical timing');
 ok(ep.steps.length===c.total_actions&&hash(ep.steps)===c.episode_sha256,'source episode hash');
 for(let j=0;j<ep.steps.length;j++){
  if(j===10){h.gate.TOP=c.open_lane==='TOP';h.gate.BOTTOM=c.open_lane==='BOTTOM';
   updated={measure_tick:10,TOP:h.gate.TOP?'OPEN':'CLOSED',BOTTOM:h.gate.BOTTOM?'OPEN':'CLOSED'};}
  let reports=[];
  if(j===12||j===22){const choose=(c.receipt_order==='FRESH_FIRST'?(j===12?updated:initial):(j===12?initial:updated));
   reports=['TOP','BOTTOM'].map(lane=>({type:'REMOTE_GATE_REPORT',lane,measure_tick:choose.measure_tick,status:choose[lane]}));
  }
  const expected=current(h,reports),row=ep.steps[j];
  ok(JSON.stringify(expected)===JSON.stringify(row.input),'observation/clock/source-report mismatch at '+split+'/'+index+'/'+j);
  ok(expected.legal.includes(row.action),'host rejected action at '+index+'/'+j+' '+row.action);
  ok((row.weight===1||row.weight===0)&&row.focus_loss_weight===row.weight,'weight');
  if(j<160)ok(row.weight===0,'prelude supervision leakage');
  if(row.weight)nonzero++;
  reportsSeen.push(...reports);advance(h,row.action);
 }
 ok(h.done&&h.painted&&h.pos[0]===h.width&&h.pos[1]===1,'paint/stop completion');
 ok(ep.steps[160].input.local_gate_visible.TOP===null&&ep.steps[160].input.local_gate_visible.BOTTOM===null,'current leak');
 ok(ep.steps[160].input.reports.length===0&&hash(ep.steps[160].input)===c.current_input_sha256,'critical hash');
 ok(c.first_optimal_action===target,'teacher route mismatch');
 if(c.off_policy){
  ok(ep.steps[160].action!=='PAINT_A'&&ep.steps[160].action!=='STOP','invalid wrong arm');
  ok(ep.steps[160].action===(target==='MOVE_N'?'MOVE_S':'MOVE_N')&&ep.steps[160].weight===0,'offpolicy wrong unmasked');
  ok(c.recovery_step>160&&ep.steps[c.recovery_step].weight===1,'recovery evidence');
  const wall=ep.steps[c.recovery_step].input;
  ok(!wall.legal.includes('MOVE_E'),'recovery did not meet physical gate');
 }else{
  ok(ep.steps[160].action===target&&ep.steps[160].weight===1&&c.recovery_step===-1,'teacher causal first decision');
 }
 ok(nonzero>0&&c.terminal_success===true&&c.synthetic_not_native_CyberSand===true,'terminal identity');
 ok(c.environmental_events.length===1&&c.environmental_events[0].tick===10&&
    c.environmental_events[0].gates.TOP===h.gate.TOP&&c.environmental_events[0].gates.BOTTOM===h.gate.BOTTOM,'physical gate event');
 ok(hash(reportsSeen)===c.history_report_sha256,'source history checksum');
 return {actions:ep.steps.length,criticalHash:c.current_input_sha256,label:target};
}
function audit(dir,quick=false){const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 ok(manifest.schema==='driv3r/corpus_factory/dynamic-gate-memory-v77/1'&&manifest.synthetic_not_native_CyberSand,'manifest/schema');
 const geometries=new Set();let totalActions=0,episodes=0,contrasts=0,wrong=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const sn=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY',stream=sn+'_gate_memory_stream.jsonl',cert=sn+'_gate_memory_certificates_EVALUATOR_ONLY.jsonl';
  for(const f of [stream,cert]){const text=fs.readFileSync(path.join(dir,f),'utf8');
   ok(hash(text)===manifest.files[f].sha256,'digest mismatch '+f);
   ok(text.split('\n').filter(Boolean).length===manifest.files[f].rows,'line count '+f);}
  const rows=fs.readFileSync(path.join(dir,stream),'utf8').trimEnd().split('\n').map(JSON.parse);
  const certificates=fs.readFileSync(path.join(dir,cert),'utf8').trimEnd().split('\n').map(JSON.parse);
  const expectedWorlds=quick?1:sizes[split],stat=manifest.splits[split];
  ok(certificates.length===rows.length&&certificates.length===expectedWorlds*8,'episode cardinality');
  let acts=0;for(let i=0;i<certificates.length;i++){
   const c=certificates[i],e=rows[i],n=c.world_index;
   ok(n===Math.floor(i/8),'world grouping');
   const arm=i%8,lanes=['TOP','BOTTOM'],orders=['FRESH_FIRST','STALE_FIRST'];
   ok(c.open_lane===lanes[Math.floor(arm/4)]&&c.receipt_order===orders[Math.floor((arm%4)/2)]&&c.off_policy===(arm%2===1),'causal arm completeness');
   const replay=verifyEpisode(c,e,split,i);acts+=replay.actions;if(c.off_policy)wrong++;
  }
  for(let i=0;i<expectedWorlds;i++){
   const matched=certificates.slice(i*8,(i+1)*8),hashes=new Set(matched.map(x=>x.current_input_sha256));
   ok(hashes.size===1&&new Set(matched.map(x=>x.first_optimal_action)).size===2,'unambiguous current-only label');
   const geo=matched[0].geometry_sha256;ok(!geometries.has(geo),'cross-split geometry leakage');geometries.add(geo);contrasts++;
  }
  ok(stat.worlds.length===expectedWorlds&&stat.episodes===rows.length&&stat.legal_actions===acts&&stat.terminal_successes===rows.length,'aggregate tampering');
  ok(stat.paired_current_observation_groups===expectedWorlds&&stat.paired_conflicting_correct_actions===expectedWorlds&&stat.offpolicy_recoveries===expectedWorlds*4,'paired aggregate tampering');
  totalActions+=acts;episodes+=rows.length;
 }
 return {worlds:geometries.size,episodes,legal_actions:totalActions,paired_conflicting_worlds:contrasts,offpolicy_recoveries:wrong,
  theoretical_current_only_first_action_ceiling:0.5,full_history_timestamp_oracle_first_action_accuracy:1,
  model_performance_measured:false,synthetic_not_native_CyberSand:true};
}
if(require.main===module){ok(process.argv[2],'usage node test_dynamic_gate_host_v77.js DIR [--quick]');console.log(JSON.stringify(audit(path.resolve(process.argv[2]),process.argv.includes('--quick')),null,2));}
module.exports={audit};
