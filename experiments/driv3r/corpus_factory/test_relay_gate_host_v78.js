#!/usr/bin/env node
'use strict';
// Independent physical-host replay. Does not import the generator or its planner.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const digest=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const requireTrue=(b,m)=>{if(!b)throw Error('v78 independent AUDIT: '+m)};
const counts={TRAIN:12,DEV:6,HELDOUT:6},dirs=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
const cell=(x,y)=>String(x)+','+String(y);
function replayWorld(split,i){
 const width=26+({TRAIN:0,DEV:28,HELDOUT:42}[split])+2*i,mid=Math.floor(width*0.48);
 const stage1={TOP:3+(3*i+1)%(mid-6),BOTTOM:3+(5*i+2)%(mid-6)};
 const stage2={TOP:mid+3+(3*i+1)%(width-mid-6),BOTTOM:mid+3+(5*i+2)%(width-mid-6)};
 const walkable=new Set([cell(-1,1),cell(0,1),cell(mid,1),cell(width,1)]);
 for(let x=0;x<=width;x++){walkable.add(cell(x,0));walkable.add(cell(x,2));}
 for(let x=2;x<width-1;x+=3){const y=(Math.floor(x/3)+i)%2===0?-1:3;walkable.add(cell(x,y));}
 return {width,mid,stage1,stage2,walkable,geometryHash:digest({width,mid,stage1_gates:[[stage1.TOP,0],[stage1.BOTTOM,2]],stage2_gates:[[stage2.TOP,0],[stage2.BOTTOM,2]],cells:[...walkable].sort()})};
}
function barrier(g,x,y){for(const s of ['S1','S2'])for(const lane of ['TOP','BOTTOM'])if(y===(lane==='TOP'?0:2)&&g[s==='S1'?'stage1':'stage2'][lane]===x)return s+'_'+lane;return null;}
function permitted(h){const [x,y]=h.position,valid=[];for(const [cmd,dx,dy] of dirs){const xx=x+dx,yy=y+dy;if(!h.g.walkable.has(cell(xx,yy)))continue;const gate=barrier(h.g,xx,yy);if(gate&&!h.open[gate])continue;valid.push(cmd);}
 if(x===h.g.width&&y===1)valid.push(h.painted?'STOP':'PAINT_A');return valid;}
function visibility(h){const [x,y]=h.position,result={};for(const s of ['S1','S2'])for(const lane of ['TOP','BOTTOM']){
  const gate=h.g[s==='S1'?'stage1':'stage2'][lane];result[s+'_'+lane]=(y===(lane==='TOP'?0:2)&&Math.abs(x-gate)<=1)?h.open[s+'_'+lane]:null;
 }return result;}
function publicObservation(h,receipts){return {schema:'driv3r/synthetic-two-relay-host/1',tick:h.clock,position:h.position.slice(),goal:[h.g.width,1],midpoint:[h.g.mid,1],legal:permitted(h),local_gate_visible:visibility(h),reports:receipts};}
function rStates(section,at,h){return ['TOP','BOTTOM'].map(lane=>({type:'REMOTE_GATE_REPORT',section,lane,measure_tick:at,status:h.open[section+'_'+lane]?'OPEN':'CLOSED'}));}
function auditEpisode(ep,c,split,index){
 const g=replayWorld(split,c.world_index),h={g,position:[0,1],clock:0,open:{S1_TOP:true,S1_BOTTOM:true,S2_TOP:true,S2_BOTTOM:true},painted:false,done:false,midEntered:false};
 requireTrue(c.schema==='driv3r/corpus_factory/relay-gate-v78-cert/1'&&c.split===split&&c.episode===index,'cert identity');
 requireTrue(c.geometry_sha256===g.geometryHash,'wrong geometry');
 requireTrue(['TOP','BOTTOM'].includes(c.stage1_open)&&['TOP','BOTTOM'].includes(c.stage2_reported_open),'arms');
 requireTrue(['FRESH_FIRST','STALE_FIRST'].includes(c.receipt_order)&&typeof c.midpoint_flip==='boolean'&&typeof c.offpolicy_stage1==='boolean','arms 2');
 requireTrue(c.first_decision_step===200&&ep.steps.length===c.total_actions&&digest(ep.steps)===c.episode_sha256,'episode metadata');
 requireTrue(c.first_action_label===(c.stage1_open==='TOP'?'MOVE_N':'MOVE_S')&&c.second_action_label===(c.stage2_reported_open==='TOP'?'MOVE_N':'MOVE_S'),'target labels');
 const events=[];let init1,init2,latest1,latest2;
 const firstHist=[],secondHist=[];let didWrongBlock=false,didSurpriseBlock=false,positive=0;
 for(let j=0;j<ep.steps.length;j++){
  if(j===10){h.open.S1_TOP=c.stage1_open==='TOP';h.open.S1_BOTTOM=c.stage1_open==='BOTTOM';latest1=rStates('S1',h.clock,h);events.push({type:'PHYSICAL_GATE_CHANGE',section:'S1',tick:h.clock,gate:{TOP:h.open.S1_TOP,BOTTOM:h.open.S1_BOTTOM}});}
  if(j===50){h.open.S2_TOP=c.stage2_reported_open==='TOP';h.open.S2_BOTTOM=c.stage2_reported_open==='BOTTOM';latest2=rStates('S2',h.clock,h);events.push({type:'PHYSICAL_GATE_CHANGE',section:'S2',tick:h.clock,gate:{TOP:h.open.S2_TOP,BOTTOM:h.open.S2_BOTTOM}});}
  if(j===0){init1=rStates('S1',0,h);init2=rStates('S2',0,h);}
  let receipts=[];for(const [section,early,late,init,fresh] of [['S1',12,22,init1,latest1],['S2',52,62,init2,latest2]]){
   if(j===early||j===late){const isFresh=c.receipt_order==='FRESH_FIRST'?(j===early):(j===late);
    receipts=(isFresh?fresh:init).map(x=>({...x}));events.push({type:'REPORT_DELIVERED',section,delivered_tick:h.clock,measure_tick:isFresh?(section==='S1'?10:50):0});
   }
  }
  const observed=publicObservation(h,receipts),row=ep.steps[j];
  requireTrue(JSON.stringify(observed)===JSON.stringify(row.input),`${split}/${index}/${j} observable mismatch`);
  requireTrue(observed.legal.includes(row.action),`${split}/${index}/${j} illegal action ${row.action}`);
  requireTrue((row.weight===0||row.weight===1)&&row.focus_loss_weight===row.weight,'weight mismatch');
  if(j<200)requireTrue(row.weight===0,'masked prelude');else if(row.weight)positive++;
  if(j===200){requireTrue(digest(observed)===c.first_decision_input_sha256,'first input hash');requireTrue(row.action===(c.offpolicy_stage1?(c.first_action_label==='MOVE_N'?'MOVE_S':'MOVE_N'):c.first_action_label),'first action');requireTrue(row.weight===(c.offpolicy_stage1?0:1),'first weight');}
  if(j===c.second_decision_step){requireTrue(digest(observed)===c.second_decision_input_sha256,'second input hash');requireTrue(row.action===c.second_action_label&&row.weight===1,'second action');requireTrue(h.position[0]===g.mid&&h.position[1]===1,'second decision junction');}
  firstHist.push(...receipts.filter(r=>r.section==='S1'));secondHist.push(...receipts.filter(r=>r.section==='S2'));
  const d=dirs.find(x=>x[0]===row.action);
  if(d){h.position[0]+=d[1];h.position[1]+=d[2];}else if(row.action==='PAINT_A'){requireTrue(!h.painted,'double paint');h.painted=true;}
  else if(row.action==='STOP'){requireTrue(h.painted&&h.position[0]===g.width&&h.position[1]===1,'bad STOP');h.done=true;}
  else throw Error('unknown action');
  h.clock++;
  if(h.position[0]===g.mid&&h.position[1]===1&&!h.midEntered){h.midEntered=true;
   events.push({type:'MIDPOINT_ENTERED',tick:h.clock});
   if(c.midpoint_flip){h.open.S2_TOP=!h.open.S2_TOP;h.open.S2_BOTTOM=!h.open.S2_BOTTOM;
    events.push({type:'UNREPORTED_PHYSICAL_GATE_CHANGE',section:'S2',tick:h.clock,gate:{TOP:h.open.S2_TOP,BOTTOM:h.open.S2_BOTTOM}});}
  }
  if(j===c.wrong_recovery_step){requireTrue(row.action==='MOVE_W'&&row.weight===1,'wrong recovery command');didWrongBlock=true;
   requireTrue(row.input.legal.includes('MOVE_W')&&!row.input.legal.includes('MOVE_E'),'wrong route was not physically blocked');}
  if(j===c.surprise_recovery_step){requireTrue(row.action==='MOVE_W'&&row.weight===1,'surprise recovery');didSurpriseBlock=true;
   requireTrue(row.input.legal.includes('MOVE_W')&&!row.input.legal.includes('MOVE_E'),'surprise not physically blocked');}
 }
 requireTrue(h.done&&h.painted&&h.midEntered,'not complete');
 requireTrue(firstHist.length===4&&secondHist.length===4,'receipt cardinality');
 for(const [section,records,freshTime,target] of [['S1',firstHist,10,c.stage1_open],['S2',secondHist,50,c.stage2_reported_open]]){
  const recent={};for(const r of records)if(!recent[r.lane]||r.measure_tick>recent[r.lane].measure_tick)recent[r.lane]=r;
  requireTrue(recent.TOP.measure_tick===freshTime&&recent.BOTTOM.measure_tick===freshTime,'missing causal receipt '+section);
  requireTrue(recent[target].status==='OPEN'&&recent[target==='TOP'?'BOTTOM':'TOP'].status==='CLOSED','physical sensor wrong '+section);
 }
 requireTrue(c.first_decision_step===200&&c.second_decision_step>200&&c.second_decision_step<ep.steps.length-2,'decision coordinates');
 requireTrue(didWrongBlock===c.offpolicy_stage1&&didSurpriseBlock===c.midpoint_flip,'recovery cardinality');
 requireTrue(digest(events)===digest(c.events),'physical/report event log disagrees');
 requireTrue(positive>0&&c.terminal_success&&c.synthetic_not_native_CyberSand,'completion cert');
 return ep.steps.length;
}
function load(file,record,expectedRows){const raw=fs.readFileSync(file,'utf8');requireTrue(Buffer.byteLength(raw)===record.bytes&&digest(raw)===record.sha256,'shard byte/digest mismatch '+file);
 const lines=raw.trimEnd().split('\n');requireTrue(lines.length===expectedRows&&lines.length===record.rows,'shard row count '+file);return lines.map(JSON.parse);}
function run(dir,quick=false){const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 requireTrue(manifest.schema==='driv3r/corpus_factory/relay-gate-replanning-v78/1'&&manifest.synthetic_not_native_CyberSand,'manifest schema');
 let total=0,episodes=0,worlds=0,wrong=0,surprises=0;const seen=new Set();
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const n=quick?1:counts[split],prefix=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';
  const streamfile=prefix+'_relay_stream.jsonl',certfile=prefix+'_relay_certificates_EVALUATOR_ONLY.jsonl';
  const rows=load(path.join(dir,streamfile),manifest.files[streamfile],32*n),certs=load(path.join(dir,certfile),manifest.files[certfile],32*n);
  let subtotal=0,w=0,s=0;const stat=manifest.splits[split];requireTrue(stat.worlds.length===n,'world count');
  for(let i=0;i<certs.length;i++){
   const c=certs[i],arm=i%32;
   requireTrue(c.world_index===Math.floor(i/32),'world episode grouping');
   const b1=['TOP','BOTTOM'][Math.floor(arm/16)],b2=['TOP','BOTTOM'][Math.floor((arm%16)/8)],order=['FRESH_FIRST','STALE_FIRST'][Math.floor((arm%8)/4)];
   const flip=(arm%4)>=2,off=arm%2===1;
   requireTrue(c.stage1_open===b1&&c.stage2_reported_open===b2&&c.receipt_order===order&&c.midpoint_flip===flip&&c.offpolicy_stage1===off,'counterfactual arm order');
   subtotal+=auditEpisode(rows[i],c,split,i);if(off)w++;if(flip)s++;
  }
  for(let k=0;k<n;k++){
   const slice=certs.slice(k*32,(k+1)*32),g=replayWorld(split,k);
   requireTrue(stat.worlds[k].geometry_sha256===g.geometryHash,'manifest geometry');
   requireTrue(new Set(slice.map(c=>c.first_decision_input_sha256)).size===1,'current first observation differs');
   requireTrue(new Set(slice.map(c=>c.first_action_label)).size===2,'no first memory discrimination');
   const second=new Map();for(const c of slice){const id=c.second_decision_input_sha256;if(!second.has(id))second.set(id,new Set());second.get(id).add(c.second_action_label);}
   requireTrue([...second.values()].some(x=>x.size===2),'no second memory discrimination');
   requireTrue(!seen.has(g.geometryHash),'split/near duplicate geometry');seen.add(g.geometryHash);
  }
  requireTrue(stat.episodes===32*n&&stat.terminal_successes===32*n&&stat.legal_actions===subtotal,'aggregate count false');
  requireTrue(stat.offpolicy_wrong_corridor_recoveries===w&&stat.unannounced_gate_replans===s,'aggregate recovery false');
  requireTrue(stat.first_choice_memory_conflicts===n&&stat.second_choice_memory_conflicts===n,'memory contrast count');
  episodes+=rows.length;total+=subtotal;wrong+=w;surprises+=s;worlds+=n;
 }
 return {schema:'driv3r/corpus_factory/relay-v78-independent-audit/1',worlds,episodes,legal_actions:total,
  physical_wrong_corridor_recoveries:wrong,unreported_gate_replans:surprises,
  first_and_second_memory_discriminative_worlds:worlds,all_episodes_terminal:true,model_trained:false,synthetic_not_native_CyberSand:true};
}
if(require.main===module){requireTrue(process.argv[2],'usage node test_relay_gate_host_v78.js DIR [--quick]');console.log(JSON.stringify(run(path.resolve(process.argv[2]),process.argv.includes('--quick')),null,2));}
module.exports={run};
