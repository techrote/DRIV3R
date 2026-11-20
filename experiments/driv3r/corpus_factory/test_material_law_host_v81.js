#!/usr/bin/env node
'use strict';
// Independent physical replay: does NOT import the generator or its planner.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const dig=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const moves={MOVE_N:[0,-1],MOVE_E:[1,0],MOVE_S:[0,1],MOVE_W:[-1,0]},lanes={TOP:0,CENTER:2,BOTTOM:4};
function audit(root){const m=JSON.parse(fs.readFileSync(path.join(root,'manifest.json')));assert.equal(m.schema,'driv3r/corpus_factory/material-law-memory-v81/1');
 const report={schema:'driv3r/v81-independent-replay/1',splits:{},total_episodes:0,total_actions:0,world_hashes:0,three_way_contrasts:0,goal_contrasts:0,law_contrasts:0,physical_recoveries:0,tamper_test_executed_separately:true};
 const hashes=new Set();for(const split of ['TRAIN','DEV','HELDOUT']){
 const pre=split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY';const sname=pre+'_material_law_stream.jsonl',cname=pre+'_material_law_certificates_EVALUATOR_ONLY.jsonl';
 const arr={};for(const name of [sname,cname]){const body=fs.readFileSync(path.join(root,name),'utf8');assert.equal(dig(body),m.files[name].sha256);arr[name]=body.trim().split('\n').map(JSON.parse);assert.equal(arr[name].length,m.files[name].rows);}
 const streams=arr[sname],certs=arr[cname];assert.equal(streams.length,certs.length);
 const wmap=new Map();for(const w of m.splits[split].worlds){assert.ok(!hashes.has(w.geometry_sha256),'split geometry leak');hashes.add(w.geometry_sha256);wmap.set(w.index,w);}
 let accepted=0,success=0,recoveries=0,wrong=0;const group=new Map(),ggroup=new Map(),lawgroups=new Map();
 for(let k=0;k<streams.length;k++){
  const steps=streams[k].steps,c=certs[k],w=wmap.get(c.world_index);assert.ok(w);assert.equal(c.episode,k);assert.equal(steps.length,c.steps);assert.equal(dig(steps),c.episode_sha256);
  assert.equal(c.geometry_sha256,w.geometry_sha256);assert.equal(c.synthetic_not_native_CyberSand,true);
  const cells=new Set(['-1,2']);for(let x=0;x<=w.width;x++)for(const y of [0,2,4])cells.add(x+','+y);
  for(const x of w.bridges)for(let y=0;y<=4;y++)cells.add(x+','+y);
  let loc=[0,2],tick=0,material=null,goal='A',law='NORMAL',flipped=false,terminal=false,gate={S1:'CENTER',S2:'CENTER'},inbox=[],discovery=0;
  const at=(x,y)=>x+','+y;const legal=()=>{const v=[];for(const [cmd,[dx,dy]] of Object.entries(moves)){
   const x=loc[0]+dx,y=loc[1]+dy;if(!cells.has(at(x,y)))continue;
   if(x===w.g1 && Object.values(lanes).includes(y) && gate.S1!==Object.keys(lanes).find(l=>lanes[l]===y))continue;
   if(x===w.g2 && Object.values(lanes).includes(y) && gate.S2!==Object.keys(lanes).find(l=>lanes[l]===y))continue;
   v.push(cmd);}
   if(at(...loc)===at(w.width,2)||at(...loc)==='0,2'){if(material===null)v.push('PAINT_A','PAINT_B');else{v.push('ERASE');if(at(...loc)===at(w.width,2)&&material===goal)v.push('STOP');}}
   return v;};
  const observed=()=>{const visibility={};for(const s of ['S1','S2'])for(const [lane,y] of Object.entries(lanes)){
   const x=s==='S1'?w.g1:w.g2;visibility[s+'_'+lane]=loc[0]>=x-1&&loc[0]<=x&&loc[1]===y?(gate[s]===lane?'OPEN':'BLOCKED'):null;}
   return {schema:'driv3r/synthetic-three-lane-law-host/1',tick,position:loc.slice(),goal_site:[w.width,2],legal:legal(),local_gate_visible:visibility,painted_material:material,reports:inbox.map(r=>({...r}))};};
  let eventpos=0;for(let n=0;n<steps.length;n++){
   while(eventpos<c.events.length && c.events[eventpos].at_step===n){const e=c.events[eventpos++];
    if(e.type==='INITIAL_LAW_SET'){assert.equal(n,0);assert.ok(['NORMAL','INVERTED'].includes(e.law));law=e.law;assert.equal(law,c.initial_law);}
    else if(e.type==='LAW_EVENT'){assert.equal(n,54);assert.equal(e.flipped,c.law_flip);flipped=e.flipped;if(flipped)law=law==='NORMAL'?'INVERTED':'NORMAL';}
    else if(e.type==='GATE_SET'||e.type==='UNREPORTED_GATE_SET'){assert.equal(e.section,'S1'===e.section?'S1':'S2');assert.ok(e.open in lanes);gate[e.section]=e.open;}
    else if(e.type==='GOAL_SET'){assert.ok(['A','B'].includes(e.material));goal=e.material;}
    else if(e.type==='REPORT_RECEIVED'){
     assert.ok([64,94].includes(n));inbox=steps[n].input.reports;assert.equal(inbox.length,4);const isFresh=c.report_order==='FRESH_FIRST'?(n===64):(n===94);
     assert.deepEqual(inbox.map(x=>x.sample_tick),isFresh?[24,44,48,54]:[4,4,4,4]);
     assert.deepEqual(inbox.map(x=>x.type),['REMOTE_GATE','REMOTE_GATE','REMOTE_OBJECTIVE','REMOTE_LAW_EVENT']);
     assert.equal(inbox[0].open,isFresh?c.stage1_open:'CENTER');assert.equal(inbox[1].open,isFresh?c.stage2_reported_open:'CENTER');
     assert.equal(inbox[2].material,isFresh?c.goal_material:'A');
     assert.equal(inbox[3].flipped,isFresh?c.law_flip:false);
    }else if(e.type==='PHYSICAL_OBSTRUCTION_DISCOVERED'){
     assert.equal(loc[0],w.g2-1);assert.equal(legal().includes('MOVE_E'),false);
     assert.deepEqual(e.at,loc);discovery++;
    }else assert.fail('unknown event '+e.type);
   }
   const row=steps[n];assert.deepEqual(row.input,observed(),`observation mismatch ep=${k} step=${n}`);
   assert.equal(row.input.tick,n);assert.ok(legal().includes(row.action),'illegal executed action '+row.action);
   assert.ok(!Object.keys(row.input).some(z=>/goal_material|hidden|physical_open|certificate|oracle/.test(z)),'future leaked');
   if(row.action in moves){loc=[loc[0]+moves[row.action][0],loc[1]+moves[row.action][1]];}
   else if(row.action.startsWith('PAINT_'))material=law==='NORMAL'?row.action.slice(-1):(row.action.endsWith('A')?'B':'A');
   else if(row.action==='ERASE')material=null;
   else if(row.action==='STOP')terminal=true;
   tick++;inbox=[];accepted++;
  }
  assert.equal(eventpos,c.events.length);assert.equal(discovery,Number(c.physical_obstruction_discovered));
  assert.equal(terminal,true);assert.equal(material,goal);assert.equal(goal,c.goal_material);assert.deepEqual(loc,[w.width,2]);success++;
  assert.equal(law,c.final_law);assert.equal(flipped,c.law_flip);
  assert.deepEqual(steps.slice(0,4).map(s=>s.action),['PAINT_A','ERASE','PAINT_B','ERASE']);
  assert.ok(steps.slice(0,4).every(s=>s.loss_weight===0));
  assert.equal(steps[1].input.painted_material,c.initial_law==='NORMAL'?'A':'B');
  assert.equal(steps[3].input.painted_material,c.initial_law==='NORMAL'?'B':'A');
  assert.equal(c.first_action,({TOP:'MOVE_N',CENTER:'MOVE_E',BOTTOM:'MOVE_S'})[c.stage1_open]);
  assert.equal(c.second_action,({TOP:'MOVE_N',CENTER:'MOVE_E',BOTTOM:'MOVE_S'})[c.stage2_reported_open]);
  const predictedPaint=law==='NORMAL'?goal:(goal==='A'?'B':'A');assert.equal(c.goal_action,'PAINT_'+predictedPaint);
  assert.equal(steps[c.first_step].action,c.first_action);assert.equal(steps[c.second_step].action,c.second_action);
  assert.equal(dig(steps[c.first_step].input),c.first_observation_sha256);
  assert.equal(dig(steps[c.second_step].input),c.second_observation_sha256);
  assert.equal(dig(steps[c.goal_step].input),c.goal_observation_sha256);
  assert.deepEqual(steps[c.first_step].input.local_gate_visible,{S1_TOP:null,S1_CENTER:null,S1_BOTTOM:null,S2_TOP:null,S2_CENTER:null,S2_BOTTOM:null});
  assert.equal(steps[c.first_step].input.reports.length,0);assert.equal(steps[c.second_step].input.reports.length,0);
  assert.ok(steps[c.first_step].input.tick-94>128,'history not long enough');assert.ok(steps[c.goal_step].input.tick-94>128);
  if(c.hidden_stage2_shift){assert.equal(steps[c.second_step].loss_weight,0);}else{assert.equal(steps[c.second_step].loss_weight,1);assert.equal(steps[c.first_step].loss_weight,1);}
  if(c.wrong_paint_recovered){wrong++;assert.equal(steps[c.goal_step].loss_weight,0);assert.equal(steps[c.goal_step+1].action,'ERASE');}else assert.equal(steps[c.goal_step].action,c.goal_action);
  if(c.physical_obstruction_discovered){recoveries++;assert.equal(c.hidden_stage2_shift,true);}
  const key=c.world_index+'/'+c.stage2_reported_open+'/'+c.goal_material+'/'+c.report_order+'/'+c.hidden_stage2_shift+'/'+c.initial_law+'/'+c.law_flip;
  if(!group.has(key))group.set(key,{inputs:new Set(),labels:new Set()});group.get(key).inputs.add(c.first_observation_sha256);group.get(key).labels.add(c.first_action);
  const key2=c.world_index+'/'+c.stage1_open+'/'+c.stage2_reported_open+'/'+c.report_order+'/'+c.hidden_stage2_shift+'/'+c.initial_law+'/'+c.law_flip;
  if(!ggroup.has(key2))ggroup.set(key2,new Set());ggroup.get(key2).add(c.goal_action);
  const lawkey=c.world_index+'/'+c.stage1_open+'/'+c.goal_material+'/'+c.report_order+'/'+c.hidden_stage2_shift;
  if(!lawgroups.has(lawkey))lawgroups.set(lawkey,{inputs:new Set(),labels:new Set(),pairs:new Map()});const lg=lawgroups.get(lawkey);lg.inputs.add(c.goal_observation_sha256);lg.labels.add(c.goal_action);lg.pairs.set(c.initial_law+'/'+c.law_flip,c.goal_action);
 }
 for(const v of group.values()){assert.equal(v.inputs.size,1);assert.equal(v.labels.size,3);report.three_way_contrasts++;}
 for(const v of ggroup.values()){assert.equal(v.size,2);report.goal_contrasts++;}
 for(const v of lawgroups.values()){assert.equal(v.inputs.size,1);assert.equal(v.labels.size,2);assert.equal(v.pairs.size,4);report.law_contrasts++;}
 assert.equal(success,m.splits[split].success);assert.equal(accepted,m.splits[split].legal_actions);
 assert.equal(recoveries,m.splits[split].physical_recoveries);assert.equal(wrong,m.splits[split].wrong_paint_recoveries);
 report.splits[split]={worlds:m.splits[split].worlds.length,episodes:success,legal_actions:accepted,physical_recoveries:recoveries,wrong_paint_recoveries:wrong};
 report.total_actions+=accepted;report.total_episodes+=success;report.physical_recoveries+=recoveries;
 }
 report.world_hashes=hashes.size;return report;
}
if(require.main===module){assert.ok(process.argv[2],'usage node test_material_law_host_v81.js DATASET_DIR');console.log(JSON.stringify(audit(path.resolve(process.argv[2])),null,2));}
module.exports={audit};
