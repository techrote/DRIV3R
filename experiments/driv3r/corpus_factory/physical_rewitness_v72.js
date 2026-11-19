#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v72 DATA ONLY: independently witnessed sensor
   re-calibration after correlated faults. Synthetic host != native CyberSand. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js'),parent=require('./cross_calibrated_receipt_v70.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(p,m)=>{if(!p)throw Error('V72 '+m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const COUNTS={TRAIN:4,DEV:2,HELDOUT:2};
const BASES={TRAIN:254000,DEV:264000,HELDOUT:274000};
const HISTORIES=[[],['A','A','A'],['B','B','B'],['A','B','B']];
const CALIBRATIONS=[{A:false,B:false},{A:true,B:false},{A:false,B:true},{A:true,B:true}];
const FAULTS=[{domain:'COMMON_BUS',flip:false},{domain:'COMMON_BUS',flip:true},{domain:'ISOLATED_SECONDARY',flip:false},{domain:'ISOLATED_SECONDARY',flip:true}];
const WITNESSES=['NONE','A','B','BOTH'];
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const streamName=s=>s==='TRAIN'?'TRAIN_physical_rewitness_stream.jsonl':s+'_EVALUATOR_ONLY_physical_rewitness_stream.jsonl';
const certName=s=>s+'_physical_rewitness_certificates_EVALUATOR_ONLY.jsonl';
function world(split,i,seen){for(let attempt=0;attempt<100;attempt++){
  const index=BASES[split]+i*163+attempt*3,w=host.scenario(split,index),h=host.start(w,'positive',false);
  const d=p=>host.dist(h.g,p,-1)[w.s];
  if(seen.has(w.geometry_sha256)||d(w.goal)<5||d(w.goal2)<5||parent.pathTo(h,w.goal)[0]===parent.pathTo(h,w.goal2)[0])continue;
  seen.add(w.geometry_sha256);return {w,index,attempt};
 }throw Error('v72 world not qualified '+split+'/'+i);}
function sense(h,pending){const z=parent.makeSense(h,pending);z.v='synthetic-v72-physical-rewitness';return z;}
function record(ctx,a,weight=0){const input=sense(ctx.h,ctx.queue);check(input.legal.includes(a),'illegal action '+a);const row={input,action:a,weight,focus_loss_weight:weight};ctx.rows.push(row);check(host.apply(ctx.h,a).accepted,'rejected '+a);}
function go(ctx,pos,weight){for(const a of parent.pathTo(ctx.h,pos))record(ctx,a,weight);}
function order(h,targets){if(targets.length<2)return targets;
 const w=h.w,d=(a,b)=>host.dist(h.g,b,-1)[a];return d(h.pos,w.goal)+d(w.goal,w.goal2)<=d(h.pos,w.goal2)+d(w.goal2,w.goal)?['A','B']:['B','A'];}
function finish(ctx,targets){for(const m of order(ctx.h,targets)){const p=m==='A'?ctx.h.w.goal:ctx.h.w.goal2;go(ctx,p,1);if(ctx.h.painted[p]!==m){check(ctx.h.painted[p]===undefined,'unexpected material');record(ctx,'PAINT_'+m,1);}}record(ctx,'STOP',1);}
function simulate(snapshot,targets){const x={h:cp(snapshot),queue:[],rows:[]};finish(x,targets);return {length:x.rows.length,success:x.h.success};}
function primaryCode(physical,calibration,flipped){const inverted=calibration!==flipped;return inverted?(physical==='INTACT'?'ERODED':'INTACT'):physical;}
function episode(w,threshold,sequence,calibration,fault,witness){
 const h=host.start(w,'positive',false),ctx={h,queue:[],rows:[],threshold,counts:{A:0,B:0},erasures:0};
 const emit=(...fields)=>ctx.queue.push([h.accepted,...fields]);
 record(ctx,'PAINT_A');if(threshold===1)delete h.painted[w.s];emit('LOCAL_PHYSICAL_COUPON',threshold===1?'ABRADED':'SURVIVED');
 const n=h.g.adj[h.pos][0][0],a=parent.pathTo(h,n)[0];record(ctx,a);delete h.painted[w.s];emit('COUPON_FIXTURE_RESET','A');record(ctx,reverse[a]);
 for(const m of ['A','B']){go(ctx,m==='A'?w.goal:w.goal2,0);record(ctx,'PAINT_'+m);emit('LOCAL_CHANNEL_CHECK_'+m,'PRIMARY',calibration[m]?'ERODED':'INTACT');}
 check(h.stage===2,'physical goals not set');go(ctx,w.s,0);
 const move=parent.pathTo(h,h.g.adj[h.pos][0][0])[0];
 const sample={},reports={A:[],B:[]},witnessed={},schedule=[];let faultAt=null;
 for(let k=0;k<112;k++){
  if([8,21,62].includes(k)){
   const m=sequence[[8,21,62].indexOf(k)]||null;schedule.push({clock:h.accepted,k,material:m});
   if(m){ctx.counts[m]++;const p=m==='A'?w.goal:w.goal2;
    if(ctx.counts[m]===threshold&&h.painted[p]===m){delete h.painted[p];ctx.erasures++;}}
   emit('UNVERIFIED_REMOTE_VIBRATION','UNKNOWN');
  }
  if(k===38||k===72){sample[k]={clock:h.accepted,read:{A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
  if(k===48){faultAt=h.accepted;emit('TELEMETRY_FAULT_DOMAIN',fault.domain);}
  if(k===74&&witness!=='NONE'){
   for(const m of witness==='BOTH'?['A','B']:[witness]){
    check(h.pos===w.s&&h.painted[w.s]===undefined,'witness at hub');
    // The local coupon is a real, host-accepted PAINT, followed by ERASE.
    record(ctx,'PAINT_'+m,0);
    check(h.painted[w.s]===m,'witness unpainted');
    const flipped=fault.domain==='COMMON_BUS'&&fault.flip;
    const code=primaryCode('INTACT',calibration[m],flipped);
    witnessed[m]={clock:h.accepted,code};emit('LOCAL_POSTFAULT_WITNESS_'+m,'PRIMARY',code);
    record(ctx,'ERASE',0);
    check(h.painted[w.s]===undefined,'witness residue');
   }
  }
  if(k===80||k===90){const src=k===80?72:38;
   for(const m of ['A','B']){
    const primaryFlip=src===72&&fault.domain==='COMMON_BUS'&&fault.flip;
    const secondaryFlip=src===72&&fault.flip;
    const p=primaryCode(sample[src].read[m],calibration[m],primaryFlip);
    const secondaryCal=fault.domain==='COMMON_BUS'?calibration[m]:!calibration[m];
    const s=primaryCode(sample[src].read[m],secondaryCal,secondaryFlip);
    for(const [ch,code] of [['PRIMARY',p],['SECONDARY',s]]){
     const row=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,ch,sample[src].clock,code];ctx.queue.push(row);reports[m].push(row);
    }
   }
  }
  record(ctx,k%2===0?move:reverse[move],0);
 }
 check(h.pos===w.s&&ctx.queue.length===0,'detour not reset');
 const critical=ctx.rows.length,snapshot=cp(h),criticalInput=sense(h,[]);
 const label={},targets=[],broken=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
 for(const m of ['A','B']){
  // Every fault is a trust-boundary event. COMMON_BUS requires a fresh,
  // *physical* local paint witness to decode post-fault PRIMARY receipts.
  // ISOLATED_SECONDARY leaves PRIMARY calibration unchanged.
  const afterTrust=fault.domain==='ISOLATED_SECONDARY'||!!witnessed[m];
  const eligible=reports[m].filter(r=>r[2]==='PRIMARY'&&(afterTrust||r[3]<faultAt)).sort((a,b)=>b[3]-a[3]);
  const last=eligible[0];let state='UNKNOWN';
  if(last){const inversion=last[3]<faultAt||fault.domain==='ISOLATED_SECONDARY'?calibration[m]:witnessed[m].code==='ERODED';
   const physical=inversion?(last[4]==='ERODED'?'INTACT':'ERODED'):last[4];
   state=physical==='ERODED'?'ERODED':schedule.some(s=>s.clock>last[3])?'UNKNOWN':'INTACT';}
  label[m]={state,measurement_clock:last?last[3]:null,physically_witnessed:!!witnessed[m]};
  if(state!=='INTACT')targets.push(m);
 }
 check(broken.every(m=>targets.includes(m)),'privileged state leakage would lose objective');
 finish(ctx,targets);check(h.done&&h.success,'teacher failed');
 const always=simulate(snapshot,['A','B']),oracle=simulate(snapshot,broken),stop=simulate(snapshot,[]);
 check(always.success&&oracle.success,'counterfactual baseline invalid');
 check(H(ctx.rows[critical].input)===H(criticalInput),'critical mismatch');
 return {episode:{steps:ctx.rows},certificate:{threshold,sequence,calibration,fault,witness,witnessed,
  critical_step:critical,critical_sha256:H(criticalInput),sampling_clocks:{early:sample[38].clock,late:sample[72].clock},fault_clock:faultAt,
  schedule,late_primary_raw:Object.fromEntries(['A','B'].map(m=>[m,reports[m].filter(r=>r[2]==='PRIMARY').sort((a,b)=>b[3]-a[3])[0][4]])),
  actual_broken:broken,teacher_labels:label,targets,teacher_first_action:ctx.rows[critical].action,
  teacher_actions:ctx.rows.length-critical,witness_actions:witness==='BOTH'?4:witness==='NONE'?0:2,
  always_actions:always.length,oracle_actions:oracle.length,
  stop_success:stop.success,physical_erasures:ctx.erasures,terminal_success:h.success,
  source_host:'synthetic-v44-plus-v67-erasure',native_CyberSand:false}};
}
function writer(p){const fd=fs.openSync(p,'wx'),hasher=crypto.createHash('sha256');let rows=0,bytes=0;return {put(value){const s=JSON.stringify(value)+'\n',b=Buffer.byteLength(s);fs.writeSync(fd,s);hasher.update(s);rows++;bytes+=b;},close(){fs.closeSync(fd);return {rows,bytes,sha256:hasher.digest('hex')}}};}
function generate(dir,{quick=false}={}){
 check(!fs.existsSync(dir),'destination exists');fs.mkdirSync(dir,{recursive:true});
 const mf={schema:'driv3r/corpus_factory/physical-rewitness-v72/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,quick,
  generator_sha256:H(fs.readFileSync(__filename)),host_sha256:H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),
  parent_v70_sha256:H(fs.readFileSync(path.join(__dirname,'cross_calibrated_receipt_v70.js'))),
  design:'Physically executed, material-selective hub witness after a possible common-bus fault; stale remote samples, invalidated calibration, physical erosion, uncertainty-driven conservative repairs. Matched full-observation critical decisions; sensor fault state hidden.',
  model_contract:'Only TRAIN_physical_rewitness_stream.jsonl trainable. Other streams and ALL certificates evaluator-only. Prefix and intervention actions masked; critical and distinct recovery actions supervised. No hidden calibration/fault/physical status in input.',
  caveats:['Entire host plus material erosion, sensor channels, and witness receipts are synthetic, not established CyberSand ABI.',
    'Detour and coupons scripted. Reinspection teacher symbolic; no trained SLM or closed-loop neural evidence.',
    'A witnessed PRIMARY local coupon is assumed sufficient to restore calibration for same material; no cross-material generalization.'],files:{},splits:{}};
 const seen=new Set(),seenPairs=new Set();let total=0,decisions=0,savings=0,netSavings=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const sw=writer(path.join(dir,streamName(split))),cw=writer(path.join(dir,certName(split)));
  const geometries=[];let d=0,ok=0,avoid=0,netAvoid=0,mask=0,distinct=0,changed=0,contrasts=0;
  for(let i=0;i<(quick?1:COUNTS[split]);i++){
   const {w,index,attempt}=world(split,i,seen);geometries.push({sha256:w.geometry_sha256,index,attempt,seed:w.seed});
   for(const threshold of [1,2])for(const seq of [HISTORIES[0],HISTORIES[1+(i%3)]]){
    const group=[];
    for(const calibration of CALIBRATIONS)for(const fault of FAULTS)for(const witness of WITNESSES){
     const {episode:e,certificate:c}=episode(w,threshold,seq,calibration,fault,witness);
     c.split=split;c.world_index=i;c.source_world_index=index;c.geometry_sha256=w.geometry_sha256;c.episode=ok;
     if(split==='TRAIN')for(let k=c.critical_step+1;k<e.steps.length;k++){
      const s=e.steps[k],sig=H([s.input,s.action]);if(seenPairs.has(sig)){s.weight=0;s.focus_loss_weight=0;mask++;}else seenPairs.add(sig);}
     check(e.steps[c.critical_step].weight===1,'masked critical');
     c.episode_sha256=H(e.steps.map(s=>[s.input,s.action,s.weight]));
     sw.put(e);cw.put(c);d+=e.steps.length;ok++;avoid+=c.always_actions-c.teacher_actions;netAvoid+=c.always_actions-c.teacher_actions-c.witness_actions;group.push(c);
     changed+=Number(c.fault.domain==='COMMON_BUS'&&c.fault.flip&&c.witness!=='NONE');
    }
    check(group.length===64,'64-arm completeness');
    check(new Set(group.map(c=>c.critical_sha256)).size===1,'identical instantaneous observation failed');
    check(new Set(group.map(c=>H(c.actual_broken))).size===1,'physical truth changed with observations');
    const choices=new Set(group.map(c=>c.teacher_first_action));distinct=Math.max(distinct,choices.size);
    contrasts+=Number(choices.size>=2);
   }
  }
  mf.files[streamName(split)]=sw.close();mf.files[certName(split)]=cw.close();
  mf.splits[split]={worlds:geometries.length,episodes:ok,decisions:d,terminal_successes:ok,counterfactual_64_arm_groups:geometries.length*4,
   groups_with_different_critical_actions:contrasts,maximum_distinct_critical_actions:distinct,
   common_fault_post_witness_cases:changed,train_repeated_followup_masked:mask,mean_gross_actions_saved_vs_always_reinspect:avoid/ok,mean_net_actions_saved_after_witness_cost:netAvoid/ok,
   geometry:geometries};total+=ok;decisions+=d;savings+=avoid;netSavings+=netAvoid;
 }
 mf.total_episodes=total;mf.total_decisions=decisions;mf.mean_gross_actions_saved_vs_always_reinspect=savings/total;mf.mean_net_actions_saved_after_witness_cost=netSavings/total;
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){check(process.argv[2],'usage node physical_rewitness_v72.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv.includes('--quick')}).splits,null,2));}
module.exports={episode,world,generate,streamName,certName,H,HISTORIES,CALIBRATIONS,FAULTS,WITNESSES};
