#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v71: DATA ONLY. Hypothetical sensor channels and
 * erosion are synthetic v44-host extensions, NOT native CyberSand ABI. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js'),parent=require('./cross_calibrated_receipt_v70.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const chk=(p,m)=>{if(!p)throw Error('V71 '+m)};
const copy=x=>JSON.parse(JSON.stringify(x));
const COUNTS={TRAIN:4,DEV:2,HELDOUT:2};
const WORLD_BASE={TRAIN:214000,DEV:224000,HELDOUT:234000};
const HISTORIES=[[],['A','A','A'],['B','B','B'],['A','B','B']];
const CALIBRATIONS=[{A:false,B:false},{A:true,B:true}];
const DOMAINS=['NO_FAULT','ISOLATED_SECONDARY','COMMON_BUS'];
const MODES=['MISSING','PAIRED'];
const reversed={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
function makeSense(h,pending){const input=parent.makeSense(h,pending);input.v='synthetic-v71-common-mode-fault';return input;}
function record(ctx,action,weight){const input=makeSense(ctx.h,ctx.pending);chk(input.legal.includes(action),'illegal '+action);ctx.steps.push({input,action,weight,focus_loss_weight:weight});chk(host.apply(ctx.h,action).accepted,'host rejected '+action);}
function visit(ctx,to,weight){for(const a of parent.pathTo(ctx.h,to))record(ctx,a,weight);}
function continuation(ctx,targets){const w=ctx.h.w,t=m=>m==='A'?w.goal:w.goal2;
 const paths=targets.length===2?[['A','B'],['B','A']]:[targets];
 const d=(a,b)=>host.dist(ctx.h.g,b,-1)[a];
 const order=paths.map(seq=>({seq,cost:seq.reduce((cost,m,i)=>cost+d(i?t(seq[i-1]):ctx.h.pos,t(m)),0)})).sort((a,b)=>a.cost-b.cost||a.seq.join().localeCompare(b.seq.join()))[0].seq;
 for(const m of order){const p=t(m);visit(ctx,p,1);const present=host.input(ctx.h).painted.find(([i])=>i===p);if(!present)record(ctx,'PAINT_'+m,1);else chk(present[1]===m,'unexpected paint');}
 record(ctx,'STOP',1);
}
function estimate(snapshot,targets){const ctx={h:copy(snapshot),pending:[],steps:[]};continuation(ctx,targets);return {actions:ctx.steps.length,success:ctx.h.success};}

const filename=s=>s==='TRAIN'?'TRAIN_common_mode_fault_stream.jsonl':s+'_EVALUATOR_ONLY_common_mode_fault_stream.jsonl';
const certificate=s=>s+'_common_mode_fault_certificates_EVALUATOR_ONLY.jsonl';
function world(split,i,seen){for(let attempt=0;attempt<80;attempt++){
 const index=WORLD_BASE[split]+i*149+attempt*3,w=host.scenario(split,index);
 const h=host.start(w,'positive',false),d=(p)=>host.dist(h.g,p,-1)[w.s];
 if(seen.has(w.geometry_sha256)||d(w.goal)<5||d(w.goal2)<5||parent.pathTo(h,w.goal)[0]===parent.pathTo(h,w.goal2)[0])continue;
 seen.add(w.geometry_sha256);return {w,index,attempt};
 }throw Error('V71 divergent unseen topology unavailable '+split+'/'+i);}
function pairedReading(material,raw,calibration,domain,flip,channel,late){
 const inverted=channel==='PRIMARY'?calibration[material]:(domain==='COMMON_BUS'?calibration[material]:!calibration[material]);
 const after=(late&&flip&&((channel==='SECONDARY'&&domain!=='NO_FAULT')||(channel==='PRIMARY'&&domain==='COMMON_BUS')));
 return inverted!==after?(raw==='INTACT'?'ERODED':'INTACT'):raw;
}
function episode(w,threshold,sequence,calibration,domain,flip,mode){
 const h=host.start(w,'positive',false),ctx={h,threshold,count:{A:0,B:0},erasures:0,pending:[],steps:[]};
 const emit=(name,...data)=>ctx.pending.push([h.accepted,name,...data]);
 // Physical coupon establishes erosion threshold. All interventions and
 // pre-decision route actions are masked from training.
 record(ctx,'PAINT_A',0);
 if(threshold===1)delete h.painted[w.s];
 emit('LOCAL_PHYSICAL_COUPON',threshold===1?'ABRADED':'SURVIVED');
 const n=h.g.adj[h.pos][0][0],out=parent.pathTo(h,n)[0];record(ctx,out,0);
 delete h.painted[w.s];emit('COUPON_FIXTURE_RESET','A');record(ctx,reversed[out],0);
 chk(h.pos===w.s,'coupon reset');
 for(const m of ['A','B']){
  visit(ctx,m==='A'?w.goal:w.goal2,0);record(ctx,'PAINT_'+m,0);
  emit('LOCAL_CHANNEL_CHECK_'+m,'PRIMARY',calibration[m]?'ERODED':'INTACT');
  emit('LOCAL_CHANNEL_CHECK_'+m,'SECONDARY',domain==='COMMON_BUS'?(calibration[m]?'ERODED':'INTACT'):(calibration[m]?'INTACT':'ERODED'));
 }
 chk(h.stage===2,'prelude physical goals');visit(ctx,w.s,0);
 const move=parent.pathTo(h,h.g.adj[h.pos][0][0])[0];
 const measurements={},delivered={A:[],B:[]},schedule=[];
 let busClock=null;
 for(let k=0;k<112;k++){
  if([8,21,62].includes(k)){
   const q=[8,21,62].indexOf(k),m=sequence[q]||null;
   schedule.push({at:h.accepted,material:m,k});if(m)parent.shock(ctx,m);
   emit('UNVERIFIED_REMOTE_VIBRATION','UNKNOWN');
  }
  if(k===38||k===72){measurements[k]={clock:h.accepted,read:{
   A:h.painted[w.goal]==='A'?'INTACT':'ERODED',
   B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
  if(k===48){busClock=h.accepted;emit('TELEMETRY_FAULT_DOMAIN',domain);}
  if(k===80||k===90){
   const sample=k===80?72:38;
   for(const m of ['A','B']){
    if(mode[m]==='MISSING'){
     if(k===80)emit('REMOTE_MATERIAL_CHECK_'+m,'NO_VERIFIED_RESULT');
     continue;
    }
    const ms=measurements[sample];
    for(const channel of ['PRIMARY','SECONDARY']){
     const raw=pairedReading(m,ms.read[m],calibration,domain,flip,channel,sample===72);
     const receipt=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,channel,ms.clock,raw];
     ctx.pending.push(receipt);delivered[m].push(receipt);
    }
   }
  }
  record(ctx,k%2===0?move:reversed[move],0);
 }
 chk(h.pos===w.s&&ctx.pending.length===0,'detour reset and flushed input');
 const critical=ctx.steps.length,criticalInput=makeSense(h,[]),snapshot=copy(h);
 // The domain event tells the agent which channel's *calibration validity*
 // is compromised. Never use an unobserved flip or the actual remote paint.
 const labels={};
 for(const m of ['A','B']){
  const rows=delivered[m].filter(x=>x[2]==='PRIMARY'&&(domain!=='COMMON_BUS'||x[3]<busClock));
  const last=rows.sort((a,b)=>b[3]-a[3])[0];
  if(!last){labels[m]={state:'UNKNOWN',clock:null};continue;}
  const decoded=calibration[m]?(last[4]==='INTACT'?'ERODED':'INTACT'):last[4];
  // Any unverified shocks after an intact report are relevant even when
  // a later remote report is delivered (timestamp, not arrival semantics).
  const possibleNewDamage=schedule.some(s=>s.at>last[3]);
  labels[m]={state:decoded==='ERODED'?'ERODED':possibleNewDamage?'UNKNOWN':'INTACT',clock:last[3]};
 }
 const targets=['A','B'].filter(m=>labels[m].state!=='INTACT');
 const broken=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
 // A lower-information teacher may inspect an intact goal, but may NEVER
 // stop while a goal could have been erased by an unverified disturbance.
 chk(broken.every(m=>targets.includes(m)),'privileged shortcut would lose objective');
 continuation(ctx,targets);chk(h.done&&h.success,'teacher terminal success');
 const always=estimate(snapshot,['A','B']);
 const oracle=estimate(snapshot,broken);
 const stop=estimate(snapshot,[]);
 chk(always.success&&oracle.success,'comparison control invalid');
 chk(H(ctx.steps[critical].input)===H(criticalInput),'critical observation mismatch');
 return {episode:{steps:ctx.steps},certificate:{threshold,sequence,calibration,domain,flip,mode,
   critical_step:critical,critical_input_sha256:H(criticalInput),
   sampling_clocks:{early:measurements[38].clock,late:measurements[72].clock},fault_at:busClock,
   schedule,actual_broken:broken,teacher_labels:labels,targets,
   teacher_first_action:ctx.steps[critical].action,teacher_actions:ctx.steps.length-critical,
   always_actions:always.actions,oracle_actions:oracle.actions,
   always_success:always.success,stop_success:stop.success,
   physical_erasures:ctx.erasures,terminal_success:h.success,
   physical_host:'synthetic-v44-plus-v67-erasure',hypothetical_observation_abi:true}};
}
function write(out,name,rows,files){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});files[name]={rows:rows.length,bytes:b.length,sha256:H(b)};}
function generate(dir,{quick=false}={}){
 chk(!fs.existsSync(dir),'destination must not exist');fs.mkdirSync(dir,{recursive:true});
 const mf={schema:'driv3r/corpus_factory/common-mode-fault-v71/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,quick,
  generator_sha256:H(fs.readFileSync(__filename)),host_sha256:H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),parent_v70_sha256:H(fs.readFileSync(path.join(__dirname,'cross_calibrated_receipt_v70.js'))),
  design:'Sensor two-channel reliability under independent-secondary versus common-mode fault; long-horizon calibration, source-time stale receipts, physical erosion and conservative reinspection. Fault domain observable at event time, whether either sensor actually flipped remains hidden. Perfectly duplicated agreement cannot certify a failed common bus.',
  model_contract:'Only TRAIN_common_mode_fault_stream.jsonl is trainable; DEV/HELDOUT and ALL certificates are evaluator-only, no training. Prefix actions weight 0; critical and distinct later recovery decisions weighted. Synthetic bus/reset/sensor signals are NOT native CyberSand ABI.',
  caveats:['All source-host physics and telemetry extensions synthetic; not native CyberSand.','Single deterministic fault warning and 112 scripted detour.','Sensor modes stable before warning; no proof of neural learnability.','A verified independent primary channel is assumed trustworthy after secondary-isolated fault.'],files:{},splits:{}};
 const seen=new Set(),seenPairs=new Set();let totalEpisodes=0,totalDecisions=0,totalAvoided=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const streams=[],certs=[],geometries=[];let decisions=0,success=0,avoided=0,safe=0,wrongStop=0,masked=0,common=0,isolated=0,distinctActions=0;
  for(let i=0;i<(quick?1:COUNTS[split]);i++){
   const {w,index,attempt}=world(split,i,seen);geometries.push({sha256:w.geometry_sha256,index,attempt,world_seed:w.seed});
   for(const threshold of [1,2])for(const sequence of [HISTORIES[0],HISTORIES[1+(i%3)]]){
    const group=[];
    for(const calibration of CALIBRATIONS)for(const domain of DOMAINS)for(const flip of (domain==='NO_FAULT'?[false]:[false,true]))for(const modeA of MODES)for(const modeB of MODES){
     if(flip&&modeA==='MISSING'&&modeB==='MISSING')continue; // unreachable sensor state, near-identical observation

     const {episode:e,certificate:c}=episode(w,threshold,sequence,calibration,domain,flip,{A:modeA,B:modeB});
     c.split=split;c.world_index=i;c.source_world_index=index;c.geometry_sha256=w.geometry_sha256;c.episode=streams.length;
     c.episode_sha256=H(e.steps.map(s=>[s.input,s.action,s.weight]));
     // Prevent repeated high-volume continuation actions from overwhelming
     // rare causal choices. Hash includes full model-visible input, never cert.
     if(split==='TRAIN')for(let k=c.critical_step+1;k<e.steps.length;k++){
      const s=e.steps[k],signature=H([s.input,s.action]);if(seenPairs.has(signature)){s.weight=0;s.focus_loss_weight=0;masked++;}else seenPairs.add(signature);
     }
     chk(e.steps[c.critical_step].weight===1,'critical supervision masked');
     c.episode_sha256=H(e.steps.map(s=>[s.input,s.action,s.weight]));
     streams.push(e);certs.push(c);group.push(c);
     decisions+=e.steps.length;success+=Number(c.terminal_success);
     avoided+=c.always_actions-c.teacher_actions;safe+=Number(c.targets.every(m=>['A','B'].includes(m)));
     wrongStop+=Number(!c.stop_success);common+=Number(domain==='COMMON_BUS');isolated+=Number(domain==='ISOLATED_SECONDARY');
    }
    chk(group.length===36,'counterfactual group size');
    chk(new Set(group.map(c=>c.critical_input_sha256)).size===1,'identical current input invariant');
    // Some physical histories necessarily demand the same repair action in every arm.
    chk(group.every(c=>H(c.actual_broken)===H(group[0].actual_broken)),'observation changes physical state');
    distinctActions=Math.max(distinctActions,new Set(group.map(c=>c.teacher_first_action)).size);
   }
  }
  write(dir,filename(split),streams,mf.files);write(dir,certificate(split),certs,mf.files);
  mf.splits[split]={worlds:geometries.length,episodes:streams.length,decisions,terminal_successes:success,counterfactual_36_arm_groups:geometries.length*2*2,
   fault_common_arm_count:common,fault_isolated_secondary_arm_count:isolated,physical_stop_only_failures:wrongStop,
   mean_actions_saved_vs_always_reinspect:avoided/streams.length,masked_train_followup_decisions:masked,
   maximum_distinct_critical_actions:distinctActions,geometry:geometries};
  totalEpisodes+=streams.length;totalDecisions+=decisions;totalAvoided+=avoided;
 }
 mf.total_episodes=totalEpisodes;mf.total_decisions=totalDecisions;mf.mean_actions_saved_vs_always_reinspect=totalAvoided/totalEpisodes;
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){chk(process.argv[2],'usage: node common_mode_fault_v71.js NEW_DIRECTORY [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv.includes('--quick')}).splits,null,2));}
module.exports={episode,generate,pairedReading,world,filename,certificate,H,HISTORIES,DOMAINS,CALIBRATIONS,MODES};
