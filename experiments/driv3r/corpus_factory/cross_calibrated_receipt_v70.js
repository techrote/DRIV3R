#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v70. DATA, not a trainer. Physical host is an
 * explicitly synthetic extension of v44, never native CyberSand.
 * Causal stress: physically identical remote erosion, crossed observation
 * availability. Unknown state requires a local visit, not privileged truth. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const prior=require('./remote_abrasion_v67.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(x,m)=>{if(!x)throw Error('V70 '+m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const S={TRAIN:6,DEV:3,HELDOUT:3};
const BASE={TRAIN:174000,DEV:184000,HELDOUT:194000};
const MODES=['NONE','REORDERED'];
const CALIBRATIONS=[{A:false,B:false},{A:false,B:true},{A:true,B:false},{A:true,B:true}];
const SEQS=[[],['A','A','A'],['B','B','B'],['A','B','A'],['B','A','B'],['A','B','B']]; // six physical shock histories, including post-sample A/B damage
const names=s=>s==='TRAIN'?s+'_cross_calibrated_receipt_stream.jsonl':s+'_EVALUATOR_ONLY_cross_calibrated_receipt_stream.jsonl';
const certName=s=>s+'_cross_calibrated_receipt_certificates_EVALUATOR_ONLY.jsonl';
function makeSense(h,pending){
 const inp=host.input(h),[x,y]=h.g.coords[h.pos];
 inp.painted=inp.painted.filter(([i])=>{const [xx,yy]=h.g.coords[i];return Math.abs(x-xx)+Math.abs(y-yy)<=1});
 inp.v='synthetic-v70-cross-calibrated-timestamped-receipt';inp.events_delta=pending.splice(0);
 return inp;
}
function record(ctx,action,weight){const input=makeSense(ctx.h,ctx.pending);check(input.legal.includes(action),'illegal '+action);ctx.steps.push({input,action,weight,focus_loss_weight:weight});check(host.apply(ctx.h,action).accepted,'host rejection '+action);}
// Intentionally use independent deterministic local routing rather than a model policy.
// v67 pathTo is lexical/private; use graph shortest-path deterministically.
function pathTo(h,to){const steps=[];let at=h.pos;while(at!==to){const ds=host.dist(h.g,to,-1);const next=h.g.adj[at].filter(([n])=>ds[n]===ds[at]-1).sort((a,b)=>a[0]-b[0])[0];check(next,'route impossible');const [x,y]=h.g.coords[at], [xx,yy]=h.g.coords[next[0]];steps.push('MOVE_'+(xx>x?'E':xx<x?'W':yy>y?'S':'N'));at=next[0];check(steps.length<h.g.coords.length,'path loop');}return steps;}
function visit(ctx,to,weight){for(const action of pathTo(ctx.h,to))record(ctx,action,weight);}
function selectWorld(split,i,seen){for(let attempt=0;attempt<75;attempt++){
 const ix=BASE[split]+i*171+attempt*3,w=host.scenario(split,ix),g=host.hydrate(w),h=host.start(w,'positive',false),d=(a,b)=>host.dist(g,b,-1)[a];
 if(d(w.s,w.goal)<5||d(w.s,w.goal2)<5||pathTo(h,w.goal)[0]===pathTo(h,w.goal2)[0]||seen.has(w.geometry_sha256))continue;
 return {w,source_index:ix,attempt};}throw Error('V70 no divergent geometry '+split+'/'+i);}
function shock(ctx,mat){const t=mat==='A'?ctx.h.w.goal:ctx.h.w.goal2;ctx.count[mat]++;
 if(ctx.count[mat]===ctx.threshold&&ctx.h.painted[t]===mat){delete ctx.h.painted[t];ctx.erasures++}
}
function continuation(ctx,targets){
 const w=ctx.h.w,matIndex=m=>m==='A'?w.goal:w.goal2;
 const orders=targets.length===2?[['A','B'],['B','A']]:[targets];
 const ds=(a,b)=>host.dist(ctx.h.g,b,-1)[a];
 const order=orders.map(seq=>({seq,cost:seq.reduce((cost,m,i)=>cost+ds(i?matIndex(seq[i-1]):ctx.h.pos,matIndex(m)),0)})).sort((a,b)=>a.cost-b.cost||a.seq.join().localeCompare(b.seq.join()))[0].seq;
 for(const m of order){const p=matIndex(m);visit(ctx,p,1);const observed=host.input(ctx.h).painted.find(([i])=>i===p);
  if(!observed){record(ctx,'PAINT_'+m,1)}else check(observed[1]===m,'unexpected observed material');}
 record(ctx,'STOP',1);
}
function estimate(snapshot,targets){const c={h:cp(snapshot),pending:[],steps:[]};continuation(c,targets);return {actions:c.steps.length,success:c.h.success,first:c.steps[0].action}}
function episode(w,threshold,sequence,mode,calibration){
 const h=host.start(w,'positive',false),ctx={h,threshold,pending:[],steps:[],count:{A:0,B:0},erasures:0};
 const coupon=[];
 record(ctx,'PAINT_A',0);if(threshold===1){delete h.painted[w.s];coupon.push('ABRADED')}else coupon.push('SURVIVED');
 ctx.pending.push([h.accepted,'LOCAL_PHYSICAL_COUPON',coupon[0]]);
 const n=h.g.adj[h.pos][0][0],out=pathTo(h,n)[0];record(ctx,out,0);
 // Synthetic test fixture resets its sacrificial coupon; not a model action.
 delete h.painted[w.s];ctx.pending.push([h.accepted,'COUPON_FIXTURE_RESET','A']);
 const back={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E',MOVE_E:'MOVE_W'}[out];record(ctx,back,0);
 check(h.pos===w.s&&h.painted[w.s]===undefined,'coupon not clean');
 visit(ctx,w.goal,0);record(ctx,'PAINT_A',0);
 ctx.pending.push([h.accepted,'LOCAL_SENSOR_CHECK_A',calibration.A?'ERODED':'INTACT']);
 visit(ctx,w.goal2,0);record(ctx,'PAINT_B',0);
 ctx.pending.push([h.accepted,'LOCAL_SENSOR_CHECK_B',calibration.B?'ERODED':'INTACT']);check(h.stage===2,'prelude mission not painted');
 visit(ctx,w.s,0);check(h.pos===w.s,'not remote');
 const step=pathTo(h,h.g.adj[h.pos][0][0])[0],rstep={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E',MOVE_E:'MOVE_W'}[step];
 const schedule=[],captured={};let report=null;const delivered={A:[],B:[]};
 for(let k=0;k<112;k++){
  if([8,21,62].includes(k)){
   const i=[8,21,62].indexOf(k),mat=sequence[i]||null;schedule.push({at:h.accepted,material:mat,k});
   if(mat)shock(ctx,mat);
   // Deliberately noncertifying disturbance. Same visible event for all physical arms.
   ctx.pending.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);
  }
  if(k===38||k===72){captured[k]={clock:h.accepted,read:{
   A:h.painted[w.goal]==='A'?'INTACT':'ERODED',
   B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
  if(k===80||k===90){
   for(const m of ['A','B']){
    let which=null;
    if(mode[m]==='NONE'&&k===80){ctx.pending.push([h.accepted,'REMOTE_MATERIAL_CHECK_'+m,'NO_VERIFIED_RESULT']);continue;}
    if(mode[m]==='REORDERED')which=k===80?72:38;
    if(which!==null){const measurement=captured[which];
     const actual=measurement.read[m],reading=calibration[m]?(actual==='INTACT'?'ERODED':'INTACT'):actual;
     const row=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,measurement.clock,reading];
     ctx.pending.push(row);delivered[m].push(row);}
   }
   if(k===90)report={at:h.accepted,mode};
  }
  record(ctx,k%2===0?step:rstep,0);
 }
 check(h.pos===w.s&&ctx.pending.length===0&&report,'bad detour end');
 const physicalBroken=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
 // Conservatism is derived ONLY from report provenance. Earlier INTACT can
 // be invalidated by later unseen disturbance; earlier ERODED is monotone.
 function receipt(m){return delivered[m].slice().sort((a,b)=>b[2]-a[2])[0]||null;}
 const latest=Object.fromEntries(['A','B'].map(m=>[m,receipt(m)]));
 const decoded=m=>!latest[m]?null:calibration[m]?(latest[m][3]==='INTACT'?'ERODED':'INTACT'):latest[m][3];
 const uncertain=['A','B'].filter(m=>!latest[m]||(decoded(m)==='INTACT'&&latest[m][2]<captured[72].clock));
 const targeted=['A','B'].filter(m=>uncertain.includes(m)||decoded(m)==='ERODED');
 const arrivalTargets=['A','B'].filter(m=>!delivered[m].length||delivered[m][delivered[m].length-1][3]==='ERODED');
 const critical=ctx.steps.length,criticalInput=makeSense(h,[]),snapshot=cp(h);
 continuation(ctx,targeted);
 check(h.done&&h.success,'expert unsuccessful');
 const naiveStop=estimate(snapshot,[]),always=estimate(snapshot,['A','B']),omniscient=estimate(snapshot,physicalBroken),arrival=estimate(snapshot,arrivalTargets);
 check(always.success&&omniscient.success,'control did not succeed');
 check(criticalInput.events_delta.length===0,'contemporaneous report leak');
 check(H(ctx.steps[critical].input)===H(criticalInput),'critical leak');
 return {episode:{steps:ctx.steps},certificate:{threshold,sequence,mode,calibration,critical_step:critical,
  report_at_accepted:report.at,remote_schedule:schedule,sampling_clocks:{early:captured[38].clock,late:captured[72].clock},
  physical_broken:physicalBroken,uncertain,planned_targets:targeted,arrival_targets:arrivalTargets,
  decoded_source_time:Object.fromEntries(['A','B'].map(m=>[m,decoded(m)])),
  initial_supervised_action:ctx.steps[critical].action,
  critical_input_sha256:H(criticalInput),host_terminal_success:h.success,
  teacher_continuation_actions:ctx.steps.length-critical,
  stop_only_success:naiveStop.success,always_reinspect_actions:always.actions,
  always_reinspect_success:always.success,privileged_oracle_actions:omniscient.actions,
  arrival_only_success:arrival.success,arrival_only_actions:arrival.actions,
  redundant_inspection_actions:ctx.steps.length-critical-omniscient.actions,
  remote_eroded_count:ctx.erasures,physical_system:'synthetic-v44-plus-physical-erasure-extension',
  event_semantics:'hypothetical monotone physical erosion and source-time receipts; NOT CyberSand ABI'}};
}
function write(out,file,rows,mf){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,file),b,{flag:'wx'});mf.files[file]={rows:rows.length,bytes:b.length,sha256:H(b)}}
function generate(out,{quick=false}={}){
 check(!fs.existsSync(out),'destination must not exist');fs.mkdirSync(out,{recursive:true});
 const mf={schema:'driv3r/corpus_factory/cross-calibrated-receipt-v70/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,
  generator_sha256:H(fs.readFileSync(__filename)),host_sha256:H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),
  parent_v67_sha256:H(fs.readFileSync(path.join(__dirname,'remote_abrasion_v67.js'))),parent_v68_sha256:H(fs.readFileSync(path.join(__dirname,'reinspection_quality_v68.js'))),
  design:'Cross two-material remote erosion with independently calibrated INVERT/MATCH sensor modes and missing/out-of-order timestamped receipts. Calibration is directly grounded by co-located physical paint, not a privileged sensor-mode label; later sensor readings are decoded using remembered local calibration plus source timestamps. No hidden state used to label teacher actions.',
  observation_contract:'Synthetic local radius-1 material view, earlier LOCAL_SENSOR_CHECK_A/B simultaneously with visible physically painted goal; remote sensor may report truth or inverted state consistently by material. Source-time receipts arrive out of order, and NONE suppresses them. No sensor-mode field is exposed. The agent must retain the physical-ground-truth comparison and source timestamps; no native CyberSand sensor or material ABI claim.',
  model_contract:'Only TRAIN cross_calibrated_receipt_stream contains trainable examples; all prefix interventions and physical probe steps loss weight 0; DEV/HELDOUT and all certificates evaluator-only. Observation arms alter input but not underlying physical intervention or host dynamics. Never train on certificates, manifests, privileged physical broken targets or diagnostic validation.',
  caveats:['No model weights trained; a symbolic source-time and historical-calibration controller remains sufficient.','Synthetic local calibration checks, sensor inversion and monotone erosion are not established native CyberSand features.','Conservatively reinspects when report is missing; teacher-forced detours do not prove autonomy.','Calibrations are stable within one episode and the source clock is trusted; these assumptions are research hypotheses.'],
  quick,files:{},splits:{}};
 const seen=new Set(),sigSeen=new Set();let sumActions=0,sumEpisodes=0,totalAvoided=0,totalUnknown=0,totalStop=0,totalArrivalLoss=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const count=quick?Math.min(2,S[split]):S[split],streams=[],certs=[],geoms=[];let actions=0,masked=0,success=0,avoided=0,oracleGap=0,unk=0,stop=0,verified=0,criticalSet=0,arrivalLoss=0;
  for(let i=0;i<count;i++){
   const {w,source_index,attempt}=selectWorld(split,i,seen);check(!seen.has(w.geometry_sha256),'split topology leak');seen.add(w.geometry_sha256);geoms.push({sha256:w.geometry_sha256,source_world_index:source_index,attempt});
   const group=[];
   for(const threshold of [1,2])for(const seq of SEQS)for(const calibration of CALIBRATIONS)for(const modeA of MODES)for(const modeB of MODES){
    const {episode:ep,certificate:c}=episode(w,threshold,seq,{A:modeA,B:modeB},calibration);
    c.split=split;c.episode=streams.length;c.world_index=i;c.source_world_index=source_index;c.geometry_sha256=w.geometry_sha256;c.topology_seed=w.seed;c.world_goal=w.goal;c.world_goal2=w.goal2;c.start=w.s;
    for(let k=c.critical_step+1;k<ep.steps.length;k++)if(split==='TRAIN'&&ep.steps[k].weight===1){
     const signature=H([ep.steps[k].input,ep.steps[k].action]);if(sigSeen.has(signature)){ep.steps[k].weight=0;ep.steps[k].focus_loss_weight=0;masked++}else sigSeen.add(signature);
    }
    c.episode_input_action_sha256=H(ep.steps.map(s=>[s.input,s.action,s.weight]));
    check(ep.steps[c.critical_step].weight===1,'masked critical');
    streams.push(ep);certs.push(c);group.push(c);
    actions+=ep.steps.length;success+=Number(c.host_terminal_success);avoided+=c.always_reinspect_actions-c.teacher_continuation_actions;
    oracleGap+=c.redundant_inspection_actions;unk+=Number(c.uncertain.length>0);stop+=Number(c.stop_only_success);verified+=Number(modeA==='REORDERED'&&modeB==='REORDERED');arrivalLoss+=Number(!c.arrival_only_success);
   }
   check(group.length===SEQS.length*2*CALIBRATIONS.length*MODES.length**2&&new Set(group.map(c=>c.critical_input_sha256)).size===1,'critical observations not identical in group');
   check(new Set(group.map(c=>c.initial_supervised_action)).size>=3,'route/STOP contrast missing');
   // For the same hidden physical state, less telemetry must never change physical world outcome.
   for(let k=0;k<group.length;k+=16){const b=group.slice(k,k+16);check(b.every(x=>H(x.physical_broken)===H(b[0].physical_broken)),'mode changed physics')}
   criticalSet++;
  }
  write(out,names(split),streams,mf);write(out,certName(split),certs,mf);
  mf.splits[split]={worlds:count,episodes:streams.length,decisions:actions,host_terminal_successes:success,critical_192_arm_groups:criticalSet,
   both_reordered_episodes:verified,arrival_order_control_goal_failures:arrivalLoss,partially_or_unverified_episodes:unk,stop_only_successes:stop,
   unnecessary_extra_inspection_actions_vs_privileged_oracle:oracleGap,travel_actions_avoided_vs_always_reinspect:avoided,
   redundant_train_followups_loss_masked:masked,geometry:geoms};
  sumActions+=actions;sumEpisodes+=streams.length;totalAvoided+=avoided;totalUnknown+=oracleGap;totalStop+=stop;totalArrivalLoss+=arrivalLoss;
 }
 mf.total_episodes=sumEpisodes;mf.total_decisions=sumActions;mf.mean_actions_avoided_vs_always_reinspect=totalAvoided/sumEpisodes;
 mf.mean_unavoidable_ambiguity_cost_vs_privileged_oracle=totalUnknown/sumEpisodes;mf.stop_only_success_rate=totalStop/sumEpisodes;mf.arrival_only_success_rate=1-totalArrivalLoss/sumEpisodes;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){check(process.argv[2],'usage: node cross_calibrated_receipt_v70.js NEW_DIRECTORY [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv.includes('--quick')}).splits,null,2))}
module.exports={generate,episode,makeSense,selectWorld,pathTo,visit,record,continuation,estimate,shock,MODES,CALIBRATIONS,SEQS,H,cp,names,certName};
