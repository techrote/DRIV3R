#!/usr/bin/env node
'use strict';
/* Independent v69 physical synthetic-host reconstruction, observation replay,
 * strict provenance and no-privileged-state training-label checks. Does not
 * import the generator, its continuation policy or test certificates into input. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(x,m)=>{if(!x)throw Error('V69 AUDIT '+m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
function* streamRows(filename){const fd=fs.openSync(filename,'r'),buf=Buffer.alloc(1<<20);let tail='';try{while(true){const n=fs.readSync(fd,buf,0,buf.length,null);if(n===0)break;const chunk=tail+buf.toString('utf8',0,n);const lines=chunk.split('\n');tail=lines.pop();for(const line of lines)if(line)yield JSON.parse(line);}if(tail.trim())yield JSON.parse(tail);}finally{fs.closeSync(fd)}}
const splits=['TRAIN','DEV','HELDOUT'],modes=['NONE','EARLY_DELAYED','LATE','REORDERED'];
const seqs=[[],['A','A','A'],['B','B','B'],['A','B','A'],['B','A','B'],['A','B','B']];
function filename(s){return s==='TRAIN'?s+'_stale_receipt_stream.jsonl':s+'_EVALUATOR_ONLY_stale_receipt_stream.jsonl'}
function localObservation(h,events){const x=host.input(h),[px,py]=h.g.coords[h.pos];
 x.painted=x.painted.filter(([id])=>{const [cx,cy]=h.g.coords[id];return Math.abs(cx-px)+Math.abs(cy-py)<=1});
 x.v='synthetic-v69-local-paint-timestamped-receipt';x.events_delta=events;return x;}
function shortest(h,t){let at=h.pos;const out=[];while(at!==t){
 const q=[[t,0]],d=new Map([[t,0]]);for(let i=0;i<q.length;i++){const [u,c]=q[i];for(const [v] of h.g.adj[u])if(!d.has(v)){d.set(v,c+1);q.push([v,c+1])}}
 const opts=h.g.adj[at].filter(([v])=>d.get(v)===d.get(at)-1).sort((a,b)=>a[0]-b[0]);check(opts.length>0,'BFS path');
 const next=opts[0][0],[x,y]=h.g.coords[at],[xx,yy]=h.g.coords[next];out.push('MOVE_'+(xx>x?'E':xx<x?'W':yy>y?'S':'N'));at=next;check(out.length<h.g.coords.length,'path loop');
 }return out;}
function plan(snapshot,targets){const h=cp(snapshot),result=[],goal=m=>m==='A'?h.w.goal:h.w.goal2;
 const dist=(a,b)=>{const x=cp(h);x.pos=a;return shortest(x,b).length};
 let orders=targets.length===2?[['A','B'],['B','A']]:[targets];
 orders.sort((aa,bb)=>{const cost=arr=>arr.reduce((s,m,i)=>s+dist(i?goal(arr[i-1]):h.pos,goal(m)),0);return cost(aa)-cost(bb)||aa.join().localeCompare(bb.join())});
 function a(action){check(host.legal(h).includes(action),'planned action not legal');result.push(action);check(host.apply(h,action).accepted,'planned action failed')}
 for(const m of orders[0]){for(const act of shortest(h,goal(m)))a(act);
  const seen=host.input(h).painted.find(([id])=>id===goal(m));if(!seen)a('PAINT_'+m);else check(seen[1]===m,'incorrect paint');}
 a('STOP');return {actions:result,success:h.success};
}
function audit(dir){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 check(mf.schema==='driv3r/corpus_factory/stale-receipt-v69/1'&&mf.synthetic_not_native_CyberSand,'schema or native claim');
 for(const [field,name] of [['host_sha256','unseen_topology_v44.js'],['parent_v67_sha256','remote_abrasion_v67.js'],['parent_v68_sha256','reinspection_quality_v68.js'],['generator_sha256','stale_receipt_v69.js']])
  check(mf[field]===H(fs.readFileSync(path.join(__dirname,name))),field+' wrong');
 for(const [name,v] of Object.entries(mf.files)){const b=fs.readFileSync(path.join(dir,name));check(H(b)===v.sha256&&b.length===v.bytes,'file hash/bytes '+name)}
 let episodes=0,decisions=0,stops=0,avoided=0,gap=0,arrivalFailures=0,mask=0,physicalChecks=0,reorderedFailures=0;
 const geometries=new Map(),groups=new Map(),noSignalHashes=new Map(),sameStatePairs=new Map(),splitStats={};
 for(const split of splits){
  const policies=streamRows(path.join(dir,filename(split))),certs=read(path.join(dir,split+'_stale_receipt_certificates_EVALUATOR_ONLY.jsonl'));
  check(certs.length%192===0,'192-arm split cardinality');
  let actionCount=0,failedArrival=0,completed=0;
  for(let id=0;id<certs.length;id++){
   const c=certs[id],chunk=policies.next();check(!chunk.done,'missing stream episode');const e=chunk.value,steps=e.steps;
   check(c.split===split&&c.episode===id&&steps.length>c.critical_step,'index / prefix');
   check(c.threshold===(Math.floor((id%192)/96)+1),'threshold order');
   check(JSON.stringify(c.sequence)===JSON.stringify(seqs[Math.floor((id%96)/16)]),'sequence order');
   check(c.mode.A===modes[Math.floor((id%16)/4)]&&c.mode.B===modes[id%4],'mode order');
   const w=host.scenario(split,c.source_world_index);
   check(c.geometry_sha256===w.geometry_sha256&&c.topology_seed===w.seed&&w.s===c.start&&w.goal===c.world_goal&&w.goal2===c.world_goal2,'world identity');
   if(!geometries.has(w.geometry_sha256))geometries.set(w.geometry_sha256,split);
   check(geometries.get(w.geometry_sha256)===split,'split geometry leak');
   const h=host.start(w,'positive',false),counts={A:0,B:0},measured={},received={A:[],B:[]};
   let gotCoupon=false,gotReset=false,remote=0,physicalCount=0,criticalSnap=null,latestReportStep=-1;
   const detourStart=c.critical_step-112;
   check(detourStart>0,'short detour');
   for(let k=0;k<steps.length;k++){
    const s=steps[k],expected=[];
    if(h.accepted===1){check(!gotCoupon,'duplicate coupon');gotCoupon=true;
     if(c.threshold===1){check(h.painted[w.s]==='A','missing painted coupon');delete h.painted[w.s]}
     expected.push([h.accepted,'LOCAL_PHYSICAL_COUPON',c.threshold===1?'ABRADED':'SURVIVED']);}
    if(h.accepted===2){check(!gotReset,'duplicate reset');gotReset=true;
     delete h.painted[w.s];expected.push([h.accepted,'COUPON_FIXTURE_RESET','A']);}
    if(k>=detourStart&&k<c.critical_step){
     const tick=k-detourStart;
     if([8,21,62].includes(tick)){
      const i=[8,21,62].indexOf(tick),mat=c.sequence[i]||null;
      check(c.remote_schedule[i].k===tick&&c.remote_schedule[i].at===h.accepted&&c.remote_schedule[i].material===mat,'physical schedule tampering');
      if(mat){counts[mat]++;const goal=mat==='A'?w.goal:w.goal2;
       if(counts[mat]===c.threshold&&h.painted[goal]===mat){delete h.painted[goal];physicalCount++;}}
      expected.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);remote++;
     }
     if(tick===38||tick===72){measured[tick]={clock:h.accepted,read:{
      A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
     if(tick===80||tick===90){
      for(const m of ['A','B']){
       const mode=c.mode[m];let measurement=null;
       if(mode==='NONE'&&tick===80){expected.push([h.accepted,'REMOTE_MATERIAL_CHECK_'+m,'NO_VERIFIED_RESULT']);continue;}
       if(mode==='EARLY_DELAYED'&&tick===90)measurement=38;
       if(mode==='LATE'&&tick===80)measurement=72;
       if(mode==='REORDERED')measurement=tick===80?72:38;
       if(measurement!==null){const r=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,measured[measurement].clock,measured[measurement].read[m]];
        expected.push(r);received[m].push(r);}
      }
      if(expected.length)latestReportStep=k;
     }
    }
    const observed=localObservation(h,expected);
    check(H(s.input)===H(observed),'observation mismatch '+split+'/'+id+'/'+k);
    check(observed.legal.includes(s.action)&&host.legal(h).includes(s.action),'illegal host action');
    check(s.weight===s.focus_loss_weight&&[0,1].includes(s.weight),'bad loss mask');
    if(k<c.critical_step)check(s.weight===0,'prefix training leakage');
    if(k===c.critical_step){criticalSnap=cp(h);check(s.weight===1,'critical label masked');
     check(k-latestReportStep>=20,'critical report too recent');
     check(expected.length===0,'last-step event leak');check(H(observed)===c.critical_input_sha256,'critical hash');
     check(s.action===c.initial_supervised_action,'critical action changed');}
    check(host.apply(h,s.action).accepted,'host rejected');decisions++;actionCount++;
   }
   check(gotCoupon&&gotReset&&remote===3,'missing dynamic physics');
   check(measured[38]&&measured[72]&&c.sampling_clocks.early===measured[38].clock&&c.sampling_clocks.late===measured[72].clock,'measurement clock fraud');
   check(criticalSnap&&h.done&&h.success,'unsuccessful episode');
   const broken=['A','B'].filter(m=>criticalSnap.painted[m==='A'?w.goal:w.goal2]!==m);
   const latest=Object.fromEntries(['A','B'].map(m=>[m,received[m].slice().sort((a,b)=>b[2]-a[2])[0]||null]));
   const unknown=['A','B'].filter(m=>!latest[m]||(latest[m][3]==='INTACT'&&latest[m][2]<measured[72].clock));
   const targets=['A','B'].filter(m=>unknown.includes(m)||(latest[m]&&latest[m][3]==='ERODED'));
   const arrivalTargets=['A','B'].filter(m=>!received[m].length||received[m][received[m].length-1][3]==='ERODED');
   check(H(broken)===H(c.physical_broken)&&H(unknown)===H(c.uncertain)&&H(targets)===H(c.planned_targets)&&H(arrivalTargets)===H(c.arrival_targets),'privileged or invalid teacher label');
   const teacher=plan(criticalSnap,targets),always=plan(criticalSnap,['A','B']),oracle=plan(criticalSnap,broken),stop=plan(criticalSnap,[]),arrival=plan(criticalSnap,arrivalTargets);
   check(H(steps.slice(c.critical_step).map(z=>z.action))===H(teacher.actions),'trajectory contradicts provenance-aware teacher');
   check(teacher.success&&always.success&&oracle.success,'control physics failure');
   check(stop.success===c.stop_only_success&&always.actions.length===c.always_reinspect_actions&&c.always_reinspect_success&&
    oracle.actions.length===c.privileged_oracle_actions&&teacher.actions.length===c.teacher_continuation_actions&&
    teacher.actions.length-oracle.actions.length===c.redundant_inspection_actions&&
    arrival.success===c.arrival_only_success&&arrival.actions.length===c.arrival_only_actions,'control cost mismatch');
   check(c.remote_eroded_count===physicalCount,'wrong erosion count');
   check(c.episode_input_action_sha256===H(steps.map(s=>[s.input,s.action,s.weight])),'row signature tampering');
   const gkey=split+':'+c.world_index;if(!groups.has(gkey))groups.set(gkey,{inputs:new Set(),actions:new Set(),modes:new Set(),physical:new Map(),count:0});
   const g=groups.get(gkey);g.inputs.add(c.critical_input_sha256);g.actions.add(c.initial_supervised_action);
   g.modes.add(c.mode.A+':'+c.mode.B);g.count++;
   const pk=H([c.threshold,c.sequence]);if(g.physical.has(pk))check(g.physical.get(pk)===H(broken),'observation changed physical outcome');else g.physical.set(pk,H(broken));
   // For completely unverified modes, all same-threshold physically different
   // histories must look identical to the model before the critical action.
   if(c.mode.A==='NONE'&&c.mode.B==='NONE'){
    const key=split+':'+c.world_index+':'+c.threshold;
    const sig=H(steps.slice(0,c.critical_step).map(z=>z.input));
    if(noSignalHashes.has(key))check(noSignalHashes.get(key)===sig,'unverified observation leaked hidden shock');else noSignalHashes.set(key,sig);
   }
   // Mode-crossed tasks with same source world and physical state support
   // learned history dependence, not a current-observation discriminator.
   const pkey=split+':'+c.world_index+':'+pk;
   if(!sameStatePairs.has(pkey))sameStatePairs.set(pkey,new Set());sameStatePairs.get(pkey).add(c.mode.A+':'+c.mode.B);
   episodes++;completed++;physicalChecks++;
   stops+=Number(stop.success);avoided+=always.actions.length-teacher.actions.length;gap+=teacher.actions.length-oracle.actions.length;
   arrivalFailures+=Number(!arrival.success);failedArrival+=Number(!arrival.success);
   if(c.mode.A==='REORDERED'||c.mode.B==='REORDERED')reorderedFailures+=Number(!arrival.success);
   mask+=steps.slice(c.critical_step+1).filter(z=>z.weight===0).length;
  }
  check(policies.next().done,'extra stream data');
  check(mf.splits[split].episodes===completed&&mf.splits[split].decisions===actionCount&&mf.splits[split].arrival_order_control_goal_failures===failedArrival,'split tallies');
  splitStats[split]={episodes:completed,replayed_decisions:actionCount,arrival_only_failures:failedArrival};
 }
 for(const [id,g] of groups)check(g.inputs.size===1&&g.actions.size>=3&&g.modes.size===16&&g.physical.size===12&&g.count===192,'causal group '+id);
 for(const set of sameStatePairs.values())check(set.size===16,'missing paired mask');
 check(mf.total_episodes===episodes&&mf.total_decisions===decisions&&mf.stop_only_success_rate===stops/episodes&&mf.arrival_only_success_rate===1-arrivalFailures/episodes,'manifest totals');
 check(Math.abs(mf.mean_actions_avoided_vs_always_reinspect-avoided/episodes)<1e-9,'savings aggregate wrong');
 check(Math.abs(mf.mean_unavoidable_ambiguity_cost_vs_privileged_oracle-gap/episodes)<1e-9,'oracle-gap aggregate wrong');
 check(new Set(geometries.values()).size===3,'split coverage absent');
 return {PASS:true,synthetic_not_native_CyberSand:true,episodes,independently_replayed_host_actions:decisions,geometry_split_disjoint_worlds:geometries.size,
  complete_192_arm_groups:groups.size,paired_physical_histories:sameStatePairs.size,unverified_history_identity_groups:noSignalHashes.size,
  all_teacher_episodes_successful:true,latest_arrival_control_goal_failures:arrivalFailures,reordered_arm_arrival_control_failures:reorderedFailures,
  mean_actions_saved_vs_always_reinspect:avoided/episodes,mean_unknown_cost_vs_privileged_oracle:gap/episodes,
  stop_only_success_rate:stops/episodes,masked_followup_steps:mask,splitStats};
}
if(require.main===module){try{check(process.argv[2],'usage: node test_stale_receipt_v69.js CORPUS_DIR');console.log(JSON.stringify(audit(path.resolve(process.argv[2])),null,2))}catch(e){console.error(e.stack);process.exitCode=1}}
module.exports={audit};
