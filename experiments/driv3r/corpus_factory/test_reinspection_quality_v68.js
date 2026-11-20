#!/usr/bin/env node
'use strict';
/* Independent v68 physical synthetic-host reconstruction, observation replay,
 * strict provenance and no-privileged-state training-label checks. Does not
 * import the generator, its continuation policy or test certificates into input. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(x,m)=>{if(!x)throw Error('V68 AUDIT '+m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
function* streamRows(filename){const fd=fs.openSync(filename,'r'),buf=Buffer.alloc(1<<20);let tail='';try{while(true){const n=fs.readSync(fd,buf,0,buf.length,null);if(n===0)break;const chunk=tail+buf.toString('utf8',0,n);const lines=chunk.split('\n');tail=lines.pop();for(const line of lines)if(line)yield JSON.parse(line);}if(tail.trim())yield JSON.parse(tail);}finally{fs.closeSync(fd)}}
const splits=['TRAIN','DEV','HELDOUT'],modes=['NONE','A_ONLY','B_ONLY','BOTH'];
const seqs=[[],['A'],['B'],['A','A'],['B','B'],['A','B'],['B','A'],['A','B','A'],['B','A','B'],['A','B','B']];
function filename(s){return s==='TRAIN'?s+'_observation_quality_stream.jsonl':s+'_EVALUATOR_ONLY_observation_quality_stream.jsonl'}
function localObservation(h,events){const x=host.input(h),[px,py]=h.g.coords[h.pos];
 x.painted=x.painted.filter(([id])=>{const [cx,cy]=h.g.coords[id];return Math.abs(cx-px)+Math.abs(cy-py)<=1});
 x.v='synthetic-v68-local-paint-uncertain-remote-receipt';x.events_delta=events;return x;}
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
function audit(dir){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 check(mf.schema==='driv3r/corpus_factory/observation-quality-v68/1'&&mf.synthetic_not_native_CyberSand,'schema or native claim');
 check(mf.host_sha256===H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),'host source wrong');
 check(mf.parent_v67_sha256===H(fs.readFileSync(path.join(__dirname,'remote_abrasion_v67.js'))),'prior source wrong');
 check(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'reinspection_quality_v68.js'))),'generator source wrong');
 for(const [name,v] of Object.entries(mf.files)){const b=fs.readFileSync(path.join(dir,name));check(H(b)===v.sha256&&b.length===v.bytes,'hash/bytes '+name)}
 let episodes=0,decisions=0,physical=0,stops=0,avoided=0,gap=0,verifiedCount=0,zeroMask=0,replays=0;
 const geometries=new Map(),groups=new Map(),noSignalHashes=new Map();
 for(const split of splits){const policies=streamRows(path.join(dir,filename(split))),certs=read(path.join(dir,split+'_observation_quality_certificates_EVALUATOR_ONLY.jsonl'));
  check(certs.length%80===0,'80-arm split cardinality');
  for(let id=0;id<certs.length;id++){
   const c=certs[id],chunk=policies.next();check(!chunk.done,'missing stream episode');const e=chunk.value,steps=e.steps;check(c.split===split&&c.episode===id&&steps.length>c.critical_step,'provenance index');
   check(c.threshold===(Math.floor((id%80)/40)+1),'threshold order');
   check(JSON.stringify(c.sequence)===JSON.stringify(seqs[Math.floor((id%40)/4)]),'sequence order');
   check(c.mode===modes[id%4],'mask order');
   const w=host.scenario(split,c.source_world_index);check(c.geometry_sha256===w.geometry_sha256&&c.topology_seed===w.seed&&w.s===c.start&&w.goal===c.world_goal&&w.goal2===c.world_goal2,'world identity');
   if(!geometries.has(w.geometry_sha256))geometries.set(w.geometry_sha256,split);
   check(geometries.get(w.geometry_sha256)===split,'split overlap');
   const h=host.start(w,'positive',false),counts={A:0,B:0};let gotCoupon=false,gotReset=false,reportSeen=false,remote=0,physicalCount=0,reportAt=-1,criticalSnap=null;
   let afterReportSteps=0;
   for(let k=0;k<steps.length;k++){
    const s=steps[k];let expected=[];
    if(h.accepted===1){check(!gotCoupon,'duplicate coupon');gotCoupon=true;
     if(c.threshold===1){check(h.painted[w.s]==='A','no coupon paint');delete h.painted[w.s]}
     expected.push([h.accepted,'LOCAL_PHYSICAL_COUPON',c.threshold===1?'ABRADED':'SURVIVED']);}
    if(h.accepted===2){check(!gotReset,'double coupon reset');gotReset=true;delete h.painted[w.s];expected.push([h.accepted,'COUPON_FIXTURE_RESET','A']);}
    const due=c.remote_schedule.filter(z=>z.at===h.accepted);
    for(const ev of due){check(remote<3&&ev.k===[8,21,34][remote],'shock timing');
      check(ev.material===(c.sequence[remote]||null),'wrong remote event');
      if(ev.material){const m=ev.material,t=m==='A'?w.goal:w.goal2;counts[m]++;
       if(counts[m]===c.threshold&&h.painted[t]===m){delete h.painted[t];physicalCount++;}}
      expected.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);remote++;}
    if(h.accepted===c.report_at_accepted){check(!reportSeen,'double report');reportSeen=true;reportAt=k;
      for(const m of ['A','B']){const yes=c.mode==='BOTH'||c.mode===m+'_ONLY';const truth=h.painted[m==='A'?w.goal:w.goal2]===m;
       expected.push([h.accepted,'REMOTE_MATERIAL_CHECK_'+m,yes?(truth?'INTACT':'ERODED'):'NO_VERIFIED_RESULT']);}}
    const observed=localObservation(h,expected);check(H(s.input)===H(observed),'observation mismatch '+split+'/'+id+'/'+k);
    check(observed.legal.includes(s.action)&&host.legal(h).includes(s.action),'illegal advertised action');
    check(s.weight===s.focus_loss_weight&&[0,1].includes(s.weight),'weight integrity');
    if(k<c.critical_step)check(s.weight===0,'expert prefix leakage');
    if(k===c.critical_step){criticalSnap=cp(h);check(s.weight===1,'critical label zeroed');check(k-reportAt>=67,'report too recent');
     check(expected.length===0,'critical had new event');check(H(s.input)===c.critical_input_sha256,'critical hash wrong');check(s.action===c.initial_supervised_action,'initial supervised action wrong');}
    check(host.apply(h,s.action).accepted,'host refused action');decisions++;
   }
   check(gotCoupon&&gotReset&&reportSeen&&remote===3,'missing physical schedules');
   check(criticalSnap&&h.done&&h.success,'host terminal failure');
   const broken=['A','B'].filter(m=>criticalSnap.painted[m==='A'?w.goal:w.goal2]!==m);
   const known=['A','B'].filter(m=>c.mode==='BOTH'||c.mode===m+'_ONLY');
   const unknown=['A','B'].filter(m=>!known.includes(m));
   const planned=['A','B'].filter(m=>!known.includes(m)||broken.includes(m));
   check(H(broken)===H(c.physical_broken)&&H(known)===H(c.reported_known)&&H(unknown)===H(c.unknown)&&H(planned)===H(c.planned_targets),'certificate contradicts reconstructed physics');
   const teacher=plan(criticalSnap,planned),always=plan(criticalSnap,['A','B']),oracle=plan(criticalSnap,broken),stop=plan(criticalSnap,[]);
   check(H(steps.slice(c.critical_step).map(z=>z.action))===H(teacher.actions),'teacher planned with hidden oracle or wrong route');
   check(teacher.success&&always.success&&oracle.success,'physical control failed');
   check(stop.success===c.stop_only_success&&always.actions.length===c.always_reinspect_actions&&
     c.always_reinspect_success&&oracle.actions.length===c.privileged_oracle_actions&&
     c.teacher_continuation_actions===teacher.actions.length&&
     c.redundant_inspection_actions===teacher.actions.length-oracle.actions.length,'control costs incorrect');
   check(c.remote_eroded_count===physicalCount,'remote erosion count wrong');
   check(c.episode_input_action_sha256===H(steps.map(z=>[z.input,z.action,z.weight])),'row hash wrong');
   const gkey=split+':'+c.world_index;if(!groups.has(gkey))groups.set(gkey,{current:new Set(),actions:new Set(),modes:new Set(),physical:new Map()});
   const g=groups.get(gkey);g.current.add(c.critical_input_sha256);g.actions.add(c.initial_supervised_action);g.modes.add(c.mode);
   const physkey=H([c.threshold,c.sequence]);const prev=g.physical.get(physkey);
   if(prev)check(prev===H(broken),'observation mask changed physical outcome');else g.physical.set(physkey,H(broken));
   // ZERO telemetry: all ten remote physical sequences produce exactly the same
   // model-visible history for each threshold/world; supervision must match.
   if(c.mode==='NONE'){const key=split+':'+c.world_index+':'+c.threshold;const sig=H(steps.slice(0,c.critical_step).map(s=>s.input));
     if(noSignalHashes.has(key))check(noSignalHashes.get(key)===sig,'hidden damage leaked to NONE observation');else noSignalHashes.set(key,sig)}
   episodes++;replays++;stops+=Number(stop.success);avoided+=always.actions.length-teacher.actions.length;
   gap+=teacher.actions.length-oracle.actions.length;verifiedCount+=Number(c.mode==='BOTH');
   zeroMask+=steps.slice(c.critical_step+1).filter(s=>s.weight===0).length;
  }
  check(policies.next().done,'unexpected additional stream rows');
 }
 for(const [id,g] of groups){check(g.current.size===1&&g.actions.size===3&&g.modes.size===4&&g.physical.size===20,'insufficient crossed choices '+id)}
 check(mf.total_episodes===episodes&&mf.total_decisions===decisions&&mf.stop_only_success_rate===stops/episodes,'manifest totals');
 check(Math.abs(mf.mean_actions_avoided_vs_always_reinspect-avoided/episodes)<1e-9,'mean avoided wrong');
 check(Math.abs(mf.mean_unavoidable_ambiguity_cost_vs_privileged_oracle-gap/episodes)<1e-9,'oracle gap wrong');
 check(new Set(geometries.values()).size===3,'split coverage absent');
 return {PASS:true,episodes,independent_host_replayed_actions:decisions,split_exclusive_geometries:geometries.size,
  crossed_80_arm_groups:groups.size,no_signal_identical_history_groups:noSignalHashes.size,
  synthetic_verified_episodes:verifiedCount,stop_only_successes:stops,
  action_savings_vs_always_reinspect:avoided,ambiguity_overhead_vs_privileged_oracle:gap,
  mean_savings:avoided/episodes,mean_ambiguity_gap:gap/episodes,
  masked_train_followup_decisions:zeroMask,all_oracle_expert_episodes_successful:true};
}
if(require.main===module){try{check(process.argv[2],'usage: node test_reinspection_quality_v68.js CORPUS_DIR');console.log(JSON.stringify(audit(path.resolve(process.argv[2])),null,2))}catch(e){console.error(e.stack);process.exitCode=1}}
module.exports={audit};
