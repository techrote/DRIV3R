#!/usr/bin/env node
'use strict';
/* Independent v72 synthetic-host auditor. Does NOT import the v72 generator,
 * teacher/planner, or trust certificates for physical state and actions. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const chk=(x,m)=>{if(!x)throw Error('V72 AUDIT '+m);};
function* lines(file){const fd=fs.openSync(file,'r'),b=Buffer.alloc(1<<20);let pending='';try{
 while(true){const n=fs.readSync(fd,b,0,b.length,null);if(!n)break;const arr=(pending+b.toString('utf8',0,n)).split('\n');pending=arr.pop();for(const s of arr)if(s)yield JSON.parse(s);}
 if(pending.trim())yield JSON.parse(pending);}finally{fs.closeSync(fd);}}
const stream=s=>s==='TRAIN'?'TRAIN_physical_rewitness_stream.jsonl':s+'_EVALUATOR_ONLY_physical_rewitness_stream.jsonl';
const cert=s=>s+'_physical_rewitness_certificates_EVALUATOR_ONLY.jsonl';
function sense(h,queue){const z=host.input(h),[x,y]=h.g.coords[h.pos];z.painted=z.painted.filter(([i])=>{const [a,b]=h.g.coords[i];return Math.abs(x-a)+Math.abs(y-b)<=1;});
 z.v='synthetic-v72-physical-rewitness';z.events_delta=queue.splice(0);return z;}
function read(physical,inverted){return inverted?(physical==='INTACT'?'ERODED':'INTACT'):physical;}
function physicallySample(h,w){return {clock:h.accepted,read:{A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
function firstMove(h,target){const d=host.dist(h.g,target,-1);let ns=h.g.adj[h.pos].filter(([i])=>d[i]===d[h.pos]-1).sort((a,b)=>a[0]-b[0]);chk(ns.length>0,'no next move');let [nx,ny]=h.g.coords[ns[0][0]],[x,y]=h.g.coords[h.pos];return 'MOVE_'+(nx>x?'E':nx<x?'W':ny>y?'S':'N');}
function firstTarget(h,targets){if(!targets.length)return null;if(targets.length===1)return targets[0];const w=h.w,d=(a,b)=>host.dist(h.g,b,-1)[a];return d(h.pos,w.goal)+d(w.goal,w.goal2)<=d(h.pos,w.goal2)+d(w.goal2,w.goal)?'A':'B';}
function auditEpisode(e,c,split,row){
 const w=host.scenario(split,c.source_world_index),h=host.start(w,'positive',false),queue=[];
 chk(w.geometry_sha256===c.geometry_sha256,'world fingerprint '+split+'/'+row);
 const nWitness=c.witness==='BOTH'?2:c.witness==='NONE'?0:1;
 const start=c.critical_step-112-2*nWitness, couponStart=start+74,couponEnd=couponStart+2*nWitness;
 const counts={A:0,B:0},sample={},remote={A:[],B:[]},witness={};let faultClock=null,erasures=0,broken=null,labels=null,first=null;
 chk(start>4&&e.steps.length>c.critical_step,'invalid episode boundaries');
 chk(e.steps[0].action==='PAINT_A'&&e.steps.at(-1).action==='STOP','invalid prelude/terminal');
 for(let i=0;i<e.steps.length;i++){
  const s=e.steps[i];
  // The sensor/report schedule is reconstructed from the host accepted-action
  // clock. Extra locally witnessed PAINT/ERASE actions must not shift k.
  let k=null,wi=null;
  if(i>=start&&i<c.critical_step){
   if(i<couponStart)k=i-start;
   else if(i<couponEnd)wi=i-couponStart;
   else k=i-start-2*nWitness;
   if(k!==null){
    chk(k>=0&&k<112,'detour step out of range');
    if([8,21,62].includes(k)){
     const m=c.sequence[[8,21,62].indexOf(k)]||null;if(m){counts[m]++;const p=m==='A'?w.goal:w.goal2;
      if(counts[m]===c.threshold&&h.painted[p]===m){delete h.painted[p];erasures++;}}
     queue.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);
    }
    if(k===38||k===72)sample[k]=physicallySample(h,w);
    if(k===48){faultClock=h.accepted;queue.push([h.accepted,'TELEMETRY_FAULT_DOMAIN',c.fault.domain]);}
    if(k===80||k===90){const origin=k===80?72:38;for(const m of ['A','B']){
     const pf=origin===72&&c.fault.domain==='COMMON_BUS'&&c.fault.flip;
     const sf=origin===72&&c.fault.flip;
     const primary=read(sample[origin].read[m],c.calibration[m]!==pf);
     const secondaryBase=c.fault.domain==='COMMON_BUS'?c.calibration[m]:!c.calibration[m];
     const secondary=read(sample[origin].read[m],secondaryBase!==sf);
     for(const [ch,value] of [['PRIMARY',primary],['SECONDARY',secondary]]){
      const r=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,ch,sample[origin].clock,value];queue.push(r);remote[m].push(r);}
    }}
   }
   if(wi!==null){const materials=c.witness==='BOTH'?['A','B']:[c.witness],m=materials[Math.floor(wi/2)];
    if(wi%2===0){chk(s.action==='PAINT_'+m&&h.pos===w.s&&h.painted[w.s]===undefined,'witness must be physical hub paint');}
    else chk(s.action==='ERASE'&&h.painted[w.s]===m,'witness must remove local coupon');
   }
  }
  if(i===c.critical_step){
   broken=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);labels={};const targets=[];
   for(const m of ['A','B']){
    const canTrustAfter=c.fault.domain==='ISOLATED_SECONDARY'||!!witness[m];
    const candidates=remote[m].filter(r=>r[2]==='PRIMARY'&&(canTrustAfter||r[3]<faultClock)).sort((a,b)=>b[3]-a[3]);
    const latest=candidates[0];let state='UNKNOWN';
    if(latest){const inversion=latest[3]<faultClock||c.fault.domain==='ISOLATED_SECONDARY'?c.calibration[m]:witness[m].code==='ERODED';
     const p=read(latest[4],inversion);
     state=p==='ERODED'?'ERODED':c.schedule.some(z=>z.clock>latest[3])?'UNKNOWN':'INTACT';}
    labels[m]={state,measurement_clock:latest?latest[3]:null,physically_witnessed:!!witness[m]};
    if(state!=='INTACT')targets.push(m);
   }
   chk(H(labels)===H(c.teacher_labels),'teacher labels do not follow physical witnesses');
   chk(H(targets)===H(c.targets),'teacher target list mismatch');
   chk(broken.every(m=>targets.includes(m)),'teacher ignores physically damaged goal');
   const firstMaterial=firstTarget(h,targets);
   first=firstMaterial?firstMove(h,firstMaterial==='A'?w.goal:w.goal2):'STOP';
   chk(s.action===first,'first action not conservative shortest choice');
   chk(H(broken)===H(c.actual_broken),'physical damage certificate mismatch');
   chk(c.fault_clock===faultClock&&c.sampling_clocks.early===sample[38].clock&&c.sampling_clocks.late===sample[72].clock,'source clock mismatch');
   chk(c.physical_erasures===erasures,'unverified erasures');
   const late=Object.fromEntries(['A','B'].map(m=>[m,remote[m].filter(r=>r[2]==='PRIMARY').sort((a,b)=>b[3]-a[3])[0][4]]));
   chk(H(late)===H(c.late_primary_raw),'late primary receipt mismatch');
   chk(c.witness_actions===2*nWitness,'witness cost false');
  }
  const obs=sense(h,queue);
  chk(H(obs)===H(s.input),'input mismatch '+split+'/'+row+'/'+i);
  chk(obs.legal.includes(s.action),'illegal action '+split+'/'+row+'/'+i);
  chk(s.weight===s.focus_loss_weight&&[0,1].includes(s.weight),'invalid loss weights');
  chk(i>=c.critical_step||s.weight===0,'prefix supervised');
  chk(i!==c.critical_step||s.weight===1,'critical unsupervised');
  chk(host.apply(h,s.action).accepted,'host action rejected');
  if(i===0){if(c.threshold===1)delete h.painted[w.s];queue.push([h.accepted,'LOCAL_PHYSICAL_COUPON',c.threshold===1?'ABRADED':'SURVIVED']);}
  if(i===1){delete h.painted[w.s];queue.push([h.accepted,'COUPON_FIXTURE_RESET','A']);}
  if(i<start&&s.action==='PAINT_A'&&h.pos===w.goal){
   chk(h.painted[w.goal]==='A','prelude A physical failure');queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_A','PRIMARY',c.calibration.A?'ERODED':'INTACT']);}
  if(i<start&&s.action==='PAINT_B'&&h.pos===w.goal2){
   chk(h.painted[w.goal2]==='B','prelude B physical failure');queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_B','PRIMARY',c.calibration.B?'ERODED':'INTACT']);}
  if(i>=couponStart&&i<couponEnd&&(i-couponStart)%2===0){
   const materials=c.witness==='BOTH'?['A','B']:[c.witness],m=materials[Math.floor((i-couponStart)/2)];
   chk(h.pos===w.s&&h.painted[w.s]===m,'invalid physical coupon');
   const inverted=c.calibration[m]!==Boolean(c.fault.domain==='COMMON_BUS'&&c.fault.flip);
   const code=read('INTACT',inverted),event=[h.accepted,'LOCAL_POSTFAULT_WITNESS_'+m,'PRIMARY',code];queue.push(event);
   witness[m]={clock:h.accepted,code};
  }
 }
 chk(h.done&&h.success&&c.terminal_success,'terminal physical objective failed');
 chk(H(witness)===H(c.witnessed),'witness provenance/codes mismatch');
 chk(H(e.steps.map(z=>[z.input,z.action,z.weight]))===c.episode_sha256,'episode digest mismatch');
 chk(H(e.steps[c.critical_step].input)===c.critical_sha256,'critical fingerprint mismatch');
 chk(e.steps[c.critical_step].action===c.teacher_first_action,'first-action certificate mismatch');
 chk(e.steps.length-c.critical_step===c.teacher_actions,'teacher length mismatch');
 chk(c.oracle_actions<=c.always_actions,'oracle baseline longer than always');
 return {decisions:e.steps.length,critical:c.critical_sha256,action:first,broken:H(broken),geometry:w.geometry_sha256,successful:true};
}
function audit(dir){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 chk(mf.schema==='driv3r/corpus_factory/physical-rewitness-v72/1'&&mf.synthetic_not_native_CyberSand,'schema');
 for(const [name,src] of [['host_sha256','unseen_topology_v44.js'],['parent_v70_sha256','cross_calibrated_receipt_v70.js'],['generator_sha256','physical_rewitness_v72.js']]){
  chk(mf[name]===H(fs.readFileSync(path.join(__dirname,src))),'source fingerprint '+src);}
 for(const [n,meta] of Object.entries(mf.files)){const b=fs.readFileSync(path.join(dir,n));chk(b.length===meta.bytes&&H(b)===meta.sha256,'shard hash '+n);}
 const worlds=new Set(),stats={};let episodes=0,decisions=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const a=lines(path.join(dir,stream(split)))[Symbol.iterator](),b=lines(path.join(dir,cert(split)))[Symbol.iterator]();
  const groups=new Map(),local=new Set();let n=0,d=0;
  while(true){const x=a.next(),y=b.next();chk(x.done===y.done,'stream/certificate cardinality');if(x.done)break;
   const c=y.value,r=auditEpisode(x.value,c,split,n);chk(c.split===split&&c.episode===n,'provenance');
   const group=[c.world_index,c.threshold,JSON.stringify(c.sequence)].join('|');if(!groups.has(group))groups.set(group,[]);groups.get(group).push(r);
   worlds.add(r.geometry);local.add(r.geometry);n++;d+=r.decisions;
  }
  let contrasted=0,max=0;for(const [k,rows] of groups){chk(rows.length===64,'group cardinality '+k);
   chk(new Set(rows.map(z=>z.critical)).size===1,'critical observations diverge '+k);
   chk(new Set(rows.map(z=>z.broken)).size===1,'different physical state in information-only group');
   const n=new Set(rows.map(z=>z.action)).size;if(n>=2)contrasted++;max=Math.max(max,n);}
  const m=mf.splits[split];chk(n===m.episodes&&d===m.decisions&&groups.size===m.counterfactual_64_arm_groups&&local.size===m.worlds,'split counts');
  chk(contrasted===m.groups_with_different_critical_actions&&max===m.maximum_distinct_critical_actions,'contrast counts');
  stats[split]={worlds:local.size,episodes:n,decisions:d,contrasted_groups:contrasted};episodes+=n;decisions+=d;
 }
 chk(worlds.size===Object.values(mf.splits).reduce((a,s)=>a+s.worlds,0),'cross-split geometry overlap');
 chk(episodes===mf.total_episodes&&decisions===mf.total_decisions,'global counts');
 const r={result:'PASS',synthetic_not_native_CyberSand:true,episodes,decisions,unique_worlds:worlds.size,splits:stats};console.log(JSON.stringify(r,null,2));return r;
}
if(require.main===module){chk(process.argv[2],'usage node test_physical_rewitness_host_v72.js DIR');audit(path.resolve(process.argv[2]));}
module.exports={audit,auditEpisode};
