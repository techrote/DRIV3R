#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v71 independent chronology and host replay.
 * Does not import v71 generator, labels, planners or replay certificates as
 * trusted physical evidence. Only the pre-existing v44 host is imported. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(p,m)=>{if(!p)throw Error('V71 AUDIT '+m)};
function* jsonLines(p){const fd=fs.openSync(p,'r'),buf=Buffer.alloc(1<<20);let rest='';try{
 while(true){const n=fs.readSync(fd,buf,0,buf.length,null);if(n===0)break;const fields=(rest+buf.toString('utf8',0,n)).split('\n');rest=fields.pop();for(const x of fields)if(x)yield JSON.parse(x);}
 if(rest.trim())yield JSON.parse(rest);
}finally{fs.closeSync(fd)}}
const splits=['TRAIN','DEV','HELDOUT'],seqs=[[],['A','A','A'],['B','B','B'],['A','B','B']];
const file=s=>s==='TRAIN'?'TRAIN_common_mode_fault_stream.jsonl':s+'_EVALUATOR_ONLY_common_mode_fault_stream.jsonl';
const cert=s=>s+'_common_mode_fault_certificates_EVALUATOR_ONLY.jsonl';
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E',MOVE_E:'MOVE_W'};
function sense(h,queue){const inp=host.input(h),[x,y]=h.g.coords[h.pos];inp.painted=inp.painted.filter(([i])=>{const [a,b]=h.g.coords[i];return Math.abs(x-a)+Math.abs(y-b)<=1});inp.v='synthetic-v71-common-mode-fault';inp.events_delta=queue.splice(0);return inp}
function measurement(h,w){return {clock:h.accepted,read:{A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}}}
function expectedReading(m,raw,c,channel,late){let inv=(channel==='PRIMARY'||c.domain==='COMMON_BUS')?c.calibration[m]:!c.calibration[m];const after=late&&c.flip&&((c.domain==='COMMON_BUS')||(channel==='SECONDARY'&&c.domain==='ISOLATED_SECONDARY'));
 if(after)inv=!inv;return inv?(raw==='INTACT'?'ERODED':'INTACT'):raw;}
function firstMove(h,target){const d=host.dist(h.g,target,-1);const [next]=h.g.adj[h.pos].filter(([i])=>d[i]===d[h.pos]-1).sort((a,b)=>a[0]-b[0])[0]||[];check(next!==undefined,'route absent');const [x,y]=h.g.coords[h.pos],[a,b]=h.g.coords[next];return 'MOVE_'+(a>x?'E':a<x?'W':b>y?'S':'N')}
function pickOrder(h,w,targets){if(targets.length<2)return targets;const d=(a,b)=>host.dist(h.g,b,-1)[a],idx=m=>m==='A'?w.goal:w.goal2;
 const a=d(h.pos,w.goal)+d(w.goal,w.goal2),b=d(h.pos,w.goal2)+d(w.goal2,w.goal);return a<=b?['A','B']:['B','A'];}
function auditEpisode(e,c,split,row){
 const w=host.scenario(split,c.source_world_index);check(w.geometry_sha256===c.geometry_sha256,'source world/hash mismatch '+row);
 const h=host.start(w,'positive',false),queue=[],counts={A:0,B:0},sampling={},remote={A:[],B:[]},grounded={},warnings=[];
 let detourStart=c.critical_step-112,failures=0,firstAction=null,observedFault=null,brokenAtCritical=null,targetsAtCritical=null;
 check(Number.isInteger(detourStart)&&detourStart>4&&e.steps.length>c.critical_step,'episode length');
 check(e.steps[0].action==='PAINT_A'&&e.steps[e.steps.length-1].action==='STOP','prelude/terminal action');
 for(let i=0;i<e.steps.length;i++){
  const s=e.steps[i];
  if(i>=detourStart&&i<c.critical_step){
   const k=i-detourStart;
   if([8,21,62].includes(k)){
    const j=[8,21,62].indexOf(k),m=c.sequence[j]||null;
    check(c.schedule[j].k===k&&c.schedule[j].at===h.accepted&&c.schedule[j].material===m,'shock chronology');
    if(m){counts[m]++;const t=m==='A'?w.goal:w.goal2;
     if(counts[m]===c.threshold&&h.painted[t]===m){delete h.painted[t];failures++;}}
    queue.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);warnings.push(h.accepted);
   }
   if(k===38||k===72)sampling[k]=measurement(h,w);
   if(k===48){observedFault=h.accepted;queue.push([h.accepted,'TELEMETRY_FAULT_DOMAIN',c.domain]);}
   if(k===80||k===90){const origin=k===80?72:38;
    for(const m of ['A','B']){
     if(c.mode[m]==='MISSING'){if(k===80)queue.push([h.accepted,'REMOTE_MATERIAL_CHECK_'+m,'NO_VERIFIED_RESULT']);continue;}
     const meas=sampling[origin];for(const ch of ['PRIMARY','SECONDARY']){
      const r=expectedReading(m,meas.read[m],c,ch,origin===72),event=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,ch,meas.clock,r];
      queue.push(event);remote[m].push(event);
     }
    }
   }
  }
  if(i===c.critical_step){
   brokenAtCritical=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
   const decoded={};targetsAtCritical=[];
   for(const m of ['A','B']){
    const valid=remote[m].filter(r=>r[2]==='PRIMARY'&&(c.domain!=='COMMON_BUS'||r[3]<observedFault)).sort((a,b)=>b[3]-a[3]);
    const latest=valid[0];let state='UNKNOWN';
    if(latest){const code=grounded[m]?(latest[4]==='INTACT'?'ERODED':'INTACT'):latest[4];
     state=code==='ERODED'?'ERODED':warnings.some(time=>time>latest[3])?'UNKNOWN':'INTACT';}
    decoded[m]={state,clock:latest?latest[3]:null};if(state!=='INTACT')targetsAtCritical.push(m);
   }
   check(H(decoded)===H(c.teacher_labels),'teacher decoded privileged or stale labels '+row);
   check(H(targetsAtCritical)===H(c.targets),'teacher target mismatch '+row);
   check(brokenAtCritical.every(m=>targetsAtCritical.includes(m)),'goal-loss policy leak '+row);
   const order=pickOrder(h,w,targetsAtCritical);
   firstAction=order.length?firstMove(h,order[0]==='A'?w.goal:w.goal2):'STOP';
   check(s.action===firstAction,'critical action not a conservative shortest-route decision '+row);
   check(H(brokenAtCritical)===H(c.actual_broken),'physical broken certificate mismatch '+row);
   check(c.fault_at===observedFault&&sampling[38].clock===c.sampling_clocks.early&&sampling[72].clock===c.sampling_clocks.late,'clock mismatch');
  }
  const inp=sense(h,queue);
  check(H(inp)===H(s.input),'observation mismatch '+split+'/'+row+'/'+i);
  check(inp.legal.includes(s.action),'illegal action '+split+'/'+row+'/'+i);
  check(s.weight===s.focus_loss_weight&&[0,1].includes(s.weight),'invalid loss weight');
  check(i>=c.critical_step||s.weight===0,'prefix supervised');
  check(i!==c.critical_step||s.weight===1,'critical masked');
  check(host.apply(h,s.action).accepted,'synthetic host rejected action '+split+'/'+row+'/'+i);
  if(i===0){if(c.threshold===1)delete h.painted[w.s];queue.push([h.accepted,'LOCAL_PHYSICAL_COUPON',c.threshold===1?'ABRADED':'SURVIVED']);}
  if(i===1){delete h.painted[w.s];queue.push([h.accepted,'COUPON_FIXTURE_RESET','A']);}
  if(i<detourStart&&s.action==='PAINT_A'&&h.pos===w.goal){
   check(h.painted[w.goal]==='A','physical A failed');
   grounded.A=c.calibration.A;queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_A','PRIMARY',grounded.A?'ERODED':'INTACT']);
   queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_A','SECONDARY',c.domain==='COMMON_BUS'?(grounded.A?'ERODED':'INTACT'):(grounded.A?'INTACT':'ERODED')]);
  }
  if(i<detourStart&&s.action==='PAINT_B'&&h.pos===w.goal2){
   check(h.painted[w.goal2]==='B','physical B failed');
   grounded.B=c.calibration.B;queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_B','PRIMARY',grounded.B?'ERODED':'INTACT']);
   queue.push([h.accepted,'LOCAL_CHANNEL_CHECK_B','SECONDARY',c.domain==='COMMON_BUS'?(grounded.B?'ERODED':'INTACT'):(grounded.B?'INTACT':'ERODED')]);
  }
 }
 check(h.done&&h.success&&c.terminal_success,'terminal goal failure '+row);
 check(failures===c.physical_erasures,'physical erasure count');
 check(H(e.steps.map(s=>[s.input,s.action,s.weight]))===c.episode_sha256,'episode bound signature');
 check(H(e.steps[c.critical_step].input)===c.critical_input_sha256,'critical input digest');
 check(e.steps[c.critical_step].action===c.teacher_first_action,'first label certificate');
 check(e.steps.length-c.critical_step===c.teacher_actions,'teacher action count');
 check(c.always_success&&c.oracle_actions<=c.always_actions,'scorer bounds');
 return {decisions:e.steps.length,critical_input:c.critical_input_sha256,first:firstAction,broken:H(brokenAtCritical),geometry:w.geometry_sha256,targets:targetsAtCritical,common:c.domain==='COMMON_BUS',stop_failed:brokenAtCritical.length>0};
}
function audit(dir){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 check(mf.schema==='driv3r/corpus_factory/common-mode-fault-v71/1'&&mf.synthetic_not_native_CyberSand,'schema or synthetic status');
 check(H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js')))===mf.host_sha256,'host source hash');
 check(H(fs.readFileSync(path.join(__dirname,'cross_calibrated_receipt_v70.js')))===mf.parent_v70_sha256,'parent source hash');
 check(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'common_mode_fault_v71.js'))),'generator source hash');
 for(const [name,entry] of Object.entries(mf.files)){
  const file=path.join(dir,name),buf=fs.readFileSync(file);check(buf.length===entry.bytes&&H(buf)===entry.sha256,'shard hash '+name);
 }
 let total=0,actions=0,distinctWorlds=new Set(),stats={};
 for(const split of splits){const it=jsonLines(path.join(dir,file(split)))[Symbol.iterator](),ci=jsonLines(path.join(dir,cert(split)))[Symbol.iterator]();
  const meta=mf.splits[split],groups=new Map(),worlds=new Set();let n=0,d=0,successful=0,common=0;
  while(true){const a=it.next(),b=ci.next();check(a.done===b.done,'stream and certificate size mismatch');if(a.done)break;
   const r=auditEpisode(a.value,b.value,split,n),c=b.value;
   check(c.split===split&&c.episode===n,'certificate provenance');
   worlds.add(r.geometry);distinctWorlds.add(r.geometry);
   const group=[c.world_index,c.threshold,JSON.stringify(c.sequence)].join('|');if(!groups.has(group))groups.set(group,[]);groups.get(group).push(r);
   n++;d+=r.decisions;successful++;if(r.common)common++;
  }
  for(const [k,gr] of groups){check(gr.length===36,'counterfactual size '+split+'/'+k);
   check(new Set(gr.map(x=>x.critical_input)).size===1,'identical current input');
   check(new Set(gr.map(x=>x.broken)).size===1,'changed physical truth under telemetry intervention');}
  check(n===meta.episodes&&d===meta.decisions&&successful===meta.terminal_successes,'split counts');
  check(worlds.size===meta.worlds&&groups.size===meta.counterfactual_36_arm_groups,'geometry/groups counts');
  check(common===meta.fault_common_arm_count,'common count');
  stats[split]={worlds:worlds.size,episodes:n,decisions:d,groups:groups.size};total+=n;actions+=d;
 }
 check(distinctWorlds.size===Object.values(mf.splits).reduce((a,b)=>a+b.worlds,0),'topology split leakage');
 check(total===mf.total_episodes&&actions===mf.total_decisions,'global totals');
 console.log(JSON.stringify({result:'PASS',synthetic_not_native_CyberSand:true,episodes:total,actions,split:stats,unique_topologies:distinctWorlds.size},null,2));return stats;
}
if(require.main===module){check(process.argv[2],'usage: node test_common_mode_fault_v71.js CORPUS_DIRECTORY');audit(path.resolve(process.argv[2]));}
module.exports={audit};
