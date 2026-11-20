#!/usr/bin/env node
'use strict';
/* v76 independent physical-host replay. Intentionally does not import the
 * v73 generator, its teacher, its active-probe policy or its certificates as
 * observations. Expected probe costs reconstructed from host graph distances. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error('V76 AUDIT '+m);};
const SCHEMA='driv3r/corpus_factory/material-specific-risk-v76/1';
const MODES=['NONE','A','B','BOTH'];
const riskCounts={TRAIN:[2,6],DEV:[3,5],HELDOUT:[1,7]};
const stream=s=>s==='TRAIN'?'TRAIN_material_specific_risk_stream.jsonl':s+'_EVALUATOR_ONLY_material_specific_risk_stream.jsonl';
const cert=s=>s+'_material_specific_risk_certificates_EVALUATOR_ONLY.jsonl';
function* lines(file){const fd=fs.openSync(file,'r'),b=Buffer.alloc(1<<20);let rem='';try{for(;;){const n=fs.readSync(fd,b,0,b.length,null);if(!n)break;const rows=(rem+b.toString('utf8',0,n)).split('\n');rem=rows.pop();for(const line of rows)if(line)yield JSON.parse(line);}if(rem.trim())yield JSON.parse(rem);}finally{fs.closeSync(fd);}}
function sense(h,q){const obs=host.input(h),[x,y]=h.g.coords[h.pos];obs.painted=obs.painted.filter(([i])=>{const [a,b]=h.g.coords[i];return Math.abs(x-a)+Math.abs(y-b)<=1;});obs.v='synthetic-v73-active-witness';obs.events_delta=q.splice(0);return obs;}
const materials=mode=>mode==='NONE'?[]:mode==='BOTH'?['A','B']:[mode];
function dist(h,a,b){return host.dist(h.g,b,-1)[a];}
function firstMove(h,target){const d=host.dist(h.g,target,-1),next=h.g.adj[h.pos].filter(([to])=>d[to]===d[h.pos]-1).sort((a,b)=>a[0]-b[0]);ok(next.length>0,'no shortest path');const [x,y]=h.g.coords[h.pos],[nx,ny]=h.g.coords[next[0][0]];return 'MOVE_'+(nx>x?'E':nx<x?'W':ny>y?'S':'N');}
function order(h,targets){if(targets.length<2)return targets;const w=h.w;return dist(h,h.pos,w.goal)+dist(h,w.goal,w.goal2)<=dist(h,h.pos,w.goal2)+dist(h,w.goal2,w.goal)?['A','B']:['B','A'];}
const PLAN_CACHE=new Map();
const HISTORIES=[[],['A','A','A'],['B','B','B'],['A','B','B']];
function independentPlanCosts(w,threshold,domain,passes,riskA,riskB){
 const key=H([w.geometry_sha256,threshold,domain,passes,riskA,riskB]);if(PLAN_CACHE.has(key))return PLAN_CACHE.get(key);
 ok(threshold===1,'threshold');const pa=riskA/8,pb=riskB/8;
 const weights=[(1-pa)*(1-pb),pa*(1-pb),(1-pa)*pb,pa*pb];
 const h=host.start(w,'positive',false),anchor=w.s,totals=Object.fromEntries(MODES.map(m=>[m,0]));
 for(let j=0;j<4;j++)for(const flip of [false,true]){
  void flip;const seq=HISTORIES[j],damage=['A','B'].filter(m=>seq.includes(m));
  for(const mode of MODES){
   const target=domain==='ISOLATED_SECONDARY'?damage:['A','B'].filter(m=>!materials(mode).includes(m)||damage.includes(m));
   const ordered=order(h,target);let cursor=anchor,cost=1;
   for(const m of ordered){const p=m==='A'?w.goal:w.goal2;cost+=dist(h,cursor,p);cursor=p;if(damage.includes(m))cost++;}
   for(const m of materials(mode))cost+=2*passes[m];
   totals[mode]+=cost*weights[j]*0.5;
  }
 }
 let selected='NONE';for(const m of MODES)if(totals[m]<totals[selected]-1e-9)selected=m;
 const result={means:totals,selected};PLAN_CACHE.set(key,result);return result;
}
function invert(code,flip){return flip?(code==='INTACT'?'ERODED':'INTACT'):code;}
function sample(h,w){return {clock:h.accepted,read:{A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};}
function auditOne(e,c,split,idx){
 const w=host.scenario(split,c.source_world_index),h=host.start(w,'positive',false),q=[];
 ok(w.geometry_sha256===c.geometry_sha256,'geometry '+split+'/'+idx);
 const chosen=materials(c.selected),coupon=chosen.reduce((a,m)=>a+2*c.passes[m],0);
 const riskCounts={TRAIN:[2,6],DEV:[3,5],HELDOUT:[1,7]};
 ok(['A','B'].every(m=>riskCounts[split].includes(c.incident_counts[m])),'material-specific risk count invalid');
 ok(c.threshold===1,'threshold must 1');ok(c.fault.domain===['COMMON_BUS','ISOLATED_SECONDARY'][(c.world_index+(split==='DEV'?1:0))%2],'fault domain/world assignment');
 const pilotMask={};const pilotActions=[];
 for(const m of ['A','B']){
  pilotMask[m]=Array.from({length:8},(_,k)=>((k*5+c.world_index*3+(m==='A'?1:4))%8)<c.incident_counts[m]);
  for(let j=0;j<8;j++){
   pilotActions.push({action:'PAINT_'+m,outcome:pilotMask[m][j]?'ABRADED':'SURVIVED',trial:j,material:m});
   if(pilotMask[m][j])pilotActions.push({action:'PAINT_'+(m==='A'?'B':'A')});
   pilotActions.push({action:'ERASE'});
  }
 }
 ok(H(pilotMask)===H(c.local_trial_masks),'forged physical pilot masks');
 let pilotStart=3;
 const start=c.critical_step-112-coupon,choice=start+74,afterCoupon=choice+coupon;
 ok(start>4&&c.decision_step===choice&&c.total_legal_actions===e.steps.length,'boundaries');
 ok(HISTORIES.some(seq=>H(seq)===H(c.sequence)),'unauthorized hidden history');
 const economics=independentPlanCosts(w,c.threshold,c.fault.domain,c.passes,c.incident_counts.A,c.incident_counts.B);
 ok(economics.selected===c.selected,'noncausal probe selection');
 for(const mode of MODES)ok(Math.abs(economics.means[mode]-c.policy_means[mode])<1e-9,'incorrect expected cost '+mode);
 const homeNeighbor=h.g.adj[h.pos][0][0],outMove=firstMove(h,homeNeighbor);
 const inverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
 const maxRisk=Math.max(...riskCounts[split]);
 for(const m of ['A','B'])for(let j=0;j<(maxRisk-c.incident_counts[m])/2;j++)pilotActions.push({action:outMove},{action:inverse[outMove]});
 ok(start>=pilotStart+pilotActions.length+4,'pilot precedes remote goals');
 let counts={A:0,B:0},erasures=0,samples={},faultClock=null,witnessed={},reports={A:[],B:[]},schedule=[],firstChoice=null;
 let physicalTruth=null,criticalDecision=null;
 for(let i=0;i<e.steps.length;i++){
  const r=e.steps[i];let k=null,wi=null;
  if(i>=start&&i<c.critical_step){
   if(i<choice)k=i-start;else if(i<afterCoupon)wi=i-choice;else k=i-start-coupon;
   if(k!==null){ok(k>=0&&k<112,'k mapping');
    if([8,21,62].includes(k)){
     const m=c.sequence[[8,21,62].indexOf(k)]||null;schedule.push({clock:h.accepted,k,material:m});
     if(m){counts[m]++;const p=m==='A'?w.goal:w.goal2;if(counts[m]===c.threshold&&h.painted[p]===m){delete h.painted[p];erasures++;}}
     q.push([h.accepted,'UNVERIFIED_REMOTE_VIBRATION','UNKNOWN']);
    }
    if(k===38||k===72)samples[k]=sample(h,w);
    if(k===48){faultClock=h.accepted;q.push([h.accepted,'TELEMETRY_FAULT_DOMAIN',c.fault.domain]);}
    if(k===80||k===90){const origin=k===80?72:38;for(const m of ['A','B']){
     const primaryFlip=origin===72&&c.fault.domain==='COMMON_BUS'&&c.fault.flip;
     const secondaryFlip=origin===72&&c.fault.flip;
     const p=invert(samples[origin].read[m],Boolean(c.calibration[m])!==Boolean(primaryFlip));
     const secondaryCal=c.fault.domain==='COMMON_BUS'?c.calibration[m]:!c.calibration[m];
     const s=invert(samples[origin].read[m],Boolean(secondaryCal)!==Boolean(secondaryFlip));
     for(const [channel,code] of [['PRIMARY',p],['SECONDARY',s]]){const event=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,channel,samples[origin].clock,code];q.push(event);reports[m].push(event);}
    }}
   }
   if(wi!==null){let rem=wi,m=null,pass=-1;for(const item of chosen){const cost=2*c.passes[item];if(rem<cost){m=item;pass=rem;break;}rem-=cost;}
    ok(m!==null,'coupon out of bounds');const expected=pass%2===0?'PAINT_'+m:'ERASE';ok(r.action===expected&&h.pos===w.s,'coupon nonphysical action');
    if(pass%2===0)ok(h.painted[w.s]===undefined,'hub occupied before coupon');else ok(h.painted[w.s]===m,'hub invalid before erase');
   }
  }
  if(i>=pilotStart&&i<pilotStart+pilotActions.length){
   const p=pilotActions[i-pilotStart];ok(r.action===p.action,'physical pilot action mismatch '+i);
   if(p.trial!==undefined)ok(h.pos===w.s&&h.painted[w.s]===undefined,'physical pilot site occupied');
  }
  if(i===c.critical_step){
   physicalTruth=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
   const states={},targets=[];
   for(const m of ['A','B']){
    const after=c.fault.domain==='ISOLATED_SECONDARY'||!!witnessed[m];
    const valid=reports[m].filter(row=>row[2]==='PRIMARY'&&(after||row[3]<faultClock)).sort((a,b)=>b[3]-a[3]);
    const last=valid[0];let state='UNKNOWN';
    if(last){const inverted=last[3]<faultClock||c.fault.domain==='ISOLATED_SECONDARY'?c.calibration[m]:witnessed[m].code==='ERODED';
     const real=invert(last[4],inverted);state=real==='ERODED'?'ERODED':schedule.some(z=>z.clock>last[3])?'UNKNOWN':'INTACT';
    }
    states[m]={state,measurement_clock:last?last[3]:null,physically_witnessed:!!witnessed[m]};if(state!=='INTACT')targets.push(m);
   }
   ok(H(states)===H(c.teacher_labels)&&H(targets)===H(c.targets),'teacher target mismatch');
   ok(physicalTruth.every(m=>targets.includes(m)),'teacher omitted physical abrasion');
   const route=order(h,targets),expected=route.length?firstMove(h,route[0]==='A'?w.goal:w.goal2):'STOP';
   ok(r.action===expected,'teacher first recovery move not shortest');
   criticalDecision=expected;
  }
  if(i===choice){firstChoice=r.action;ok(r.action===(chosen.length?'PAINT_'+chosen[0]:firstMove(h,h.g.adj[h.pos][0][0])),'first active choice action');}
  // At choice and recovery, learner sees only source-time observation stream.
  const obs=sense(h,q);ok(H(obs)===H(r.input),'model observation mismatch '+split+'/'+idx+'/'+i);
  ok(obs.legal.includes(r.action),'illegal '+split+'/'+idx+'/'+i);
  ok(r.weight===r.focus_loss_weight&&[0,1].includes(r.weight),'weights');
  if(i<choice)ok(r.weight===0,'prefix leaked');
  if(i===choice||i===c.critical_step)ok(r.weight===1,'pivotal decision unsupervised');
  if(i>choice&&i<c.critical_step){
   const isSecond=c.selected==='BOTH'&&i===choice+2*c.passes.A;
   const isMove=i===afterCoupon;
   if(isSecond||isMove)ok(r.weight===1,'secondary intervention unsupervised');else ok(r.weight===0,'unused detour/coupon supervised');
  }
  ok(host.apply(h,r.action).accepted,'host rejected '+split+'/'+idx+'/'+i);
  if(i===0){if(c.threshold===1)delete h.painted[w.s];q.push([h.accepted,'LOCAL_PHYSICAL_COUPON',c.threshold===1?'ABRADED':'SURVIVED']);q.push([h.accepted,'WITNESS_PROCEDURE_PASSES','A',c.passes.A,'B',c.passes.B]);}
  if(i===1){delete h.painted[w.s];q.push([h.accepted,'COUPON_FIXTURE_RESET','A']);}
  if(i>=pilotStart&&i<pilotStart+pilotActions.length){
   const p=pilotActions[i-pilotStart];
   if(p.trial!==undefined){
    if(p.outcome==='ABRADED')delete h.painted[w.s];
    q.push([h.accepted,'LOCAL_STRESS_TRIAL',p.material,p.trial,p.outcome]);
   }
  }
  if(i<start&&r.action==='PAINT_A'&&h.pos===w.goal){ok(h.painted[w.goal]==='A','prelude A failed');q.push([h.accepted,'LOCAL_CHANNEL_CHECK_A','PRIMARY',c.calibration.A?'ERODED':'INTACT']);}
  if(i<start&&r.action==='PAINT_B'&&h.pos===w.goal2){ok(h.painted[w.goal2]==='B','prelude B failed');q.push([h.accepted,'LOCAL_CHANNEL_CHECK_B','PRIMARY',c.calibration.B?'ERODED':'INTACT']);}
  if(i>=choice&&i<afterCoupon){let rem=i-choice,m=null,pass=-1;for(const item of chosen){const len=2*c.passes[item];if(rem<len){m=item;pass=rem;break;}rem-=len;}
   if(pass%2===0){const flipped=c.fault.domain==='COMMON_BUS'&&c.fault.flip;
    const code=invert('INTACT',Boolean(c.calibration[m])!==Boolean(flipped));
    ok(h.painted[w.s]===m,'physical coupon did not paint');q.push([h.accepted,'LOCAL_POSTFAULT_WITNESS_'+m,'PRIMARY',code]);witnessed[m]={clock:h.accepted,code};
   }
  }
 }
 ok(h.done&&h.success&&c.terminal_success,'physical episode unsuccessful');
 ok(firstChoice===c.decision_action&&criticalDecision===e.steps[c.critical_step].action,'decision mismatch');
 ok(H(witnessed)===H(c.witnessed)&&H(physicalTruth)===H(c.actual_broken),'witness/truth certificate mismatch');
 ok(erasures===c.physical_erasures,'erosion mismatch');
 ok(c.fault_clock===faultClock&&samples[38].clock===c.sampling_clocks.early&&samples[72].clock===c.sampling_clocks.late,'host clock');
 ok(H(e.steps[choice].input)===c.decision_sha256&&H(e.steps[c.critical_step].input)===c.critical_sha256,'decision hash');
 ok(H(e.steps.map(s=>[s.input,s.action,s.weight]))===c.episode_sha256,'episode hash');
 return {decisionHash:c.decision_sha256,chosen:c.selected,actions:e.steps.length};
}
function audit(dir){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));ok(mf.schema===SCHEMA&&mf.synthetic_not_native_CyberSand,'schema');
 for(const [k,f] of [['host_sha256','unseen_topology_v44.js'],['parent_v67_sha256','remote_abrasion_v67.js'],['parent_v70_sha256','cross_calibrated_receipt_v70.js'],['parent_v72_sha256','physical_rewitness_v72.js'],['generator_sha256','material_specific_risk_v76.js']])ok(mf[k]===H(fs.readFileSync(path.join(__dirname,f))),'source SHA mismatch '+f);
 for(const [f,meta] of Object.entries(mf.files)){const b=fs.readFileSync(path.join(dir,f));ok(b.length===meta.bytes&&H(b)===meta.sha256,'file SHA mismatch '+f);}
 const geometry=new Set(),statistics={},total={episodes:0,decisions:0};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const e=lines(path.join(dir,stream(split)))[Symbol.iterator](),c=lines(path.join(dir,cert(split)))[Symbol.iterator](),groups=new Map(),local=new Set();
  let episodes=0,decisions=0;const modes={NONE:0,A:0,B:0,BOTH:0};
  for(;;){const a=e.next(),b=c.next();ok(a.done===b.done,'paired shard length');if(a.done)break;
   const x=b.value;ok(x.split===split&&x.episode===episodes,'split/episode provenance');
   const result=auditOne(a.value,x,split,episodes);const key=[x.world_index,JSON.stringify(x.calibration),x.fault.domain].join('|');
   if(!groups.has(key))groups.set(key,[]);groups.get(key).push({...result,passes:x.passes,history:x.sequence,flip:x.fault.flip,risk:x.incident_counts,sourceIndex:x.source_world_index,threshold:x.threshold,domain:x.fault.domain});
   modes[x.selected]++;local.add(x.geometry_sha256);geometry.add(x.geometry_sha256);episodes++;decisions+=result.actions;
  }
  let variedTariff=0,variedRisk=0,variedA=0,variedB=0,uniformWrong=0,uniformRegret=0;
  for(const [name,items] of groups){ok(items.length===128,'incomplete 4 risks x4 tariffs x8 hidden counterfactual '+name);
   const priced=new Map();for(const t of items){const p=t.passes.A+'|'+t.passes.B;
    if(!priced.has(p))priced.set(p,[]);priced.get(p).push(t);}
   ok(priced.size===4&&Array.from(priced.values()).every(v=>v.length===32),'profile group incomplete');
   for(const row of priced.values()){
    const riskGroups=new Map();for(const z of row){const k=z.risk.A+'|'+z.risk.B;if(!riskGroups.has(k))riskGroups.set(k,[]);riskGroups.get(k).push(z);}
    ok(riskGroups.size===4,'missing independently varied risk counterfactuals');
    const selected=new Map();for(const [key,arms] of riskGroups){
     ok(arms.length===8,'hidden history/fault group incomplete');
     ok(new Set(arms.map(z=>z.chosen)).size===1,'hidden future leaks into ex-ante choice');
     ok(new Set(arms.map(z=>z.decisionHash)).size===1,'hidden future leaked into source-time observations');
     selected.set(key,arms[0].chosen);
     const z=arms[0],w=host.scenario(split,z.sourceIndex),uniform=independentPlanCosts(w,z.threshold,z.domain,z.passes,4,4);
     const own=independentPlanCosts(w,z.threshold,z.domain,z.passes,z.risk.A,z.risk.B);if(own.means[uniform.selected]-own.means[own.selected]>1e-9){uniformWrong++;uniformRegret+=own.means[uniform.selected]-own.means[own.selected];}
    }
    const values=riskCounts[split];const k=(a,b)=>a+'|'+b;
    variedRisk+=Number(new Set(selected.values()).size>1);
    for(const v of values){variedA+=Number(selected.get(k(values[0],v))!==selected.get(k(values[1],v)));
     variedB+=Number(selected.get(k(v,values[0]))!==selected.get(k(v,values[1])));}
   }
   variedTariff+=Number(new Set(items.map(z=>z.chosen)).size>1);
  }
  const m=mf.splits[split];ok(m.episodes===episodes&&m.decisions===decisions&&m.terminal_successes===episodes&&m.worlds.length===local.size,'counts');
  ok(m.counterfactual_expected_cost_groups===groups.size*16&&m.groups_with_different_optimal_probe_across_cost_profiles===variedTariff&&m.groups_with_different_optimal_probe_across_risk===variedRisk&&m.groups_with_different_optimal_probe_across_A_risk===variedA&&m.groups_with_different_optimal_probe_across_B_risk===variedB,'group counts');
  ok(H(m.probe_choices)===H(modes),'choice distribution');ok(m.uniform_half_prior_mistakes===uniformWrong&&Math.abs(m.uniform_half_prior_regret-uniformRegret/m.counterfactual_expected_cost_groups)<1e-9,'uniform prior baseline mismatch');
  statistics[split]={worlds:local.size,episodes,decisions,choice_distribution:modes,contrasted_cost_groups:variedTariff,contrasted_risk_groups:variedRisk,A_specific_contrasts:variedA,B_specific_contrasts:variedB,uniform_half_prior_mistakes:uniformWrong,uniform_half_prior_regret:uniformRegret/m.counterfactual_expected_cost_groups};total.episodes+=episodes;total.decisions+=decisions;
 }
 ok(geometry.size===Object.values(statistics).reduce((a,s)=>a+s.worlds,0),'cross-split geometry overlap');
 ok(mf.total_episodes===total.episodes&&mf.total_decisions===total.decisions,'global count');
 const result={result:'PASS',synthetic_not_native_CyberSand:true,...total,splits:statistics};console.log(JSON.stringify(result,null,2));return result;
}
if(require.main===module){ok(process.argv[2],'usage node test_material_specific_risk_host_v76.js CORPUS_DIR');audit(path.resolve(process.argv[2]));}
module.exports={audit,independentPlanCosts};
