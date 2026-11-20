#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v75 — DATA, not a trained policy. A synthetic physical
 * coupon may be chosen endogenously under known action costs and hidden fault/
 * erosion scenarios. Source-time evidence only; synthetic != native CyberSand. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const parent=require('./cross_calibrated_receipt_v70.js');
const v72=require('./physical_rewitness_v72.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(p,m)=>{if(!p)throw Error('V75 '+m)};
const clone=x=>JSON.parse(JSON.stringify(x));
const COUNTS={TRAIN:8,DEV:4,HELDOUT:4},BASES={TRAIN:344000,DEV:354000,HELDOUT:364000};
const RISK_COUNTS={TRAIN:[2,6],DEV:[3,5],HELDOUT:[1,7]};
const HISTORIES=[[],['A','A','A'],['B','B','B'],['A','B','B']];
const CALIBRATIONS=[{A:false,B:false},{A:true,B:true}];
const FAULT_DOMAINS=['COMMON_BUS','ISOLATED_SECONDARY'];
const PROFILES=[{A:1,B:1},{A:1,B:6},{A:6,B:1},{A:6,B:6}];
const MODES=['NONE','A','B','BOTH'];
const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const stream=s=>s==='TRAIN'?'TRAIN_physical_prior_shift_stream.jsonl':s+'_EVALUATOR_ONLY_physical_prior_shift_stream.jsonl';
const cert=s=>s+'_physical_prior_shift_certificates_EVALUATOR_ONLY.jsonl';
function pickWorld(split,i,seen){for(let attempt=0;attempt<100;attempt++){
 const index=BASES[split]+i*167+attempt*3,w=host.scenario(split,index),h=host.start(w,'positive',false),d=p=>host.dist(h.g,p,-1)[w.s];
 if(seen.has(w.geometry_sha256)||d(w.goal)<5||d(w.goal2)<5||parent.pathTo(h,w.goal)[0]===parent.pathTo(h,w.goal2)[0])continue;
 seen.add(w.geometry_sha256);return {w,index,attempt};
 }throw Error('v73 no geometry '+split+'/'+i);}
function sense(h,q){const a=parent.makeSense(h,q);a.v='synthetic-v73-active-witness';return a;}
function act(ctx,cmd,weight=0){const obs=sense(ctx.h,ctx.q);check(obs.legal.includes(cmd),'illegal action '+cmd);ctx.steps.push({input:obs,action:cmd,weight,focus_loss_weight:weight});check(host.apply(ctx.h,cmd).accepted,'rejected '+cmd);}
function go(ctx,p,weight=0){for(const cmd of parent.pathTo(ctx.h,p))act(ctx,cmd,weight);}
function sortTargets(h,targets){if(targets.length<2)return targets;const d=(a,b)=>host.dist(h.g,b,-1)[a],w=h.w;return d(h.pos,w.goal)+d(w.goal,w.goal2)<=d(h.pos,w.goal2)+d(w.goal2,w.goal)?['A','B']:['B','A'];}
function finish(ctx,targets){for(const m of sortTargets(ctx.h,targets)){const p=m==='A'?ctx.h.w.goal:ctx.h.w.goal2;go(ctx,p,1);if(ctx.h.painted[p]!==m){check(ctx.h.painted[p]===undefined,'unexpected paint');act(ctx,'PAINT_'+m,1);}}act(ctx,'STOP',1);}
function inversionCode(real,cal,flipped){return cal!==flipped?(real==='INTACT'?'ERODED':'INTACT'):real;}
function witnesses(mode){return mode==='BOTH'?['A','B']:mode==='NONE'?[]:[mode];}
// The policy selection is ex ante: each hidden physical history and fault flip
// has equal weight. No single episode's hidden future is used to choose its label.
const BASE_COST_CACHE=new Map();
function expectedCosts(w,threshold,calibration,domain,passes,historyAlt,incidentCount){
 check(Number.isInteger(incidentCount)&&incidentCount>=1&&incidentCount<=7,'invalid prior sample count');
 const risk=incidentCount/8;
 const totals=Object.fromEntries(MODES.map(m=>[m,0]));const arms=[];
 for(const history of [[],historyAlt])for(const flip of [false,true])for(const mode of MODES){
  const key=H([w.geometry_sha256,threshold,history,calibration,domain,flip,mode]);
  if(!BASE_COST_CACHE.has(key)){const {certificate:c}=v72.episode(w,threshold,history,calibration,{domain,flip},mode);BASE_COST_CACHE.set(key,c.teacher_actions);}
  const cost=BASE_COST_CACHE.get(key)+witnesses(mode).reduce((sum,m)=>sum+2*passes[m],0);
  totals[mode]+=(history.length?risk:1-risk)*0.5*cost;arms.push({history,flip,mode,cost});
 }
 const means=Object.fromEntries(MODES.map(m=>[m,totals[m]]));
 const selected=MODES.reduce((best,m)=>means[m]<means[best]-1e-9?m:best,'NONE');
 const baselines={none:means.NONE,always_both:means.BOTH,oracle_hindsight:arms.reduce((a,z)=>a+(z.history.length?risk:1-risk)*0.5*z.cost,0)/4};
 return {selected,means,baselines,action_gap:Math.min(...MODES.filter(m=>m!==selected).map(m=>means[m]-means[selected]))};
}
function episode(w,threshold,seq,cal,fault,passes,selected,policy,incidentCount,worldIndex,split){
 const h=host.start(w,'positive',false),ctx={h,q:[],steps:[]},emit=(...e)=>ctx.q.push([h.accepted,...e]),schedule=[];
 act(ctx,'PAINT_A');if(threshold===1)delete h.painted[w.s];emit('LOCAL_PHYSICAL_COUPON',threshold===1?'ABRADED':'SURVIVED');
 // Cost tariff is a historically available intervention procedure, not an
 // evaluator certificate or a contemporaneous first-choice observation.
 emit('WITNESS_PROCEDURE_PASSES','A',passes.A,'B',passes.B);
 const n=h.g.adj[h.pos][0][0],dir=parent.pathTo(h,n)[0];act(ctx,dir);delete h.painted[w.s];emit('COUPON_FIXTURE_RESET','A');act(ctx,reverse[dir]);
 // Eight genuinely executed local stress probes, including material painting
 // and physically erased trial cells. Each probe outcome is observable only in
 // the chronological event stream; no risk score is given to the model.
 const pilotMask=Array.from({length:8},(_,k)=>((k*5+worldIndex*3+1)%8)<incidentCount);
 check(pilotMask.filter(Boolean).length===incidentCount,'pilot samples');
 for(let j=0;j<8;j++){
  act(ctx,'PAINT_A');const abraded=pilotMask[j];
  if(abraded)delete h.painted[w.s];
  emit('LOCAL_STRESS_TRIAL',j,abraded?'ABRADED':'SURVIVED');
  if(abraded)act(ctx,'PAINT_B');
  act(ctx,'ERASE');
  check(h.painted[w.s]===undefined,'pilot fixture not restored');
 }
 // Equalize accepted host clocks across low/high risk worlds in each split.
 // This prevents clock offsets leaking the risk class into durable host state.
 const maxRisk=Math.max(...RISK_COUNTS[split]);
 for(let j=0;j<(maxRisk-incidentCount)/2;j++){act(ctx,dir);act(ctx,reverse[dir]);}
 for(const m of ['A','B']){go(ctx,m==='A'?w.goal:w.goal2);act(ctx,'PAINT_'+m);emit('LOCAL_CHANNEL_CHECK_'+m,'PRIMARY',cal[m]?'ERODED':'INTACT');}
 check(h.stage===2,'material goals not physically complete');go(ctx,w.s);
 const move=parent.pathTo(h,h.g.adj[h.pos][0][0])[0],counts={A:0,B:0},sample={},reports={A:[],B:[]},witnessed={};
 let faultClock=null,erasures=0,decision=-1,secondDecision=null;
 for(let k=0;k<112;k++){
  if([8,21,62].includes(k)){
   const m=seq[[8,21,62].indexOf(k)]||null;schedule.push({clock:h.accepted,k,material:m});
   if(m){counts[m]++;const p=m==='A'?w.goal:w.goal2;if(counts[m]===threshold&&h.painted[p]===m){delete h.painted[p];erasures++;}}
   emit('UNVERIFIED_REMOTE_VIBRATION','UNKNOWN');
  }
  if(k===38||k===72)sample[k]={clock:h.accepted,read:{A:h.painted[w.goal]==='A'?'INTACT':'ERODED',B:h.painted[w.goal2]==='B'?'INTACT':'ERODED'}};
  if(k===48){faultClock=h.accepted;emit('TELEMETRY_FAULT_DOMAIN',fault.domain);}
  if(k===74){
   decision=ctx.steps.length;
   for(const [idx,m] of witnesses(selected).entries()){
    // First PAINT per chosen material is a selected legal physical action.
    // Multiple PAINT/ERASE cycles represent a material-specific tool cost.
    for(let pass=0;pass<passes[m];pass++){
     check(h.pos===w.s&&h.painted[w.s]===undefined,'hub dirty before witness');
     if(idx>0&&pass===0)secondDecision=ctx.steps.length;
     act(ctx,'PAINT_'+m,pass===0?1:0);
     const flipped=fault.domain==='COMMON_BUS'&&fault.flip,code=inversionCode('INTACT',cal[m],flipped);
     witnessed[m]={clock:h.accepted,code};emit('LOCAL_POSTFAULT_WITNESS_'+m,'PRIMARY',code);
     act(ctx,'ERASE',0);check(h.painted[w.s]===undefined,'hub dirty after witness');
    }
   }
  }
  if(k===80||k===90){const src=k===80?72:38;for(const m of ['A','B']){
   const primaryFlip=src===72&&fault.domain==='COMMON_BUS'&&fault.flip;
   const secondaryFlip=src===72&&fault.flip;
   const p=inversionCode(sample[src].read[m],cal[m],primaryFlip);
   const secondaryCal=fault.domain==='COMMON_BUS'?cal[m]:!cal[m];
   const s=inversionCode(sample[src].read[m],secondaryCal,secondaryFlip);
   for(const [ch,code] of [['PRIMARY',p],['SECONDARY',s]]){
    const r=[h.accepted,'REMOTE_MATERIAL_CHECK_'+m,ch,sample[src].clock,code];ctx.q.push(r);reports[m].push(r);
   }
  }}
  act(ctx,k%2===0?move:reverse[move],k===74?1:0);
 }
 check(h.pos===w.s&&ctx.q.length===0,'hub detour did not reset');
 const critical=ctx.steps.length,obs=sense(h,[]),truth=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m),labels={},targets=[];
 for(const m of ['A','B']){
  const trustAfter=fault.domain==='ISOLATED_SECONDARY'||!!witnessed[m];
  const eligible=reports[m].filter(r=>r[2]==='PRIMARY'&&(trustAfter||r[3]<faultClock)).sort((a,b)=>b[3]-a[3]);
  const latest=eligible[0];let state='UNKNOWN';
  if(latest){const inverted=latest[3]<faultClock||fault.domain==='ISOLATED_SECONDARY'?cal[m]:witnessed[m].code==='ERODED';
   const physical=inverted?(latest[4]==='ERODED'?'INTACT':'ERODED'):latest[4];
   state=physical==='ERODED'?'ERODED':schedule.some(s=>s.clock>latest[3])?'UNKNOWN':'INTACT';
  }
  labels[m]={state,measurement_clock:latest?latest[3]:null,physically_witnessed:!!witnessed[m]};if(state!=='INTACT')targets.push(m);
 }
 check(truth.every(m=>targets.includes(m)),'insufficient inspection of damaged material');
 finish(ctx,targets);check(h.done&&h.success,'teacher physical failure');
 check(H(ctx.steps[critical].input)===H(obs),'critical input mismatch');
 const firstChoice=ctx.steps[decision].action;
 check(firstChoice===(selected==='NONE'?move:'PAINT_'+witnesses(selected)[0]),'bad endogenous choice');
 // v73 does not give the model the counterfactual cost table or hidden flip.
 return {episode:{steps:ctx.steps},certificate:{threshold,sequence:seq,calibration:cal,fault,passes,selected,policy_means:policy.means,incident_count:incidentCount,local_trial_mask:pilotMask,
  decision_step:decision,second_decision_step:secondDecision,decision_sha256:H(ctx.steps[decision].input),decision_action:firstChoice,
  critical_step:critical,critical_sha256:H(obs),schedule,sampling_clocks:{early:sample[38].clock,late:sample[72].clock},fault_clock:faultClock,
  witnessed,actual_broken:truth,targets,teacher_labels:labels,physical_erasures:erasures,terminal_success:h.success,
  total_legal_actions:ctx.steps.length,cost_regret_to_none:policy.means.NONE-policy.means[selected],
  source_host:'synthetic-v44-plus-v67-erasure',native_CyberSand:false}};
}
function writer(p){const fd=fs.openSync(p,'wx'),sha=crypto.createHash('sha256');let rows=0,bytes=0;return {put(x){const s=JSON.stringify(x)+'\n';fs.writeSync(fd,s);sha.update(s);rows++;bytes+=Buffer.byteLength(s);},close(){fs.closeSync(fd);return {rows,bytes,sha256:sha.digest('hex')}}};}
function generate(dir,{quick=false}={}){
 check(!fs.existsSync(dir),'destination exists');fs.mkdirSync(dir,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/physical-prior-shift-v75/1',date:'2026-11-19',quick,synthetic_not_native_CyberSand:true,
  risk_distribution_counts_per_8:RISK_COUNTS,
  generator_sha256:H(fs.readFileSync(__filename)),host_sha256:H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),
  parent_v70_sha256:H(fs.readFileSync(path.join(__dirname,'cross_calibrated_receipt_v70.js'))),
  parent_v72_sha256:H(fs.readFileSync(path.join(__dirname,'physical_rewitness_v72.js'))),
  design:'Split-specific shifted nonuniform erosion priors derived from eight physically enacted local stress tests. Select active witness EX ANTE using sample-frequency-weighted risk, without knowing the future remote abrasion or fault flip. Paired low/high observed risk histories have identical current-only state after >100 steps. No privileged risk scalar in model observations. World geometries are new and split-disjoint.',
  split_contract:'Only TRAIN_physical_prior_shift_stream.jsonl is trainable. DEV and HELDOUT streams plus all certificates and oracle policy_means are evaluator-only. Prefix actions masked, first chosen intervention and recovery supervised; exact source manifest and deduplicated TRAIN repeated followups.',
  caveats:['Synthetic v44 host + added erosion, multichannel telemetry and coupon procedure, not native CyberSand ABI.',
   'The eight coupon trials are scripted synthetic physical evidence; an 8-sample empirical rate is not a calibrated real-world probability and is assumed to describe remote future abrasion.',
   'Train/DEV/HELDOUT differ in local physical sample rate with deterministic split assignments. This evaluates input-conditioned shift, not autonomous neural performance.'],files:{},splits:{}};
 const seen=new Set(),pairs=new Set(),counts={},total={episodes:0,decisions:0,regret:0};
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const sw=writer(path.join(dir,stream(split))),cw=writer(path.join(dir,cert(split))),stats={worlds:[],episodes:0,decisions:0,terminal_successes:0,masked_train_followups:0,probe_choices:{NONE:0,A:0,B:0,BOTH:0},counterfactual_expected_cost_groups:0,groups_with_different_optimal_probe_across_cost_profiles:0,groups_with_different_optimal_probe_across_risk:0,mean_expected_actions_saved_vs_no_probe:0};
  for(let i=0;i<(quick?1:COUNTS[split]);i++){
   const {w,index,attempt}=pickWorld(split,i,seen);stats.worlds.push({index,attempt,seed:w.seed,geometry_sha256:w.geometry_sha256});
   const threshold=1+(i%2),alternate=HISTORIES[1+(i%3)];
   for(const calibration of CALIBRATIONS)for(const domain of FAULT_DOMAINS){
    const riskChoices=[];
    for(const passes of PROFILES){const perRisk=[];
     for(const incidentCount of RISK_COUNTS[split]){
      const policy=expectedCosts(w,threshold,calibration,domain,passes,alternate,incidentCount),chosen=policy.selected;
      perRisk.push(chosen);riskChoices.push(chosen);stats.counterfactual_expected_cost_groups++;
      for(const sequence of [[],alternate])for(const flip of [false,true]){
       const {episode:e,certificate:c}=episode(w,threshold,sequence,calibration,{domain,flip},passes,chosen,policy,incidentCount,i,split);
       c.split=split;c.world_index=i;c.source_world_index=index;c.geometry_sha256=w.geometry_sha256;c.episode=stats.episodes;
       if(split==='TRAIN')for(let t=c.critical_step+1;t<e.steps.length;t++){
        const r=e.steps[t],key=H([r.input,r.action]);if(pairs.has(key)){r.weight=0;r.focus_loss_weight=0;stats.masked_train_followups++;}else pairs.add(key);
       }
       check(e.steps[c.decision_step].weight===1&&e.steps[c.critical_step].weight===1,'critical supervision masked');
       c.episode_sha256=H(e.steps.map(s=>[s.input,s.action,s.weight]));
       sw.put(e);cw.put(c);stats.episodes++;stats.decisions+=e.steps.length;stats.terminal_successes++;stats.probe_choices[chosen]++;
       stats.mean_expected_actions_saved_vs_no_probe+=c.cost_regret_to_none;
      }
     }
     stats.groups_with_different_optimal_probe_across_risk+=Number(new Set(perRisk).size>1);
    }
    stats.groups_with_different_optimal_probe_across_cost_profiles+=Number(new Set(riskChoices).size>1);
   }
  }
  stats.mean_expected_actions_saved_vs_no_probe/=stats.episodes;
  manifest.files[stream(split)]=sw.close();manifest.files[cert(split)]=cw.close();manifest.splits[split]=stats;
  total.episodes+=stats.episodes;total.decisions+=stats.decisions;total.regret+=stats.mean_expected_actions_saved_vs_no_probe*stats.episodes;
 }
 manifest.total_episodes=total.episodes;manifest.total_decisions=total.decisions;manifest.mean_expected_actions_saved_vs_no_probe=total.regret/total.episodes;
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){check(process.argv[2],'usage node physical_prior_shift_v75.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv.includes('--quick')}).splits,null,2));}
module.exports={generate,episode,expectedCosts,pickWorld,H,PROFILES,CALIBRATIONS,FAULT_DOMAINS,RISK_COUNTS,stream,cert};
