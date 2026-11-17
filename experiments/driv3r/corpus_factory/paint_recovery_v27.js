'use strict';
// DRIV3R Corpus Factory v27: synthetic, observation-grounded paint-error recovery.
// Do not confuse this local host with the canonical native CyberSand action ABI.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v26=require('./causal_gate_v26.js');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const cp=x=>JSON.parse(JSON.stringify(x));
const SPLITS=v26.SPLITS,GROUP_IDS={TRAIN:[0,1,2,3,4,5,6,7],DEV:[1,2,3,4],HELDOUT:[1,2,3,4]};
const MODES=['correct-paint-control','wrong-material-recovery','wrong-branch-and-material-recovery'];
const state=h=>[h.pos,h.prev,h.status,h.latched];
// Ensure the erroneous paint and gate change have aged out of a latest-64 window.
// Wander without touching any paintable cell, then start the labelled suffix away
// from the damaged target. Delayed movement is scripted and disclosed as such.
function delayAway(h,minimum=72){
 const w=h.w,origin=h.pos,dist=new Map([[origin,0]]),parent=new Map(),q=[origin];
 for(let k=0;k<q.length;k++)for(const n of w.adjacency[q[k]]){
  if(!v26.edgeAllowed(w,q[k],n,h.latched)||dist.has(n))continue;
  dist.set(n,dist.get(q[k])+1);parent.set(n,q[k]);q.push(n);
 }
 const ranked=[...dist].filter(([n,d])=>d>=3&&!w.target.includes(w.nodes[n])).sort((a,b)=>b[1]-a[1]||w.nodes[a[0]].localeCompare(w.nodes[b[0]]));
 if(!ranked.length)throw Error('no remote resting point for delayed memory');
 const dest=ranked[0][0],route=[];for(let n=dest;n!==origin;n=parent.get(n))route.unshift(n);
 function moveTo(n){const a=v26.options(w,state(h)).find(x=>x.next&&x.next[0]===n&&x.action.startsWith('MOVE_')||x.next&&x.next[0]===n&&x.action==='BACKTRACK');
  if(!a||!v26.apply(h,a.action).accepted)throw Error('remote walk refused');return a.action;}
 const actions=[];for(const n of route)actions.push(moveTo(n));
 const near=parent.get(dest);if(near===undefined)throw Error('missing remote neighbor');
 while(actions.length<minimum){actions.push(moveTo(near));actions.push(moveTo(dest));}
 if(h.pos!==dest||dist.get(dest)<3)throw Error('no distant resting point');
 if(h.history.events.slice(-64).some(e=>e.action?.startsWith('PAINT')||e.kind==='GATE_TRANSITION'))throw Error('paint event leaked into latest64');
 return {actions,remote_distance:dist.get(dest),resting_cell:w.nodes[dest]};
}
function prepare(c,split,g,v,mode){
 if(!MODES.includes(mode))throw Error('unknown mode');
 const h=new v26.Host(c.w,split,g,v),actions=[],both=mode==='wrong-branch-and-material-recovery',wrong=mode!=='correct-paint-control';
 h.id+=`-${mode}`;h.remaining=320;
 if(both){const a=c.first==='BACKTRACK'?'MOVE_N':'BACKTRACK';if(!v26.apply(h,a).accepted)throw Error('wrong branch refused');actions.push(a);}
 let chosen=null;
 for(let k=0;k<110;k++){
  const ranked=c.p.rank(state(h))[0];if(!ranked||!Number.isFinite(ranked.cost)||ranked.action==='STOP')throw Error('pre-paint prefix cannot reach goal');
  const paint=ranked.action==='PAINT_A'||ranked.action==='PAINT_B';
  const action=paint&&wrong?(ranked.action==='PAINT_A'?'PAINT_B':'PAINT_A'):ranked.action;
  const receipt=v26.apply(h,action);if(!receipt.accepted)throw Error('prefix refused '+action);
  actions.push(action);
  if(paint){chosen={expert_paint:ranked.action,actual_paint:action,cell:c.w.nodes[h.pos],prefix_actions:actions.length,wrong_paint:wrong};break;}
 }
 if(!chosen||h.latched!==1||h.done)throw Error('prefix did not trigger gate');
 const drift=delayAway(h);actions.push(...drift.actions);chosen={...chosen,delay_steps:drift.actions.length,remote_distance:drift.remote_distance,resting_cell:drift.resting_cell};
 const outcome=v26.correct(c.w,h.status);
 if(wrong&&outcome.wrong!==1)throw Error('missing accepted bad-material paint');
 if(!wrong&&(outcome.count!==1||outcome.wrong!==0))throw Error('bad clean control');
 return {h,chosen,actions};
}
function auditFrozen(c,h){const p=c.p,old=c.frozenPlanner||(c.frozenPlanner=v26.planner(c.w,{fixed:c.config.initial_open}));let harmful=0,illegal=0,disagree=0;
 const snapshots=[];
 while(!h.done){const s=state(h),dyn=p.rank(s)[0],frozen=old.rank(s)[0];if(!dyn||!Number.isFinite(dyn.cost))throw Error('unsolved');
  if(frozen&&dyn.action!==frozen.action){disagree++;const actual=p.rank(s).find(x=>x.action===frozen.action);if(!actual||!Number.isFinite(actual.cost))illegal++;else if(actual.cost>dyn.cost)harmful++;}
  snapshots.push({action:dyn.action,regret_vs_frozen:frozen?(p.rank(s).find(x=>x.action===frozen.action)?.cost??null)-dyn.cost:null});
  if(!v26.apply(h,dyn.action).accepted)throw Error('oracle refused');
  if(snapshots.length>120)throw Error('excessive suffix');
 }
 return {harmful,illegal,disagree,steps:snapshots.length};
}
function replay(c,split,g,v,mode,emit){
 const {h,chosen,actions}=prepare(c,split,g,v,mode),first=v26.observe(h);
 const frozenH=prepare(c,split,g,v,mode).h,audit=auditFrozen(c,frozenH);
 const header={schema:'driv3r/corpus_factory/paint-recovery-v27/episode',episode_id:h.id,split,episode_mode:mode,synthetic_host:true,
  world_group:`${split}-${g}`,world_sha256:c.w.world_sha256,geometry_sha256:c.w.graph_sha256,
  difficulty:{graph_nodes:c.w.nodes.length,first_branch_regret:c.regret,accepted_predecision_actions:actions.length,first_paint_cell:chosen.cell,remote_start_distance:chosen.remote_distance,post_paint_distractor_moves:chosen.delay_steps},
  intervention:{kind:mode,accepted_wrong_paint:chosen.wrong_paint,accepted_wrong_branch:mode==='wrong-branch-and-material-recovery'},
  initial_history:cp(h.history),observation_keys:Object.keys(first)};
 let e=h.history.events.length,z=h.history.positions.length,steps=0,paint=0,erase=0,retrace=0;
 while(!h.done){const o=v26.observe(h),rank=c.p.rank(state(h))[0];if(!rank||!Number.isFinite(rank.cost))throw Error('unreachable');
  const events_added=cp(o.history.events.slice(e)),positions_added=o.history.positions.slice(z);e=o.history.events.length;z=o.history.positions.length;
  const receipt=v26.apply(h,rank.action);if(!receipt.accepted)throw Error('invalid expert action');
  emit({schema:'driv3r/corpus_factory/paint-recovery-v27/decision',episode_id:h.id,step:steps,
   state:Object.fromEntries(Object.entries(o).filter(([k])=>k!=='history')),events_added,positions_added,
   model_input_sha256:sha(o),target:rank.action,receipt,training_weight:1},o);
  if(rank.action==='ERASE')erase++;
  if(rank.action.startsWith('PAINT'))paint++;
  if(rank.action==='BACKTRACK')retrace++;
  if(++steps>120)throw Error('suffix budget');
 }
 if(steps!==audit.steps)throw Error('audit/trace mismatch');
 const wrong=mode!=='correct-paint-control';
 if(wrong&&erase<1)throw Error('wrong material not corrected');
 if(!wrong&&erase!==0)throw Error('control unexpectedly erases');
 if(v26.correct(c.w,h.status).wrong!==0)throw Error('residual contamination');
 if(!v26.correct(c.w,h.status).count||!h.done)throw Error('incomplete recovery');
 return {...header,terminal_success:true,steps,expert_paint:paint,corrective_erase:erase,expert_retrace:retrace,
  first_paint_correct:!wrong,accepted_bad_paint:wrong?1:0,accepted_wrong_branch:mode==='wrong-branch-and-material-recovery'?1:0,
  initial_gate_open:c.config.initial_open,causal_frozen_gate_disagreements:audit.disagree,
  causal_frozen_gate_harmful:audit.harmful,causal_frozen_gate_illegal:audit.illegal};
}
function meta(f,rows){const b=fs.readFileSync(f);return {sha256:sha(b),bytes:b.length,rows};}
function generate(dest,{quick=false}={}){
 if(fs.existsSync(dest))throw Error('refusing overwrite');fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},inputOwners=new Map(),worldOwners=new Map();
 for(const [split,n] of Object.entries(SPLITS)){
  const episodes=[],decisions=[],exact=new Map(),near=new Map(),groups=quick?1:n;
  let near_downweighted=0,near_ambiguous=0;
  for(let i=0;i<groups;i++){
   const g=GROUP_IDS[split][i],cs=v26.family(split,g);
   for(let k=0;k<cs.length;k++){
    const c=cs[k],own=worldOwners.get(c.w.world_sha256);if(own&&own!==split)throw Error('world cross-split');worldOwners.set(c.w.world_sha256,split);
    for(const mode of MODES){const ep=replay(c,split,g,k,mode,(r,o)=>{
     const prev=exact.get(r.model_input_sha256);if(prev&&prev!==r.target)throw Error('contradictory complete-input label');
     const owner=inputOwners.get(r.model_input_sha256);if(owner&&owner!==split)throw Error('complete-input split leakage');inputOwners.set(r.model_input_sha256,split);
     if(prev){if(split==='TRAIN')r.training_weight=0;}
     else exact.set(r.model_input_sha256,r.target);
     // Conservative near-duplicate: public state, original remote survey,
     // and latest eight public action receipts/positions. Conflicting labels stay.
     const nearKey=sha([o.protocol,o.position,o.local_view,o.legal_actions,o.painted_correct,
        o.goal,o.history.events[0],o.history.events.slice(-8),o.history.positions.slice(-8)]);
     if(near.has(nearKey)&&near.get(nearKey)===r.target&&split==='TRAIN'&&r.training_weight===1){
       r.training_weight=0;near_downweighted++;
     }else if(near.has(nearKey)&&near.get(nearKey)!==r.target)near_ambiguous++;
     else if(!near.has(nearKey))near.set(nearKey,r.target);
     decisions.push(r);
    });episodes.push(ep);}
   }
  }
  for(const kind of ['episodes','decisions']){const f=`${split}_${kind}.jsonl`,rows=kind==='episodes'?episodes:decisions;fs.writeFileSync(path.join(dest,f),rows.map(JSON.stringify).join('\n')+'\n');shards[f]=meta(path.join(dest,f),rows.length);}
  splits[split]={groups,group_ids:GROUP_IDS[split].slice(0,groups),episodes:episodes.length,decisions:decisions.length,
   unique_inputs:exact.size,exact_duplicates:decisions.length-exact.size,near_downweighted,near_ambiguous,
   modes:Object.fromEntries(MODES.map(mode=>[mode,episodes.filter(e=>e.episode_mode===mode).length])),
   corrective_erases:episodes.reduce((n,e)=>n+e.corrective_erase,0),accepted_bad_paints:episodes.reduce((n,e)=>n+e.accepted_bad_paint,0),
   accepted_wrong_branches:episodes.reduce((n,e)=>n+e.accepted_wrong_branch,0),
   strict_causal_episodes:episodes.filter(e=>e.causal_frozen_gate_harmful+e.causal_frozen_gate_illegal>0).length,
   frozen_gate_harmful_recommendations:episodes.reduce((n,e)=>n+e.causal_frozen_gate_harmful,0),
   frozen_gate_illegal_recommendations:episodes.reduce((n,e)=>n+e.causal_frozen_gate_illegal,0),
   min_train_weight:Math.min(...decisions.map(r=>r.training_weight)),zero_weight:decisions.filter(r=>r.training_weight===0).length};
 }
 const manifest={schema:'driv3r/corpus_factory/paint-recovery-v27/manifest',host:'synthetic-not-native-CyberSand',
  generator:'paint_recovery_v27.js',generator_sha256:sha(fs.readFileSync(__filename)),
  dependencies:['causal_gate_v26.js','outcome_forks_v25.js'].map(f=>({file:f,sha256:sha(fs.readFileSync(path.join(__dirname,f)))})),
  oracle:'v26 reverse BFS minimum terminal-cost suffix',split_worlds_disjoint:true,
  initial_96_move_history_scripted:true,pre_first_paint_actions_scripted:true,
  accepted_bad_material_in_prefix:true,post_paint_remote_drift_scripted:true,
  accepted_paint_receipt_outside_latest64:true,post_intervention_suffix_oracle_labelled:true,
  consumer_warning:'Do not pass episode mode, intervention, difficulty or frozen-gate diagnostics to model. Reconstruct observation from initial_history and deltas; verify SHA-256. Post-intervention trajectories are oracle, NOT model rollouts.',
  quick,splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={MODES,prepare,delayAway,auditFrozen,replay,generate};
if(require.main===module){if(!process.argv[2])throw Error('usage node paint_recovery_v27.js NEW_OUTPUT [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
