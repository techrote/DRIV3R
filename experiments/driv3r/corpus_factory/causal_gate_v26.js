'use strict';
// DRIV3R Corpus Factory v26. Synthetic host, NOT the native CyberSand ABI.
// Causal, paint-triggered edge gate changes; labels are terminal-outcome BFS.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./outcome_forks_v25.js');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
const DIRS=[[0,-1,'MOVE_N'],[1,0,'MOVE_E'],[0,1,'MOVE_S'],[-1,0,'MOVE_W']];
const SPLITS={TRAIN:8,DEV:4,HELDOUT:4};
const GROUP_IDS={TRAIN:[0,1,2,3,4,5,6,7],DEV:[1,2,3,4],HELDOUT:[1,2,3,4]};
function world(split,g,config){
 const w=base.world(split,g,{blocked:config.blocked,upper_link:config.upper_link,target_mask:config.target_mask,phase:config.phase});
 const a=w.nidx.get(`-1,0`),b=w.nidx.get(`-1,1`);
 if(a===undefined||b===undefined||!w.adjacency[a].includes(b))throw Error('missing gate edge');
 w.gate=[a,b];w.gate_initial_open=config.initial_open;
 w.rule={kind:'FIRST_ACCEPTED_PAINT_TOGGLES_GATE',gate_edge:[w.nodes[a],w.nodes[b]],initial_open:config.initial_open};
 w.world_sha256=sha({graph:w.graph_sha256,targets:w.target,recipes:w.recipes,rule:w.rule});
 return w;
}
function digit(s,i){return Math.floor(s/Math.pow(3,i))%3;}
function writeDigit(s,i,v){return s+(v-digit(s,i))*Math.pow(3,i);}
function correct(w,status){let count=0,wrong=0;for(let i=0;i<w.target.length;i++){let d=digit(status,i),want=w.recipes[i]==='A'?1:2;if(d===want)count++;else if(d!==0)wrong++;}return {count,wrong};}
function isTerminal(w,s){const [p,,status,latched]=s,q=correct(w,status);return p===w.exit&&latched===1&&q.count>=w.quota&&q.wrong===0;}
function sKey(s){return s.join('|');}
function gateOpen(w,latched,fixed){return fixed===undefined?(Boolean(w.gate_initial_open)!==Boolean(latched)):fixed;}
function edgeAllowed(w,p,n,latched,fixed){return !((p===w.gate[0]&&n===w.gate[1])||(p===w.gate[1]&&n===w.gate[0]))||gateOpen(w,latched,fixed);}
function options(w,s,fixed){const [p,prev,status,latched]=s,actions=[];
 for(const n of w.adjacency[p]){if(!edgeAllowed(w,p,n,latched,fixed))continue;
  const [x,y]=w.nodes[p].split(',').map(Number),[u,v]=w.nodes[n].split(',').map(Number);
  const action=n===prev?'BACKTRACK':DIRS.find(d=>d[0]===u-x&&d[1]===v-y)?.[2];if(!action)throw Error('nonorthogonal node');
  actions.push({action,next:[n,p,status,latched]});
 }
 const i=w.target.indexOf(w.nodes[p]);if(i>=0){const d=digit(status,i);
  if(d===0)for(let m=1;m<=2;m++)actions.push({action:m===1?'PAINT_A':'PAINT_B',next:[p,prev,writeDigit(status,i,m),1]});
  else actions.push({action:'ERASE',next:[p,prev,writeDigit(status,i,0),latched]});
 }
 if(isTerminal(w,s))actions.push({action:'STOP',next:null});
 return actions.sort((a,b)=>a.action.localeCompare(b.action));
}
function planner(w,{fixed}={}){
 const reverse=new Map(),dist=new Map(),queue=[];let all=0;
 for(let p=0;p<w.nodes.length;p++)for(const prev of w.adjacency[p])for(let status=0;status<27;status++)for(let latched=0;latched<=1;latched++){
  const s=[p,prev,status,latched],key=sKey(s);all++;
  if(isTerminal(w,s)){dist.set(key,1);queue.push(key);}
  for(const a of options(w,s,fixed))if(a.next){const to=sKey(a.next);if(!reverse.has(to))reverse.set(to,[]);reverse.get(to).push(key);}
 }
 for(let i=0;i<queue.length;i++){const k=queue[i],d=dist.get(k);for(const pred of reverse.get(k)||[])if(!dist.has(pred)){dist.set(pred,d+1);queue.push(pred);}}
 function rank(s){return options(w,s,fixed).map(a=>({action:a.action,cost:a.next?1+(dist.get(sKey(a.next))??Infinity):1})).sort((a,b)=>a.cost-b.cost||a.action.localeCompare(b.action));}
 return {rank,reachable:dist.size,total_states:all};
}
function prefix(w){
 const [a,b]=w.gate,protected_cells=['0,1','-2,1'].filter(c=>!w.nidx.has(c));
 const e=[{kind:'REMOTE_SURVEY',tiles:w.nodes.map((n,i)=>({cell:n,neighbors:w.adjacency[i].map(j=>w.nodes[j])})),paint_targets:w.target.map((t,i)=>({cell:t,material:w.recipes[i]})),protected_cells,goal:{quota:w.quota,exit:w.nodes[w.exit]},live_gate:w.rule}];
 const pos=[w.nodes[w.exit]];
 for(let i=0;i<96;i++){const at=i%2===0?w.nodes[w.previous]:w.nodes[w.exit];e.push({kind:'MOVE_ACK',accepted:true,at});pos.push(at);}
 e.push({kind:'MOVE_ACK',accepted:true,at:w.nodes[w.root]});pos.push(w.nodes[w.root]);return {events:e,positions:pos};
}
class Host{constructor(w,split,group,v){this.w=w;this.pos=w.root;this.prev=w.previous;this.status=0;this.latched=0;this.history=prefix(w);this.remaining=120;this.done=false;this.invalid=false;this.id=`${split}-gate-${String(group).padStart(2,'0')}-${v}`;}}
function observe(h){const w=h.w,p=h.pos,[x,y]=w.nodes[p].split(',').map(Number),view=[];
 for(const [dx,dy] of [[0,0],...DIRS.map(d=>d.slice(0,2))]){
  const n=`${x+dx},${y+dy}`,j=w.nidx.get(n),reach=j!==undefined&&w.adjacency[p].includes(j)&&edgeAllowed(w,p,j,h.latched),ti=w.target.indexOf(n),paintable=ti>=0&&(reach||(dx===0&&dy===0));
  view.push({offset:[dx,dy],reachable:reach,paintable,painted:paintable?['EMPTY','A','B'][digit(h.status,ti)]:null});
 }
 return {protocol:'driv3r.synthetic-causal-gate/26',position:w.nodes[p],local_view:view,legal_actions:options(w,[h.pos,h.prev,h.status,h.latched]).map(a=>a.action),goal:{quota:w.quota,exit:w.nodes[w.exit]},painted_correct:correct(w,h.status).count,remaining:h.remaining,history:clone(h.history)};
}
function apply(h,action){if(h.done||h.remaining<=0)throw Error('terminal or budget');const s=[h.pos,h.prev,h.status,h.latched];const a=options(h.w,s).find(x=>x.action===action);
 if(!a){h.history.events.push({kind:'REFUSED',accepted:false,action});h.remaining--;return {accepted:false,action};}
 if(a.next)[h.pos,h.prev,h.status,h.latched]=a.next;else h.done=true;
 h.remaining--;const receipt={kind:'ACTION_ACK',accepted:true,action,at:h.w.nodes[h.pos],painted_correct:correct(h.w,h.status).count};h.history.events.push(receipt);h.history.positions.push(h.w.nodes[h.pos]);
 if(!s[3]&&h.latched){h.history.events.push({kind:'GATE_TRANSITION',accepted:true,at:h.w.rule.gate_edge,open:gateOpen(h.w,h.latched),cause:'FIRST_ACCEPTED_PAINT'});}
 return receipt;
}
function trace(w,p,split,g,v){const h=new Host(w,split,g,v);let steps=0,gate_changes=0,after_gate=0,paints=0,erasures=0,retraces=0;
 while(!h.done){const r=p.rank([h.pos,h.prev,h.status,h.latched])[0];if(!r||!Number.isFinite(r.cost)||steps>118)throw Error('no suffix from '+h.id);
  const before=h.latched;const receipt=apply(h,r.action);if(!receipt.accepted)throw Error('oracle refused');steps++;
  if(!before&&h.latched)gate_changes++;if(before)after_gate++;
  if(r.action.startsWith('PAINT'))paints++;if(r.action==='ERASE')erasures++;if(r.action==='BACKTRACK')retraces++;
 }
 return {steps,gate_changes,after_gate,paints,erasures,retraces,terminal:!h.invalid&&h.done};
}
// Sparse candidates based on v25 proven suffix-positive worlds; recompute all labels using v26's dynamic solver.
function family(split,g){
 const seed=base.candidates(split,g),pool=[];
 for(const c of seed)for(const initial_open of [false,true]){
  const config={...c.config,initial_open},w=world(split,g,config),p=planner(w),rank=p.rank([w.root,w.previous,0,0]);
  if(!['BACKTRACK','MOVE_N'].includes(rank[0].action)||!Number.isFinite(rank[0].cost)||rank[0].cost>110)continue;
  const other=rank.find(x=>x.action===(rank[0].action==='BACKTRACK'?'MOVE_N':'BACKTRACK'));
  if(!other||!Number.isFinite(other.cost)||other.cost-rank[0].cost<2)continue;
  const completed=trace(w,p,split,g,0);if(completed.gate_changes!==1||!completed.terminal)continue;
  pool.push({w,p,config,first:rank[0].action,optimal:rank[0].cost,alternative:other.cost,regret:other.cost-rank[0].cost,completed});
 }
 // Eight cases, two per initial gate status and optimal action. Including the lower
 // target in MOVE_N cases forces an observable causal reroute on the expert suffix.
 const selected=[];
 for(const initial_open of [false,true])for(const first of ['BACKTRACK','MOVE_N']){
  const possible=pool.filter(x=>x.config.initial_open===initial_open&&x.first===first)
   .sort((a,b)=>{
    if(first==='MOVE_N'){const pref=x=>x.config.target_mask===21?3:x.config.target_mask===25?2:Boolean(x.config.target_mask&16)?1:0;if(pref(a)!==pref(b))return pref(b)-pref(a);}
    return b.regret-a.regret||a.config.target_mask-b.config.target_mask||a.config.phase-b.config.phase;
   });
  if(possible.length<2)throw Error(`v26 insufficient ${split}:${g}:${initial_open}:${first}; candidates=${pool.length}`);
  selected.push(possible[0]);selected.push(possible.find(x=>x.config.target_mask!==possible[0].config.target_mask)||possible.find(x=>x.config.phase!==possible[0].config.phase)||possible[1]);
 }
 // Explicit frozen-gate counterfactual oracle. This is diagnostic/control only,
 // never presented as ground-truth policy input or causal regret.
 for(const c of selected){
  const frozen=planner(c.w,{fixed:c.config.initial_open}),h=new Host(c.w,split,g,0);
  let afterGateDifferences=0,postGateActions=0,frozenIllegal=0,frozenSuboptimal=0;
  while(!h.done){const s=[h.pos,h.prev,h.status,h.latched],dyn=c.p.rank(s)[0],old=frozen.rank(s)[0];
   if(h.latched){postGateActions++;if(dyn.action!==old.action){afterGateDifferences++;const actual=c.p.rank(s).find(x=>x.action===old.action);if(!actual||!Number.isFinite(actual.cost))frozenIllegal++;else if(actual.cost>dyn.cost)frozenSuboptimal++;}}
   apply(h,dyn.action);
  }
  c.post_gate_policy_differences=afterGateDifferences;c.post_gate_actions=postGateActions;c.frozen_gate_illegal=frozenIllegal;c.frozen_gate_strict_regret=frozenSuboptimal;
 }
 // if no contrast, retain as negative control with explicit metrics
 // Pair outcomes with same recent view, legality, budget; earlier remote survey carries the differentiator.
 const reference=observe(new Host(selected[0].w,split,g,0));
 for(const c of selected){const o=observe(new Host(c.w,split,g,0));
  for(const k of ['local_view','legal_actions','goal','remaining'])if(JSON.stringify(o[k])!==JSON.stringify(reference[k]))throw Error('alias mismatch '+k);
  if(JSON.stringify(o.history.events.slice(-64))!==JSON.stringify(reference.history.events.slice(-64)))throw Error('last64 mismatch');
 }
 return selected;
}
function replay(c,split,g,v,emit,mode='clean'){
 const h=new Host(c.w,split,g,v);
 if(mode==='wrong-first-recovery'){
  const wrong=c.first==='BACKTRACK'?'MOVE_N':'BACKTRACK';
  if(!apply(h,wrong).accepted)throw Error('recovery intervention not host legal');
  h.id+='-recovery';
 } else if(mode!=='clean')throw Error('unknown mode');
 const o=observe(h),header={schema:'driv3r/corpus_factory/causal-gate-v26/episode',episode_id:h.id,split,episode_mode:mode,intervention:mode==='clean'?null:{kind:'ACCEPTED_SUBOPTIMAL_FIRST_ACTION',action:c.first==='BACKTRACK'?'MOVE_N':'BACKTRACK'},world_group:`${split}-${g}`,synthetic_host:true,geometry_sha256:c.w.graph_sha256,world_sha256:c.w.world_sha256,difficulty:{nodes:c.w.nodes.length,optimal_suffix_steps:c.optimal,proven_first_action_regret:c.regret},initial_history:clone(h.history),observation_keys:Object.keys(o)};
 let e=h.history.events.length,z=h.history.positions.length,step=0,paints=0,erasures=0,retraces=0,gate_changes=0;
 while(!h.done){const input=observe(h),r=c.p.rank([h.pos,h.prev,h.status,h.latched])[0];if(!Number.isFinite(r.cost))throw Error('unsolved');
  const events_added=clone(input.history.events.slice(e)),positions_added=input.history.positions.slice(z);e=input.history.events.length;z=input.history.positions.length;
  const before=h.latched,receipt=apply(h,r.action);if(!receipt.accepted)throw Error('invalid expert');
  emit({schema:'driv3r/corpus_factory/causal-gate-v26/decision',episode_id:h.id,step,state:Object.fromEntries(Object.entries(input).filter(([k])=>k!=='history')),events_added,positions_added,model_input_sha256:sha(input),target:r.action,receipt,training_weight:1});
  if(!before&&h.latched)gate_changes++;if(r.action.startsWith('PAINT'))paints++;if(r.action==='ERASE')erasures++;if(r.action==='BACKTRACK')retraces++;step++;if(step>119)throw Error('timeout');
 }
 const expected=mode==='clean'?c.optimal:c.alternative-1;
 if(step!==expected)throw Error('not optimal '+step+' vs '+expected+' mode '+mode);
 return {...header,steps:step,paints,erasures,retraces,gate_changes,terminal_success:true,first_target:mode==='clean'?c.first:null,first_action_regret:c.regret,initial_gate_open:c.config.initial_open,post_gate_policy_differences_vs_frozen_gate:mode==='clean'?c.post_gate_policy_differences:null,post_gate_actions:mode==='clean'?c.post_gate_actions:null,frozen_gate_illegal:mode==='clean'?c.frozen_gate_illegal:null,frozen_gate_strict_regret:mode==='clean'?c.frozen_gate_strict_regret:null,accepted_predecision_interventions:mode==='clean'?0:1};
}
function fileMeta(p,rows){const b=fs.readFileSync(p);return {sha256:sha(b),bytes:b.length,rows};}
function generate(dest,{quick=false}={}){
 if(fs.existsSync(dest))throw Error('refusing overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
 const splits={},shards={},worldOwners=new Map(),inputOwners=new Map();
 for(const [split,n] of Object.entries(SPLITS)){
  const headers=[],rows=[],seen=new Map(),firstCounts={BACKTRACK:0,MOVE_N:0},gateCounts={open:0,closed:0};const groups=quick?1:n;
  for(let i=0;i<groups;i++){
   const g=GROUP_IDS[split][i],variants=family(split,g);
   for(let v=0;v<variants.length;v++){
    const c=variants[v],prior=worldOwners.get(c.w.world_sha256);if(prior&&prior!==split)throw Error('split world leakage');worldOwners.set(c.w.world_sha256,split);
    firstCounts[c.first]++;gateCounts[c.config.initial_open?'open':'closed']++;
    for(const mode of ['clean','wrong-first-recovery']){
     const ep=replay(c,split,g,v,r=>{
      const old=seen.get(r.model_input_sha256);if(old!==undefined){if(old!==r.target)throw Error('conflicting labels');if(split==='TRAIN')r.training_weight=0;}else seen.set(r.model_input_sha256,r.target);
      const owner=inputOwners.get(r.model_input_sha256);if(owner&&owner!==split)throw Error('cross split input overlap');inputOwners.set(r.model_input_sha256,split);rows.push(r);
     },mode);headers.push(ep);
    }
   }
  }
  if(firstCounts.BACKTRACK!==firstCounts.MOVE_N||gateCounts.open!==gateCounts.closed)throw Error('unbalanced family');
  const ef=`${split}_episodes.jsonl`,df=`${split}_decisions.jsonl`;fs.writeFileSync(path.join(dest,ef),headers.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dest,df),rows.map(JSON.stringify).join('\n')+'\n');shards[ef]=fileMeta(path.join(dest,ef),headers.length);shards[df]=fileMeta(path.join(dest,df),rows.length);
  splits[split]={groups,group_ids:GROUP_IDS[split].slice(0,groups),episodes:headers.length,clean_episodes:headers.filter(e=>e.episode_mode==='clean').length,wrong_first_recovery_episodes:headers.filter(e=>e.episode_mode==='wrong-first-recovery').length,decisions:rows.length,unique_inputs:seen.size,duplicate_inputs:rows.length-seen.size,first_action:firstCounts,gate_initial:gateCounts,gate_transitions:headers.reduce((n,e)=>n+e.gate_changes,0),causal_clean_episode_count:headers.filter(e=>e.episode_mode==='clean'&&e.post_gate_policy_differences_vs_frozen_gate>0).length,post_gate_action_disagreements_vs_frozen_gate_clean_only:headers.reduce((n,e)=>n+(e.post_gate_policy_differences_vs_frozen_gate||0),0),frozen_gate_illegal_clean_only:headers.reduce((n,e)=>n+(e.frozen_gate_illegal||0),0),frozen_gate_strict_regret_clean_only:headers.reduce((n,e)=>n+(e.frozen_gate_strict_regret||0),0),strict_causal_clean_episodes:headers.filter(e=>e.episode_mode==='clean'&&((e.frozen_gate_illegal||0)+(e.frozen_gate_strict_regret||0))>0).length,paint_actions:headers.reduce((n,e)=>n+e.paints,0),erase_actions:headers.reduce((n,e)=>n+e.erasures,0),retrace_actions:headers.reduce((n,e)=>n+e.retraces,0),min_first_regret:Math.min(...headers.map(e=>e.first_action_regret))};
 }
 const manifest={schema:'driv3r/corpus_factory/causal-gate-v26/manifest',host:'synthetic-not-native-CyberSand',generator:'causal_gate_v26.js',generator_sha256:sha(fs.readFileSync(__filename)),dependency:'outcome_forks_v25.js',dependency_sha256:sha(fs.readFileSync(require.resolve('./outcome_forks_v25.js'))),quick,oracle:'reverse-BFS over position, immediate predecessor, 3-target paint state, gate-transition latch',initial_preparatory_history_scripted:true,host_legal_wrong_first_intervention_prefix_in_recovery:true,gate_transition:'first accepted paint toggles lower-west loop entrance once',oracle_actions_entire_post_prelude:true,split_worlds_disjoint:true,full_last64_alias_within_family:true,splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={SPLITS,sha,world,options,planner,Host,observe,apply,family,replay,generate,correct,gateOpen,edgeAllowed};
if(require.main===module){if(!process.argv[2])throw Error('usage: node causal_gate_v26.js NEW_DIRECTORY [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}