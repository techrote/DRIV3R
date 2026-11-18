'use strict';
// DRIV3R Corpus Factory v28: remote paint-error LOCATION memory counterfactuals.
// Synthetic host only; never imply native CyberSand ABI compatibility.
const fs=require('node:fs'), path=require('node:path'), crypto=require('node:crypto');
const v26=require('./causal_gate_v26.js');
const sha=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const cp=x=>JSON.parse(JSON.stringify(x));
const S=v26.SPLITS, G={TRAIN:[0,1,2,3,4,5,6,7],DEV:[1,2,3,4],HELDOUT:[1,2,3,4]};
const s=h=>[h.pos,h.prev,h.status,h.latched];
function route(w,start,finish,latched){
 const seen=new Map([[start,null]]),q=[start];for(let k=0;k<q.length;k++){const n=q[k];if(n===finish)break;
  for(const to of w.adjacency[n])if(!seen.has(to)&&v26.edgeAllowed(w,n,to,latched)){seen.set(to,n);q.push(to);}}
 if(!seen.has(finish))return null;
 const out=[];for(let n=finish;n!==start;n=seen.get(n))out.unshift(n);return out;
}
function move(h,next){const choice=v26.options(h.w,s(h)).find(x=>x.next?.[0]===next);if(!choice||!v26.apply(h,choice.action).accepted)throw Error('nonlegal route move');return choice.action;}
function prepare(c,split,g,variant,targetIndex,rest){
 const w=c.w,h=new v26.Host(w,split,g,variant),target=w.nidx.get(w.target[targetIndex]);
 h.id+=`-remote-error-${targetIndex}`;h.remaining=320;
 const forward=route(w,h.pos,target,0);if(!forward)return null;
 for(const n of forward)move(h,n);
 const wrong=w.recipes[targetIndex]==='A'?'PAINT_B':'PAINT_A';
 const receipt=v26.apply(h,wrong);if(!receipt.accepted||h.latched!==1)throw Error('wrong paint not accepted');
 const backward=route(w,h.pos,rest,1);if(!backward)return null;
 for(const n of backward)move(h,n);
 const adjacent=w.adjacency[rest].find(n=>v26.edgeAllowed(w,rest,n,1)&&!w.target.includes(w.nodes[n]));
 if(adjacent===undefined)return null;
 const afterPaintMoves=backward.length;
 while(h.remaining>140){move(h,adjacent);move(h,rest);}
 if(h.pos!==rest||h.remaining<139)throw Error('unstable rest');
 const seen=v26.correct(w,h.status);
 if(seen.count!==0||seen.wrong!==1)throw Error('wrong-paint cohort corruption count changed');
 const o=v26.observe(h),rank=c.p.rank(s(h));
 return {h,o,rank,paint_cell:w.target[targetIndex],targetIndex,forward_steps:forward.length,backward_steps:backward.length,remote_distance:route(w,rest,target,1)?.length??Infinity,
   aged_out:o.history.events.slice(-64).every(e=>!e.action?.startsWith('PAINT')&&e.kind!=='GATE_TRANSITION'),
   after_paint_moves:afterPaintMoves+(320-h.remaining-forward.length-1-backward.length)};
}
function restricted(o){return {protocol:o.protocol,position:o.position,local_view:o.local_view,legal_actions:o.legal_actions,
 goal:o.goal,painted_correct:o.painted_correct,remaining:o.remaining,
 recent_events:o.history.events.slice(-64),recent_positions:o.history.positions.slice(-64),first_public_survey:o.history.events[0]};}
function pairFor(c,split,g,v){
 const w=c.w,choices=[];
 for(let rest=0;rest<w.nodes.length;rest++){
  if(w.target.includes(w.nodes[rest]))continue;
  if(w.adjacency[rest].length<2)continue;
  const items=[0,1,2].map(t=>prepare(c,split,g,v,t,rest));
  for(let a=0;a<3;a++)for(let b=a+1;b<3;b++){
   const x=items[a],y=items[b];if(!x||!y||x.remote_distance<3||y.remote_distance<3||!x.aged_out||!y.aged_out)continue;
   if(x.rank[0].action===y.rank[0].action||!Number.isFinite(x.rank[0].cost)||!Number.isFinite(y.rank[0].cost))continue;
   if(sha(restricted(x.o))!==sha(restricted(y.o)))continue;
   if(sha(x.o)===sha(y.o))throw Error('full inputs should differ');
   const wrongForX=x.rank.find(z=>z.action===y.rank[0].action),wrongForY=y.rank.find(z=>z.action===x.rank[0].action);
   const regrets=[(wrongForX?.cost??Infinity)-x.rank[0].cost,(wrongForY?.cost??Infinity)-y.rank[0].cost];
   if(!regrets.every(z=>Number.isFinite(z)&&z>=2))continue;
   choices.push({cases:[x,y],rest_cell:w.nodes[rest],regrets,min_regret:Math.min(...regrets),min_remote:Math.min(x.remote_distance,y.remote_distance)});
  }
 }
 choices.sort((a,b)=>b.min_regret-a.min_regret||b.min_remote-a.min_remote||a.rest_cell.localeCompare(b.rest_cell)||a.cases[0].targetIndex-b.cases[0].targetIndex);
 return choices[0]||null;
}
function replay(c,split,g,v,paired,arm,emit){
 const chosen=paired.cases[arm],h=chosen.h,first=cp(chosen.o);
 const id=`${split}-remote-v28-g${g}-w${v}-arm${arm}`;h.id=id;
 const head={schema:'driv3r/corpus_factory/remote-error-alias-v28/episode',episode_id:id,split,world_group:`${split}-${g}`,pair_id:`${split}-g${g}-w${v}`,
  synthetic_host:true,world_sha256:c.w.world_sha256,geometry_sha256:c.w.graph_sha256,
  difficulty:{graph_nodes:c.w.nodes.length,remote_distance:chosen.remote_distance,first_action_regret:paired.regrets[arm],older_than_64:true,
   scripted_prelabel_actions:320-h.remaining,post_paint_distractor_moves:chosen.after_paint_moves},
  hidden_intervention:{target_cell:chosen.paint_cell,wrong_material:true,target_index:chosen.targetIndex},
  initial_history:cp(h.history),observation_keys:Object.keys(first),
  initial_complete_observation_sha256:sha(first),restricted_alias_sha256:sha(restricted(first))};
 let ev=h.history.events.length,ps=h.history.positions.length,steps=0,erase=0,paint=0,retrace=0;
 while(!h.done){const o=v26.observe(h),rank=c.p.rank(s(h))[0];if(!rank||!Number.isFinite(rank.cost)||steps>120)throw Error('oracle unreachable');
  const record={schema:'driv3r/corpus_factory/remote-error-alias-v28/decision',episode_id:id,step:steps,
   state:Object.fromEntries(Object.entries(o).filter(([k])=>k!=='history')),
   events_added:cp(o.history.events.slice(ev)),positions_added:o.history.positions.slice(ps),model_input_sha256:sha(o),target:rank.action,training_weight:1};
  ev=o.history.events.length;ps=o.history.positions.length;
  const receipt=v26.apply(h,rank.action);if(!receipt.accepted)throw Error('oracle illegal');record.receipt=receipt;emit(record,o);
  if(rank.action==='ERASE')erase++;if(rank.action.startsWith('PAINT_'))paint++;if(rank.action==='BACKTRACK')retrace++;steps++;
 }
 if(erase!==1||!h.done||v26.correct(c.w,h.status).wrong||v26.correct(c.w,h.status).count<2)throw Error('remote contamination not corrected');
 if(steps!==chosen.rank[0].cost)throw Error('oracle shortest suffix mismatch');
 return {...head,terminal_success:true,first_action:chosen.rank[0].action,first_action_regret:paired.regrets[arm],steps,erases:erase,paint,retrace};
}
function writeShard(dest,file,rows){const bytes=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dest,file),bytes);return {rows:rows.length,bytes:bytes.length,sha256:sha(bytes)};}
function generate(dir,{quick=false}={}){
 if(fs.existsSync(dir))throw Error('refuse output overwrite');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},owners=new Map(),worldOwners=new Map();
 for(const [split,n] of Object.entries(S)){
  const episodes=[],decisions=[],seen=new Map(),pairs=[],missed=[],groups=quick?1:n;
  for(let i=0;i<groups;i++){const g=G[split][i],fam=v26.family(split,g);
   for(let v=0;v<fam.length;v++){
    const c=fam[v],p=pairFor(c,split,g,v);if(!p){missed.push({g,v});continue;}
    const worldOwner=worldOwners.get(c.w.world_sha256);if(worldOwner&&worldOwner!==split)throw Error('world split leakage');worldOwners.set(c.w.world_sha256,split);
    if(p.cases[0].rank[0].action===p.cases[1].rank[0].action)throw Error('not discriminative');
    pairs.push({pair_id:`${split}-g${g}-w${v}`,first_actions:p.cases.map(x=>x.rank[0].action),regrets:p.regrets,rest:p.rest_cell,remote_distance:p.cases.map(x=>x.remote_distance)});
    for(let arm=0;arm<2;arm++)episodes.push(replay(c,split,g,v,p,arm,(row,obs)=>{
     const old=seen.get(row.model_input_sha256);if(old&&old!==row.target)throw Error('conflicting full-input target');
     const owner=owners.get(row.model_input_sha256);if(owner&&owner!==split)throw Error('cross split exact input');owners.set(row.model_input_sha256,split);
     if(old&&split==='TRAIN')row.training_weight=0;else if(!old)seen.set(row.model_input_sha256,row.target);
     decisions.push(row);
    }));
   }
  }
  if(!pairs.length)throw Error(`zero strict pairs for ${split}`);
  const ef=split+'_episodes.jsonl',df=split+'_decisions.jsonl';shards[ef]=writeShard(dir,ef,episodes);shards[df]=writeShard(dir,df,decisions);
  splits[split]={groups,group_ids:G[split].slice(0,groups),worlds_considered:groups*8,pairs:pairs.length,missing_strict_worlds:missed,
   episodes:episodes.length,decisions:decisions.length,exact_duplicates:decisions.length-seen.size,zero_weight:decisions.filter(x=>x.training_weight===0).length,
   first_action_distribution:episodes.reduce((m,e)=>(m[e.first_action]=(m[e.first_action]||0)+1,m),{}),
   min_regret:Math.min(...episodes.map(x=>x.first_action_regret)),
   min_remote_distance:Math.min(...episodes.map(x=>x.difficulty.remote_distance)),erases:episodes.reduce((n,e)=>n+e.erases,0),
   total_aged_prelabel_actions:episodes.reduce((n,e)=>n+e.difficulty.scripted_prelabel_actions,0)};
 }
 const manifest={schema:'driv3r/corpus_factory/remote-error-alias-v28/manifest',synthetic_host_not_native_CyberSand:true,
  generator:'remote_error_alias_v28.js',generator_sha256:sha(fs.readFileSync(__filename)),dependencies:['causal_gate_v26.js','outcome_forks_v25.js'].map(file=>({file,sha256:sha(fs.readFileSync(path.join(__dirname,file)))})),
  oracle:'v26 exhaustive reverse BFS; action labels minimal successful terminal suffix',quick,paired_counterfactual:true,
  invariant:'Within each pair current public state, first public map, and last 64 event/position receipts are identical, whereas the remotely damaged target and optimal first action differ with regret >=2.',
  caution:'Intervention, mode, pair ID, difficulty, target location, split and oracle cost are metadata ONLY and must never be passed into model observation. All prelabel actions scripted; postlabel actions oracle, NOT model rollouts.',
  split_worlds_disjoint:true,canonical_CyberSand_ABI_verified:false,slm_training_or_inference_done:false,
  splits,shards};fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={route,move,prepare,restricted,pairFor,replay,generate};
if(require.main===module){if(!process.argv[2])throw Error('usage node remote_error_alias_v28.js NEW_OUTPUT [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
