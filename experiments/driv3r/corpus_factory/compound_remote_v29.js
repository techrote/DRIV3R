'use strict';
// DRIV3R Corpus Factory v29 — two independent remote wrong-material incidents.
// Delayed-memory action labels; synthetic graph host, NOT native CyberSand.
// Metadata (including contamination locations, pair IDs, oracle regret) is never policy input.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_gate_v26.js');
const v28=require('./remote_error_alias_v28.js');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const copy=x=>JSON.parse(JSON.stringify(x));
const GROUPS={TRAIN:[0,1,2,3,4,5,6,7],DEV:[1,2,3,4],HELDOUT:[1,2,3,4]};
const TWO_HIT=[[0,1],[0,2],[1,2]];
function current(h){return [h.pos,h.prev,h.status,h.latched];}
function prepare(c,split,group,variant,targets,rest){
 const w=c.w,h=new base.Host(w,split,group,variant);h.remaining=460;h.id+=`-compound-${targets.join('')}`;
 let steps=0;const paintReceipts=[];
 for(const i of targets){
  const to=w.nidx.get(w.target[i]);const route=v28.route(w,h.pos,to,h.latched);if(!route)return null;
  for(const n of route){v28.move(h,n);steps++;}
  const wrong=w.recipes[i]==='A'?'PAINT_B':'PAINT_A';
  const r=base.apply(h,wrong);if(!r.accepted)throw Error('scripted paint refused');paintReceipts.push(r);steps++;
 }
 if(h.latched!==1||base.correct(w,h.status).wrong!==2||base.correct(w,h.status).count!==0)throw Error('not exactly two wrong cells');
 const back=v28.route(w,h.pos,rest,1);if(!back)return null;for(const n of back){v28.move(h,n);steps++;}
 const d=w.adjacency[rest].find(n=>base.edgeAllowed(w,rest,n,1)&&!w.target.includes(w.nodes[n]));if(d===undefined)return null;
 // At least 160 accepted actions, identical for every arm, eclipse both paint receipts.
 for(let i=0;i<80;i++){v28.move(h,d);v28.move(h,rest);steps+=2;}
 if(h.pos!==rest)throw Error('incorrect rendezvous');
 const o=base.observe(h),rank=c.p.rank(current(h)),recent=o.history.events.slice(-64);
 const agedOut=recent.every(e=>!e.action?.startsWith('PAINT')&&e.kind!=='GATE_TRANSITION');
 return {h,o,rank,targets,rest,remote:Math.min(...targets.map(i=>v28.route(w,rest,w.nidx.get(w.target[i]),1)?.length??Infinity)),
         agedOut,scripted_actions:steps,paintReceipts};
}
function restricted(o){return v28.restricted(o);}
function choose(c,split,g,variant){
 const w=c.w;
 for(let rest=0;rest<w.nodes.length;rest++){
  if(w.target.includes(w.nodes[rest])||w.adjacency[rest].length<2)continue;
  const items=TWO_HIT.map(t=>prepare(c,split,g,variant,t,rest));
  for(let i=0;i<3;i++)for(let j=i+1;j<3;j++){
   const a=items[i],b=items[j];if(!a||!b||a.remote<3||b.remote<3||!a.agedOut||!b.agedOut)continue;
   if(a.rank[0].action===b.rank[0].action||!Number.isFinite(a.rank[0].cost)||!Number.isFinite(b.rank[0].cost))continue;
   if(sha(restricted(a.o))!==sha(restricted(b.o))||sha(a.o)===sha(b.o))continue;
   const ar=a.rank.find(r=>r.action===b.rank[0].action),br=b.rank.find(r=>r.action===a.rank[0].action);
   const regret=[(ar?.cost??Infinity)-a.rank[0].cost,(br?.cost??Infinity)-b.rank[0].cost];
   if(regret.some(r=>!Number.isFinite(r)||r<2))continue;
   // First qualifying pair in deterministic graph order; avoid exhaustive rest-location tuning.
   return {arms:[a,b],regret,rest:w.nodes[rest],minRegret:Math.min(...regret),minRemote:Math.min(a.remote,b.remote)};
  }
 }
 return null;
}
function replay(c,split,group,v,pair,arm,emit){
 const a=pair.arms[arm],h=a.h,first=copy(a.o);
 const id=`${split}-compound-v29-g${group}-w${v}-arm${arm}`;h.id=id;
 const head={schema:'driv3r/corpus_factory/compound-remote-v29/episode',episode_id:id,split,world_group:`${split}-${group}`,
  pair_id:`${split}-g${group}-w${v}`,synthetic_host:true,world_sha256:c.w.world_sha256,geometry_sha256:c.w.graph_sha256,
  difficulty:{nodes:c.w.nodes.length,remote_distance:a.remote,first_action_regret:pair.regret[arm],scripted_prelabel_actions:a.scripted_actions,
              post_paint_distractor_moves:160,requires_two_corrective_erasures:true},
  hidden_interventions:{wrong_material_cells:a.targets.map(i=>c.w.target[i]),target_indices:a.targets},
  initial_history:copy(h.history),observation_keys:Object.keys(first),initial_complete_observation_sha256:sha(first),
  restricted_alias_sha256:sha(restricted(first))};
 let eventCursor=h.history.events.length,positionCursor=h.history.positions.length,steps=0,erases=0,paints=0,retraces=0;
 while(!h.done){const o=base.observe(h),rank=c.p.rank(current(h))[0];
  if(!rank||!Number.isFinite(rank.cost)||steps>120)throw Error('oracle unreachable');
  const record={schema:'driv3r/corpus_factory/compound-remote-v29/decision',episode_id:id,step:steps,
   state:Object.fromEntries(Object.entries(o).filter(([k])=>k!=='history')),
   events_added:copy(o.history.events.slice(eventCursor)),positions_added:o.history.positions.slice(positionCursor),
   model_input_sha256:sha(o),target:rank.action,training_weight:1};
  eventCursor=o.history.events.length;positionCursor=o.history.positions.length;
  const receipt=base.apply(h,rank.action);if(!receipt.accepted)throw Error('oracle action was refused');record.receipt=receipt;emit(record,o);
  if(rank.action==='ERASE')erases++;if(rank.action.startsWith('PAINT_'))paints++;if(rank.action==='BACKTRACK')retraces++;steps++;
 }
 if(erases!==2||steps!==a.rank[0].cost||!h.done||base.correct(c.w,h.status).wrong||base.correct(c.w,h.status).count<2)
  throw Error('failed minimal two-hit recovery');
 return {...head,terminal_success:true,steps,erases,paints,retraces,first_action:a.rank[0].action,first_action_regret:pair.regret[arm]};
}
function shard(dest,name,records){const b=Buffer.from(records.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dest,name),b);return {rows:records.length,bytes:b.length,sha256:sha(b)};}
function generate(dir,{quick=false}={}){
 if(fs.existsSync(dir))throw Error('refusing overwrite of output directory');fs.mkdirSync(dir,{recursive:true});
 const shards={},splits={},inputOwners=new Map(),worldOwners=new Map(),geometryOwners=new Map();
 for(const [split,ids] of Object.entries(GROUPS)){
  const episodes=[],decisions=[],seen=new Map(),missing=[],pairs=[],groupIds=quick?ids.slice(0,1):ids;
  for(const g of groupIds){const family=base.family(split,g);
   for(let v=0;v<family.length;v++){
    const c=family[v],paired=choose(c,split,g,v);
    if(!paired){missing.push({group:g,variant:v});continue;}
    for(const [map,hash,kind] of [[worldOwners,c.w.world_sha256,'world'],[geometryOwners,c.w.graph_sha256,'geometry']]){
     const old=map.get(hash);if(old&&old!==split)throw Error('cross-split '+kind+' leakage');map.set(hash,split);
    }
    pairs.push({pair_id:`${split}-g${g}-w${v}`,first_actions:paired.arms.map(a=>a.rank[0].action),regrets:paired.regret,
       rest:paired.rest,remote:paired.arms.map(a=>a.remote),wrong_indices:paired.arms.map(a=>a.targets)});
    for(let arm=0;arm<2;arm++)episodes.push(replay(c,split,g,v,paired,arm,(r,o)=>{
     const previous=seen.get(r.model_input_sha256);if(previous&&previous!==r.target)throw Error('conflicting full-input labels');
     const otherSplit=inputOwners.get(r.model_input_sha256);if(otherSplit&&otherSplit!==split)throw Error('cross-split exact input leak');
     inputOwners.set(r.model_input_sha256,split);
     if(previous){if(split==='TRAIN')r.training_weight=0;}else seen.set(r.model_input_sha256,r.target);
     decisions.push(r);
    }));
   }
  }
  if(!pairs.length)throw Error(`no pairs in ${split}`);
  const epName=split+'_episodes.jsonl',deName=split+'_decisions.jsonl';
  shards[epName]=shard(dir,epName,episodes);shards[deName]=shard(dir,deName,decisions);
  splits[split]={groups:groupIds.length,group_ids:groupIds,worlds_considered:groupIds.length*8,pairs:pairs.length,missing_strict_worlds:missing,
   episodes:episodes.length,decisions:decisions.length,erases:episodes.reduce((n,e)=>n+e.erases,0),
   first_action:episodes.reduce((d,e)=>(d[e.first_action]=(d[e.first_action]||0)+1,d),{}),
   min_wrong_choice_regret:Math.min(...episodes.map(x=>x.first_action_regret)),min_remote_distance:Math.min(...episodes.map(x=>x.difficulty.remote_distance)),
   exact_duplicates:decisions.length-seen.size,zero_weight:decisions.filter(r=>r.training_weight===0).length,
   unique_input_count:seen.size};
 }
 const manifest={schema:'driv3r/corpus_factory/compound-remote-v29/manifest',synthetic_host_not_native_CyberSand:true,
  generator:'compound_remote_v29.js',generator_sha256:sha(fs.readFileSync(__filename)),
  dependencies:['remote_error_alias_v28.js','causal_gate_v26.js','outcome_forks_v25.js'].map(file=>({file,sha256:sha(fs.readFileSync(path.join(__dirname,file)))})),
  oracle:'v26 terminal reverse-BFS shortest-successful suffix',quick,split_worlds_disjoint:true,
  challenge:'Two remote wrong-material cells, 160 accepted moves after both, paired invisible-to-last-64 counterfactuals with distinct optimal first recovery actions, >=2 suffix regret.',
  restricted_first_action_theoretical_ceiling:0.5,
  warning:'Only observation reconstructed from state + initial_history + events_added + positions_added is model input. Episode metadata, hidden_interventions, oracle costs, pair IDs, split and targets are held out from policy.',
  limitations:['Synthetic graph, only 3 paint targets and two-hit episodes, script-built prelude.','Symbols, complete remote survey and full history permit deterministic solver.','No real SLM training or native CyberSand replay.'],
  splits,shards};fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={prepare,restricted,choose,replay,generate};
if(require.main===module){if(!process.argv[2])throw Error('usage: node compound_remote_v29.js NEW_DIRECTORY [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
