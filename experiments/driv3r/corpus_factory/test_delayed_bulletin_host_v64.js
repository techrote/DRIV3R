#!/usr/bin/env node
'use strict';
/** Independent v64 synthetic host auditor. Does not load the v64 generator or
 * evaluator certificates as policy inputs. Reconstructs full host transitions,
 * exact model-visible event streams, all labels and historical timestamps. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const partial=require('./frontier_memory_v45.js');
const stream=require('./offpolicy_repair_v58.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const check=(x,m)=>{if(!x)throw Error('V64 AUDIT '+m)};
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const SPLITS=['TRAIN','DEV','HELDOUT'];
const DIR={MOVE_N:[0,-1],MOVE_S:[0,1],MOVE_E:[1,0],MOVE_W:[-1,0]};
function replay(world,cert,ep,delay){
 const h=partial.prelude(world,cert.kind,cert.old_closed),old=h.events.filter(x=>x.kind==='EDGE_GATE').at(-1);
 check(old&&old.value===Number(cert.old_closed)&&old.at===cert.stale_receipt_measurement_at,'old source measurement');
 check(h.pos===world.s&&h.stage===0&&!h.done,'initial host state');
 const visited=new Set(h.positions),gated=new Set([world.pivotal,world.control]);
 let previous=[],critical=null,detour=0,terminal=0;
 for(let t=0;t<ep.steps.length;t++){
  if(t===delay/2){h.closed=cert.revised_closed;h.events.push({kind:'EDGE_GATE',accepted:true,edge:h.gate,value:Number(cert.revised_closed),at:h.accepted});}
  if(t===cert.stale_delivery_step){h.events.push({...old});}
  const r=ep.steps[t],raw=partial.observe(h),observed=stream.streamify(raw,previous);
  check(eq(r.input,observed),'pre-action input mismatch ep '+cert.episode+' step '+t);
  check(!Object.hasOwn(observed,'durable')&&!Object.hasOwn(observed,'gate')&&!Object.hasOwn(observed,'closed'),'privileged state leak');
  check(raw.legal.includes(r.action),'illegal action '+t);
  const events=observed.events_delta.filter(x=>x[1]==='EDGE_GATE');
  if(t===0)check(events.length===2&&events.at(-1)[4]===Number(cert.old_closed),'prelude old event mismatch');
  else if(t===delay/2)check(events.length===1&&events[0][4]===Number(cert.revised_closed),'new physical measurement mismatch');
  else if(t===cert.stale_delivery_step)check(events.length===1&&events[0][0]===old.at&&events[0][4]===old.value,'late old measurement mismatched');
  else check(events.length===0,'unexpected gate event at '+t);
  if(t<delay){
   check(r.weight===0&&r.focus_loss_weight===0&&r.action in DIR,'unmasked or nonmove intervention '+t);
   const xy=h.g.coords[h.pos],d=DIR[r.action];
   const valid=h.g.adj[h.pos].some(([v,e])=>!gated.has(e)&&visited.has(v)&&h.g.coords[v][0]===xy[0]+d[0]&&h.g.coords[v][1]===xy[1]+d[1]);
   check(valid,'detour crossed obstacle or unexplored node');detour++;
  }
  if(t===delay){
   check(h.pos===world.s&&h.stage===0&&observed.events_delta.length===0,'critical observation leak');
   check(r.weight===1&&r.focus_loss_weight===8,'critical loss mask');
   check(H(observed)===cert.critical_input_sha256&&r.action===cert.critical_action,'critical signed label mismatch');
   const latestByTime=raw.durable.filter(x=>x[1]==='EDGE_GATE').reduce((a,b)=>b[0]>a[0]?b:a);
   const latestByArrival=raw.durable.filter(x=>x[1]==='EDGE_GATE').at(-1);
   check(Boolean(latestByTime[4])===cert.revised_closed&&Boolean(latestByTime[4])===cert.highest_timestamp_state,'valid-time interpretation');
   check(Boolean(latestByArrival[4])===cert.last_arrival_state,'arrival interpretation');
   const ranks=host.ranks(h.g,h.pos,world.goal,cert.revised_closed?h.gate:-1);
   check(ranks[0]?.action===r.action,'wrong optimal physical first action');
   // Independent visible-survey teacher: gate reports sorted by original measurement timestamp.
   const visible={...raw,durable:[...raw.durable.filter(x=>x[1]!=='EDGE_GATE'),
    ...raw.durable.filter(x=>x[1]==='EDGE_GATE').sort((a,b)=>a[0]-b[0])]};
   check(partial.teacher(visible)===r.action,'cannot derive optimal route from visible measurement timestamps');
   critical={input:observed,action:r.action,latestByTime:Boolean(latestByTime[4]),latestByArrival:Boolean(latestByArrival[4])};
  }
  previous=raw.durable;const before=h.stage;
  check(host.apply(h,r.action).accepted,'action rejected '+t);
  if(before===0&&h.stage===1)terminal++;
 }
 check(h.done&&h.success&&h.stage===2&&terminal===1,'host objective failed');
 check(critical&&detour===delay&&cert.loss_masked_detour_steps===delay&&cert.critical_step===delay,'detour or critical mismatch');
 return {critical,decisions:ep.steps.length};
}
function audit(out){
 const m=JSON.parse(fs.readFileSync(path.join(out,'manifest.json')));
 check(m.schema==='driv3r/corpus_factory/delayed-bulletin-v64/1','schema mismatch');
 for(const [name,meta] of Object.entries(m.files)){
  const b=fs.readFileSync(path.join(out,name));check(b.length===meta.bytes&&H(b)===meta.sha256,'bad shard hash '+name);
  check(read(path.join(out,name)).length===meta.rows,'wrong row count '+name);
 }
 const geom=new Map(),controls=new Map();let episodes=0,groups=0,decisions=0,positives=0,irrelevant=0,arrivalCorrect=0,timeCorrect=0,memorylessCorrect=0,staleFails=0,wrongRegretMin=Infinity;
 for(const split of SPLITS){
  const name=split+(split==='TRAIN'?'_delayed_bulletin_stream.jsonl':'_EVALUATOR_ONLY_delayed_bulletin_stream.jsonl');
  const eps=read(path.join(out,name)),certs=read(path.join(out,split+'_delayed_bulletin_certificates_EVALUATOR_ONLY.jsonl'));
  check(eps.length===certs.length&&eps.length===m.splits[split].episodes&&eps.length%16===0,'split episodes');
  let splitDecisions=0,splitPos=0,splitArrival=0,splitTime=0,splitCurrent=0,splitStaleRevision=0;
  for(let base=0;base<eps.length;base+=16){
   const all=certs.slice(base,base+16),sourceIndex=all[0].source_world_index,world=host.scenario(split,sourceIndex);
   check(world.seed===all[0].seed&&world.geometry_sha256===all[0].geometry_sha256,'incorrect source world');
   const prev=geom.get(world.geometry_sha256);check(!prev||prev===split,'geometry leakage');geom.set(world.geometry_sha256,split);
   const forks=[];
   for(let i=0;i<16;i++){
    const c=all[i],kind=i<8?'positive':'control',old=Math.floor((i%8)/4)===1,rev=Math.floor((i%4)/2)===1,delivery=i%2?'stale-last':'fresh-last';
    check(c.episode===base+i&&c.group===split+':'+(base/16)&&c.split===split&&c.kind===kind&&c.old_closed===old&&c.revised_closed===rev&&c.delivery===delivery,'crossed arm not preserved');
    check(c.geometry_sha256===world.geometry_sha256&&c.host_terminal_success,'certificate source mismatch');
    const at=delivery==='fresh-last'?Math.floor(m.splits[split].long_delay_actions/4):m.splits[split].late_stale_delivery_step;
    check(c.revision_step===m.splits[split].long_delay_actions/2&&c.stale_delivery_step===at,'revision injection position');
    const r=replay(world,c,eps[base+i],m.splits[split].long_delay_actions);
    forks.push({...r.critical,...c});splitDecisions+=r.decisions;episodes++;
   }
   check(forks.every(x=>eq(x.input,forks[0].input)),'critical current state differs across arms');
   const pos=forks.slice(0,8),neg=forks.slice(8);
   check(pos.filter(x=>!x.revised_closed).every(x=>x.action===pos[0].action),'open action changed');
   check(pos.filter(x=>x.revised_closed).every(x=>x.action===pos[2].action),'closed action changed');
   check(pos[0].action!==pos[2].action,'no causal route contrast');
   check(neg.every(x=>x.action===neg[0].action),'irrelevant bulletin changed optimal action');
   const open=host.ranks(host.hydrate(world),world.s,world.goal,-1),closed=host.ranks(host.hydrate(world),world.s,world.goal,world.pivotal);
   const wrongOpen=open.find(x=>x.action===pos[2].action),wrongClosed=closed.find(x=>x.action===pos[0].action);
   check(wrongOpen&&wrongClosed&&wrongOpen.cost-open[0].cost>=2&&wrongClosed.cost-closed[0].cost>=2,'missing physical wrong-turn regret');
   wrongRegretMin=Math.min(wrongRegretMin,wrongOpen.cost-open[0].cost,wrongClosed.cost-closed[0].cost);
   for(const p of pos){
    positives++;splitPos++;
    const rightTime=p.latestByTime===p.revised_closed,rightArrival=p.latestByArrival===p.revised_closed;
    timeCorrect+=Number(rightTime);splitTime+=Number(rightTime);
    arrivalCorrect+=Number(rightArrival);splitArrival+=Number(rightArrival);
    const currentRight=p.action===pos[0].action;memorylessCorrect+=Number(currentRight);splitCurrent+=Number(currentRight);
    if(p.delivery==='stale-last'&&p.old_closed!==p.revised_closed){splitStaleRevision++;staleFails+=Number(!rightArrival);}
   }
   irrelevant+=8;groups++;
  }
  check(splitDecisions===m.splits[split].decisions&&splitPos===m.splits[split].positive_pivotal_episodes,'split totals');
  check(splitCurrent/splitPos===m.splits[split].current_only_accuracy&&splitArrival/splitPos===m.splits[split].latest_arrival_accuracy&&splitTime/splitPos===m.splits[split].highest_measurement_time_accuracy,'split control score');
  check(splitStaleRevision===m.splits[split].stale_last_actual_revisions,'split stale revision count');
  decisions+=splitDecisions;
 }
 check(episodes===m.total_episodes&&decisions===m.total_decisions&&geom.size===m.distinct_world_geometries,'aggregate counts');
 check(positives===m.positive_pivotal_episodes&&irrelevant===m.irrelevant_gate_control_episodes&&staleFails===m.stale_last_revision_failures_for_latest_arrival,'aggregate arm counts');
 check(arrivalCorrect===positives*3/4&&timeCorrect===positives&&memorylessCorrect===positives/2,'baseline calculations');
 const summary={pass:true,synthetic_not_native_CyberSand:true,independent_host_replay:true,
  groups,episodes,decisions,distinct_worlds:geom.size,successes:episodes,positive_pivotal_episodes:positives,
  irrelevant_gate_controls:irrelevant,minimum_wrong_turn_regret_moves:wrongRegretMin,
  current_only_accuracy:memorylessCorrect/positives,most_recent_arrival_accuracy:arrivalCorrect/positives,
  highest_measurement_time_accuracy:timeCorrect/positives,stale_last_actual_revision_failures:staleFails};
 console.log(JSON.stringify(summary));return summary;
}
if(require.main===module){check(process.argv[2],'usage: node test_delayed_bulletin_host_v64.js CORPUS_DIR');audit(path.resolve(process.argv[2]));}
module.exports={audit};
