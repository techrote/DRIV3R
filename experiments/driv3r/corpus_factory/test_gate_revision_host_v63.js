#!/usr/bin/env node
'use strict';
/* Independent v63 host receipt audit. Does NOT import generator v63 or rely
 * on evaluator certificates as ground truth for acceptance. Synthetic only. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const host=require('./unseen_topology_v44.js');
const partial=require('./frontier_memory_v45.js');
const stream=require('./offpolicy_repair_v58.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const eq=(x,y)=>JSON.stringify(x)===JSON.stringify(y);
const check=(b,msg)=>{if(!b)throw Error('V63 AUDIT '+msg)};
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const SPLITS=['TRAIN','DEV','HELDOUT'];
function fresh(s,ix){return host.scenario(s,ix);}
function validateOne(w,c,ep,delay){
 const h=partial.prelude(w,c.kind,c.old_closed);
 check(h.pos===w.s&&h.stage===0&&!h.done,'invalid prelude');
 check(h.gate===(c.kind==='positive'?w.pivotal:w.control),'gate source mismatch');
 check(h.closed===c.old_closed,'starting prelude state');
 const originalSurvey=new Set(h.positions),safeEdges=new Set([w.pivotal,w.control]);
 let prior=[],first=null,last=null,detourActions=0;const checks=[];
 for(let t=0;t<ep.steps.length;t++){
  if(t===delay/2){h.closed=c.new_closed;h.events.push({kind:'EDGE_GATE',accepted:true,edge:h.gate,value:c.new_closed?1:0,at:h.accepted});}
  const row=ep.steps[t],raw=partial.observe(h),inp=stream.streamify(raw,prior);
  check(eq(row.input,inp),'host pre-action input differs episode='+c.episode+' step='+t);
  check(!Object.hasOwn(inp,'durable')&&!Object.hasOwn(inp,'closed')&&!Object.hasOwn(inp,'gate'),'hidden host state leaked');
  check(raw.legal.includes(row.action),'action not host-legal');
  if(t===0){
   const e=inp.events_delta.filter(z=>z[1]==='EDGE_GATE');
   check(e.length===2&&e.at(-1)[4]===Number(c.old_closed),'old gate provenance mismatch');
  }else if(t===delay/2){
   const events=inp.events_delta.filter(z=>z[1]==='EDGE_GATE');
   check(events.length===1&&events[0][4]===Number(c.new_closed),'superseding gate bulletin missing at midpoint');
  }else check(!inp.events_delta.some(z=>z[1]==='EDGE_GATE'),'gate event repeated outside expected inputs');
  if(t<delay){
   check(row.action.startsWith('MOVE_')&&row.weight===0&&row.focus_loss_weight===0,'unmasked intervention');
   const original=h.pos,point=h.g.coords[original];
   const d={MOVE_N:[0,-1],MOVE_S:[0,1],MOVE_E:[1,0],MOVE_W:[-1,0]}[row.action];
   const neighbor=h.g.adj[original].find(([v,e])=>!safeEdges.has(e)&&originalSurvey.has(v)&&h.g.coords[v][0]===point[0]+d[0]&&h.g.coords[v][1]===point[1]+d[1]);
   check(neighbor,'detour crossed gate or visited non-survey node');detourActions++;
  }
  if(t===delay){
   check(h.pos===w.s&&h.stage===0,'critical fork not at original start');
   check(inp.events_delta.length===0,'historical events leaked at critical');
   check(H(inp)===c.critical_input_sha256,'critical current input digest');
   check(row.action===c.critical_action&&row.weight===1&&row.focus_loss_weight===8,'critical label/weight mismatch');
   const ranks=host.ranks(h.g,h.pos,w.goal,c.new_closed?h.gate:-1);
   check(ranks[0]?.action===row.action,'critical first action not exact host-optimal');
   first={input:inp,action:row.action,old:c.old_closed,newest:c.new_closed,kind:c.kind,legal:raw.legal};
  }
  prior=raw.durable;
  const stage=h.stage;check(host.apply(h,row.action).accepted,'host refused action');
  if(stage===0&&h.stage===1)last=t;
 }
 check(detourActions===delay&&c.detour_actions===delay&&c.revision_step===delay/2&&c.critical_step===delay,'detour count');
 check(h.done&&h.success&&h.stage===2&&last!==null,'failure or incomplete objectives');
 return {first,decisions:ep.steps.length,detour:detourActions,coverage:originalSurvey.size/h.g.coords.length};
}
function audit(root){
 const mf=JSON.parse(fs.readFileSync(path.join(root,'manifest.json'),'utf8'));
 check(mf.schema==='driv3r/corpus_factory/gate-revision-v63/1','schema mismatch');
 for(const [name,meta] of Object.entries(mf.files)){
  const b=fs.readFileSync(path.join(root,name));
  check(b.length===meta.bytes&&H(b)===meta.sha256&&read(path.join(root,name)).length===meta.rows,'file integrity '+name);
 }
 const geometries=new Map();let groups=0,episodes=0,decisions=0,positives=0,controls=0;
 let realHostSuccess=0,wrongRegretMin=Infinity,criticalCurrentOnlyRight=0,staleFirstRight=0,latestRight=0;
 for(const split of SPLITS){
  const f=split+(split==='TRAIN'?'_gate_stream.jsonl':'_EVALUATOR_ONLY_gate_stream.jsonl');
  const episodesList=read(path.join(root,f)),certs=read(path.join(root,split+'_gate_certificates_EVALUATOR_ONLY.jsonl'));
  check(episodesList.length===certs.length&&episodesList.length===mf.splits[split].episodes&&episodesList.length%8===0,'split count');
  let splitDecisions=0;
  for(let i=0;i<certs.length;i+=8){
   const eight=certs.slice(i,i+8),world=fresh(split,eight[0].source_index);
   check(world.geometry_sha256===eight[0].geometry_sha256&&world.seed===eight[0].seed,'world provenance');
   const oldOwner=geometries.get(world.geometry_sha256);check(!oldOwner||oldOwner===split,'cross-split topology collision');geometries.set(world.geometry_sha256,split);
   const forks=[];
   for(let j=0;j<8;j++){
    const c=eight[j],kind=j<4?'positive':'control',old=j%4>=2,newest=j%2===1;
    check(c.split===split&&c.episode===i+j&&c.group===eight[0].group&&c.kind===kind&&c.old_closed===old&&c.new_closed===newest,'arm ordering');
    check(c.host_terminal_success&&c.geometry_sha256===world.geometry_sha256,'certificate provenance');
    const r=validateOne(world,c,episodesList[i+j],mf.splits[split].delay_actions);
    forks.push(r.first);splitDecisions+=r.decisions;realHostSuccess++;episodes++;
    if(kind==='positive')positives++;else controls++;
   }
   check(forks.every(x=>eq(x.input,forks[0].input)),'identical current critical observation not satisfied');
   check(forks[0].action===forks[2].action&&forks[1].action===forks[3].action&&forks[0].action!==forks[1].action,'positive latest gate causal contrast');
   check(forks.slice(4).every(x=>x.action===forks[4].action),'irrelevant gate negative control');
   const open=host.ranks(host.hydrate(world),world.s,world.goal,-1),closed=host.ranks(host.hydrate(world),world.s,world.goal,world.pivotal);
   const regretOpen=open.find(x=>x.action===forks[1].action),regretClosed=closed.find(x=>x.action===forks[0].action);
   check(regretOpen&&regretClosed&&regretOpen.cost-open[0].cost>=2&&regretClosed.cost-closed[0].cost>=2,'insufficient wrong-turn physical regret');
   wrongRegretMin=Math.min(wrongRegretMin,regretOpen.cost-open[0].cost,regretClosed.cost-closed[0].cost);
   // Deterministic controls are scored only on pivotal positive arms.
   for(let j=0;j<4;j++){
    const expected=forks[j].action;
    criticalCurrentOnlyRight+=Number(expected===forks[0].action);
    staleFirstRight+=Number(expected===forks[(j%4>=2)?3:0].action); // first gate state instead of newest
    latestRight+=Number(expected===forks[j].action);
   }
   groups++;
  }
  check(splitDecisions===mf.splits[split].decisions,'split decision total');decisions+=splitDecisions;
 }
 check(episodes===mf.positive_episodes+mf.control_episodes&&decisions===mf.total_decisions&&groups===mf.total_groups,'aggregate totals');
 check(geometries.size===mf.distinct_world_geometries,'world totals');
 check(criticalCurrentOnlyRight===positives/2&&staleFirstRight===positives/2&&latestRight===positives,'control scores');
 check(wrongRegretMin===mf.wrong_first_turn_minimum_regret,'regret manifest mismatch');
 const result={pass:true,synthetic_host_only:true,independent_host_receipt_replay:true,groups,episodes,
  decisions,world_geometries:geometries.size,host_successes:realHostSuccess,
  positive_pivotal_episodes:positives,irrelevant_control_episodes:controls,
  minimum_wrong_first_move_regret:wrongRegretMin,
  positive_current_only_accuracy:criticalCurrentOnlyRight/positives,
  positive_stale_first_bulletin_accuracy:staleFirstRight/positives,
  positive_latest_event_symbolic_accuracy:latestRight/positives};
 console.log(JSON.stringify(result));return result;
}
if(require.main===module){check(process.argv[2],'usage: node test_gate_revision_host_v63.js CORPUS_DIR');audit(process.argv[2]);}
module.exports={audit};
