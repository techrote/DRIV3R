#!/usr/bin/env node
'use strict';
/* Corpus Factory v63 — causal remote-gate revisions, misleading older bulletins,
 * partially surveyed branches and physical route regret. Synthetic graph host,
 * NOT native CyberSand. DATA ONLY; no policy training. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const host=require('./unseen_topology_v44.js');
const partial=require('./frontier_memory_v45.js');
const stream=require('./offpolicy_repair_v58.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,s)=>{if(!x)throw Error('V63 '+s)};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const rev={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'};
const COUNT={TRAIN:8,DEV:4,HELDOUT:4};
const DELAY={TRAIN:48,DEV:96,HELDOUT:160};
const PARENTS=['frontier_memory_v45.js','unseen_topology_v44.js','offpolicy_repair_v58.js'];
function routeAction(g,from,to){const [x,y]=g.coords[from],[X,Y]=g.coords[to];return X===x+1?'MOVE_E':X===x-1?'MOVE_W':Y===y+1?'MOVE_S':Y===y-1?'MOVE_N':null;}
// Discovery operates on novel worlds, disjoint by generated index and geometry;
// challenge qualification is separate from training and never returned as input.
function qualifiedWorld(split,index){
 for(let trial=0;trial<70;trial++){
  const sourceIndex=24000+index*101+trial;
  const w=host.scenario(split,sourceIndex);
  try{
   const p0=partial.prelude(w,'positive',false),p1=partial.prelude(w,'positive',true);
   const c0=partial.prelude(w,'control',false),c1=partial.prelude(w,'control',true);
   const a0=partial.teacher(partial.observe(p0)),a1=partial.teacher(partial.observe(p1));
   const b0=partial.teacher(partial.observe(c0)),b1=partial.teacher(partial.observe(c1));
   if(a0===a1||b0!==b1||a0!==b0)continue;
   const open=host.ranks(p0.g,p0.pos,w.goal,-1),closed=host.ranks(p1.g,p1.pos,w.goal,w.pivotal);
   if(a0!==open[0]?.action||a1!==closed[0]?.action||b0!==open[0]?.action)continue;
   const wrong0=open.find(z=>z.action===a1),wrong1=closed.find(z=>z.action===a0);
   if(!wrong0||!wrong1||wrong0.cost-open[0].cost<2||wrong1.cost-closed[0].cost<2)continue;
   if(new Set(p0.positions).size/p0.g.coords.length>.83)continue;
   return {w,sourceIndex,trial};
  }catch(e){if(e.message.includes('no qualified topology'))throw e;}
 }
 throw Error('V63 cannot qualify topology '+split+'/'+index);
}
function detourActions(h,w,length){
 ok(length%2===0,'detour length must be even');
 let pos=h.pos,remaining=length,seg=0;const actions=[];let nontrivial=0;const surveyed=new Set(h.positions);
 while(remaining){
  const maxSteps=Math.min(3+(seg%4),Math.floor(remaining/2));
  const walk=[];let previous=null;
  for(let j=0;j<maxSteps;j++){
   const safe=h.g.adj[pos].filter(([nb,e])=>e!==w.pivotal&&e!==w.control&&surveyed.has(nb)&&nb!==previous)
    .sort((a,b)=>a[0]-b[0]);
   if(!safe.length)break;
   const [next]=safe[(w.seed +seg*13+j*17)%safe.length];
   const a=routeAction(h.g,pos,next);ok(a,'non-cardinal safe detour');
   walk.push(a);previous=pos;pos=next;
  }
  ok(walk.length>0,'no safe branch detour');
  if(walk.length>1)nontrivial++;
  for(let j=walk.length-1;j>=0;j--)actions.push(rev[walk[j]]);
  actions.splice(actions.length-walk.length,0,...walk);
  // Each segment is an excursion and retrace from the identical origin.
  for(let j=walk.length-1;j>=0;j--){const a=rev[walk[j]],neighbor=h.g.adj[pos].find(([nb,e])=>e!==w.pivotal&&e!==w.control&&routeAction(h.g,pos,nb)===a);ok(neighbor,'reverse safety');pos=neighbor[0];}
  remaining-=walk.length*2;seg++;ok(seg<500,'detour loop');
 }
 ok(pos===h.pos&&actions.length===length,'detour endpoint');
 return {actions,nontrivial};
}
function episode(w,kind,oldClosed,newClosed,delay){
 const h=partial.prelude(w,kind,oldClosed),gate=h.gate;
 ok(h.pos===w.s&&h.stage===0&&!h.done,'bad starting state');
 // Old prelude gate state is initially live. Mid-episode revised remote
 // bulletin supersedes it; the visible event is delivered exactly once.
 ok(h.closed===oldClosed,'initial gate state wrong');
 const rec=detourActions(h,w,delay),steps=[];let prior=[],critical=null,final=null;
 function step(a,weight,focus=0){
  const raw=partial.observe(h),input=stream.streamify(raw,prior),idx=steps.length;
  ok(raw.legal.includes(a),'illegal '+a+' at '+idx);
  steps.push({input,action:a,weight,focus_loss_weight:focus});prior=raw.durable;
  const before=h.stage;ok(host.apply(h,a).accepted,'rejected '+a);
  if(before===0&&h.stage===1)final={index:idx,action:a};
 }
 for(let j=0;j<rec.actions.length;j++){
  if(j===delay/2){h.closed=newClosed;h.events.push({kind:'EDGE_GATE',accepted:true,edge:gate,value:newClosed?1:0,at:h.accepted});}
  step(rec.actions[j],0);
 }
 ok(h.pos===w.s&&h.stage===0,'detour did not return');
 const raw=partial.observe(h),input=stream.streamify(raw,prior),action=partial.teacher(raw);
 ok(input.events_delta.length===0,'old bulletin leaked at checkpoint');
 const ranks=host.ranks(h.g,w.s,w.goal,newClosed?gate:-1);
 const chosen=ranks.find(z=>z.action===action);ok(chosen&&chosen.cost===ranks[0]?.cost,'non-optimal first route '+kind+' old='+oldClosed+' new='+newClosed+' teacher='+action+' ranks='+JSON.stringify(ranks));
 critical={index:steps.length,action,input_sha256:H(input),input,regret:null};
 step(action,1,8);
 for(let k=0;!h.done&&k<350;k++){
  const obs=partial.observe(h),a=partial.teacher(obs);
  step(a,1,1);
 }
 ok(h.done&&h.success&&h.stage===2,'host did not finish');
 return {steps,critical,final,detour:rec};
}
function write(root,name,rows,meta){const b=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(root,name),b,{flag:'wx'});meta[name]={sha256:H(b),bytes:b.length,rows:rows.length};}
function generate(root,{quick=false}={}){
 ok(!fs.existsSync(root),'destination exists');fs.mkdirSync(root,{recursive:true});
 const mf={schema:'driv3r/corpus_factory/gate-revision-v63/1',date:'2026-11-19',quick,
  synthetic_not_native_CyberSand:true,generator_sha256:H(fs.readFileSync(__filename)),
  parent_shas:Object.fromEntries(PARENTS.map(n=>[n,H(fs.readFileSync(path.join(__dirname,n)))])),
  visible_contract:'Only ordered steps.input and prior selected actions; no evaluator cert fields, no durable event ledger repeating. All actions are legal MOVE/PAINT/ERASE/STOP.',
  scenario:'Old remotely reported gate state can be superseded by newer gate bulletin. Positive pivotal obstruction changes optimal first route; irrelevant control gate does not. Physically executed partial-survey, branch exploration, retrace and two painted goals.',
  evaluation:'Positive cases: current-only 50% critical action ceiling, stale-first-gate controller 50% critical accuracy, latest-event controller 100% symbolic baseline. Wrong first turn incurs >=2 legal moves of regret in full graph. Irrelevant gate is negative control.',
  leakage:'Distinct split-specific procedural topology sizes/seeds; evaluator certificates and group arm labels are separate. No hidden gate state in model input. Redundant TRAIN noncritical same-state same-action rows are zero-weight; intentionally contradictory critical pairs are preserved.',
  intervention:'Old history is host-executed before stream; latest gate event delivered at the midpoint of the visible sequential episode; detours legal, reversible, masked. Those detours are not demonstrations of learned exploration.',
  files:{},splits:{},limitations:['Synthetic host != native CyberSand, and these are oracle-labelled episodes, not neural closed-loop successes.',
    'Latest-gate symbolic event-memory policy solves routing; navigation beyond surveyed frontier is not proved learned.',
    'Host-executed detours are zero-weight scripted interventions; no target SLM weight update, neural latency/RSS or immutable Actions dataset artifact.']};
 const geometry=new Map();let total=0,positive=0,negative=0,regrets=[];
 for(const [split,n] of Object.entries(COUNT)){
  const streams=[],certs=[],seenHistory=new Set(),seenStateAction=new Set();let masked=0,dups=0,criticalPos=0,criticalNeg=0;
  for(let i=0;i<(quick?Math.min(1,n):n);i++){
   const {w,sourceIndex,trial}=qualifiedWorld(split,i),owner=geometry.get(w.geometry_sha256);
   ok(!owner||owner===split,'cross-split geometry leakage');geometry.set(w.geometry_sha256,split);
   const arms=[];
   for(const kind of ['positive','control'])for(const old of [false,true])for(const newest of [false,true]){
    let r;try{r=episode(w,kind,old,newest,DELAY[split]);}catch(e){e.message += ' split='+split+' worldIndex='+i+' sourceIndex='+sourceIndex;throw e;}arms.push({kind,old,newest,r});
   }
   // Different gate history; identical current observation and legal action menu.
   const input=arms[0].r.critical.input;
   ok(arms.every(x=>eq(x.r.critical.input,input)),'eight-arm instantaneous state differs');
   const pos=arms.filter(x=>x.kind==='positive'),ctl=arms.filter(x=>x.kind==='control');
   ok(pos.filter(x=>x.newest===false).every(x=>x.r.critical.action===pos[0].r.critical.action),'open route differed');
   ok(pos.filter(x=>x.newest===true).every(x=>x.r.critical.action===pos[1].r.critical.action),'closed route differed');
   ok(pos[0].r.critical.action!==pos[1].r.critical.action,'positive reroute absent');
   ok(ctl.every(x=>x.r.critical.action===ctl[0].r.critical.action),'irrelevant control changed route');
   const primary=host.ranks(host.hydrate(w),w.s,w.goal,-1),closed=host.ranks(host.hydrate(w),w.s,w.goal,w.pivotal);
   const wrongOpen=primary.find(z=>z.action===pos[1].r.critical.action),wrongClosed=closed.find(z=>z.action===pos[0].r.critical.action);
   ok(wrongOpen&&wrongClosed&&wrongOpen.cost-primary[0].cost>=2&&wrongClosed.cost-closed[0].cost>=2,'non-regrettable route fork');
   regrets.push(wrongOpen.cost-primary[0].cost,wrongClosed.cost-closed[0].cost);
   for(const arm of arms){
    const {r}=arm,ep=streams.length,critical=r.critical;
    // Loss mask duplicate exact sequence prefixes within TRAIN. Do not mask
    // counterfactual critical decisions simply because current state aliases.
    let digest='epoch';for(let j=0;j<r.steps.length;j++){
     const row=r.steps[j];digest=H(digest+JSON.stringify([row.input,row.action]));
     if(split==='TRAIN'&&row.weight&&j!==critical.index){
      const stateAction=H([row.input,row.action]);
      if(seenHistory.has(digest)||seenStateAction.has(stateAction)){row.weight=0;row.focus_loss_weight=0;dups++;}
      else {seenHistory.add(digest);seenStateAction.add(stateAction);}
     }
    }
    ok(r.steps[critical.index].weight===1,'critical masked');
    streams.push({steps:r.steps});masked+=r.detour.actions.length;
    if(arm.kind==='positive'){positive++;criticalPos++;}else{negative++;criticalNeg++;}
    certs.push({episode:ep,group:split+':'+i,split,world_index:i,source_index:sourceIndex,source_trial:trial,
     seed:w.seed,geometry_sha256:w.geometry_sha256,kind:arm.kind,old_closed:arm.old,new_closed:arm.newest,
     detour_actions:r.detour.actions.length,revision_step:DELAY[split]/2,nontrivial_excursions:r.detour.nontrivial,
     critical_step:critical.index,critical_action:critical.action,critical_input_sha256:critical.input_sha256,
     host_terminal_success:true,route_wrong_turn_regret:arm.kind==='positive'?(arm.newest?wrongClosed.cost-closed[0].cost:wrongOpen.cost-primary[0].cost):0});
   }
  }
  const filename=split+(split==='TRAIN'?'_gate_stream.jsonl':'_EVALUATOR_ONLY_gate_stream.jsonl');
  write(root,filename,streams,mf.files);write(root,split+'_gate_certificates_EVALUATOR_ONLY.jsonl',certs,mf.files);
  const splitRows=streams.reduce((sum,x)=>sum+x.steps.length,0);
  mf.splits[split]={worlds:quick?Math.min(1,n):n,groups:certs.length/8,episodes:streams.length,
   successful_host_episodes:streams.length,decisions:splitRows,positive_pivotal_forks:criticalPos,
   irrelevant_gate_negative_controls:criticalNeg,zero_weight_detour_actions:masked,
   duplicate_train_state_action_labels_masked:dups,delay_actions:DELAY[split]};total+=splitRows;
 }
 mf.total_decisions=total;mf.total_groups=Object.values(mf.splits).reduce((z,s)=>z+s.groups,0);
 mf.distinct_world_geometries=geometry.size;mf.positive_episodes=positive;mf.control_episodes=negative;
 mf.wrong_first_turn_minimum_regret=Math.min(...regrets);
 fs.writeFileSync(path.join(root,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){const [dest,mode]=process.argv.slice(2);ok(dest,'usage: node gate_revision_v63.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dest),{quick:mode==='--quick'}).splits));}
module.exports={qualifiedWorld,episode,generate,COUNT,DELAY};
