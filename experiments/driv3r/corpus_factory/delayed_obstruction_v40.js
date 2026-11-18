'use strict';
// Supplemental v39: remote obstruction announcements that change optimal
// legal first actions despite identical immediate root geometry.
// Synthetic game Host only; no neural policy, no CyberSand native runtime.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const fail=(x,s)=>{if(!x)throw Error(s);};
const SPLITS={TRAIN:4,DEV:2,HELDOUT:2};
const closures=[['1,0','2,0'],['1,0','1,1']];
function makeWorld(split,g,v){
 const orig=base.world(split,g,v),adj=new Map(orig.nodes.map((p,i)=>[p,new Set(orig.adjacency[i].map(j=>orig.nodes[j]))]));
 const join=(a,b)=>{fail(adj.has(a)&&adj.has(b),'missing corridor node');adj.get(a).add(b);adj.get(b).add(a);};
 adj.set('2,1',new Set());adj.set('3,1',new Set());
 join('1,1','2,1');join('2,1','3,1');join('3,1','3,0');
 const nodes=[...adj.keys()].sort(),nidx=new Map(nodes.map((s,i)=>[s,i]));
 const adjacency=nodes.map(p=>[...adj.get(p)].sort().map(n=>nidx.get(n)));
 const target=orig.target.slice(),recipes=orig.recipes.slice(),gate=orig.rule.gate_edge.map(s=>nidx.get(s));
 const w={...orig,nodes,nidx,adjacency,target,recipes,root:nidx.get('0,0'),previous:nidx.get('0,-1'),exit:nidx.get(orig.nodes[orig.exit]),gate,
   geometry_sha256:sha({nodes,adjacency}),world_sha256:sha({nodes,adjacency,target,recipes,rule:orig.rule})};
 fail(w.root!==undefined&&w.exit!==undefined&&w.gate.every(x=>x!==undefined),'bad graph');return w;
}
function obstruct(w,phase){
 const nodes=w.nodes,nidx=w.nidx;const adjacency=w.adjacency.map(x=>x.slice());
 if(phase==='BLOCKED')for(const edge of closures){const [i,j]=edge.map(k=>nidx.get(k));adjacency[i]=adjacency[i].filter(n=>n!==j);adjacency[j]=adjacency[j].filter(n=>n!==i);}
 return {...w,adjacency,phase};
}
function forgetObstruction(input){
 const x=JSON.parse(JSON.stringify(input));
 x.durable=x.durable.filter(e=>e[1]!=='REMOTE_OBSTRUCTION');
 return x;
}
function temporalInput(obs){
 const x=view.compress(obs);
 const changes=obs.history.events.flatMap((e,i)=>e.kind==='OBSTRUCTION_TRANSITION'&&e.accepted?
   [[i,'REMOTE_OBSTRUCTION',e.blocked?1:0,e.edges.map(pair=>pair.map(k=>x.cells.indexOf(k)))]]:[]);
 fail(changes.length===1,'expected one accepted obstruction transition');
 fail(changes[0][3].every(pair=>pair.every(i=>i>=0)),'unknown obstruction edge');
 x.v='synthetic-v40-delayed-obstruction';
 x.durable.push(...changes);
 x.durable.sort((a,b)=>a[0]-b[0]);
 return x;
}
function caseAt(w,split,g,v,arm,phase){
 // Generate the same legally accepted historical wrong-paint preparation as v39.
 // At the start of the final 160 accepted movement actions, have the *host*
 // change the remote junction, then replay those actions under the altered graph.
 // This avoids falsely backdating a transition after the fact.
 const original=base.prepare(w,split,g,v,arm).h;
 const suffixCount=160,events=original.history.events,positions=original.history.positions;
 const tail=events.slice(-suffixCount),prefix=events.slice(0,-suffixCount),prefixPos=positions.slice(0,-suffixCount);
 fail(tail.length===suffixCount&&tail.every(e=>e.kind==='ACTION_ACK'&&e.accepted&&!e.action.startsWith('PAINT')&&e.action!=='ERASE'),'unqualified ageing suffix');
 fail(prefixPos.at(-1)==='0,0'&&prefixPos.length>2,'not at common safe junction before transition');
 const shadow=obstruct(w,phase),h=new host.Host(shadow,split,g,v);
 h.pos=shadow.nidx.get(prefixPos.at(-1));h.prev=shadow.nidx.get(prefixPos.at(-2));
 h.status=original.status;h.latched=original.latched;h.remaining=original.remaining+suffixCount;
 h.history={events:JSON.parse(JSON.stringify(prefix)),positions:prefixPos.slice()};
 fail(h.pos!==undefined&&h.prev!==undefined&&h.latched===1,'bad reconstruction');
 const bulletin={kind:'OBSTRUCTION_TRANSITION',accepted:true,edges:closures,blocked:phase==='BLOCKED',cause:'REMOTE_ACTUATOR'};
 h.history.events.push(bulletin);
 const localBefore=host.observe(h);
 for(let i=0;i<tail.length;i++){
   const e=tail[i],r=host.apply(h,e.action);
   fail(r.accepted&&r.action===e.action&&r.at===e.at&&r.painted_correct===e.painted_correct,'replayed movement differs at '+i);
 }
 fail(h.pos===original.pos&&h.prev===original.prev&&h.status===original.status&&h.latched===original.latched&&h.remaining===original.remaining,'intervened history changed common state');
 const after=host.observe(h),recent=after.history.events.slice(-64);
 fail(!recent.some(e=>e.kind==='OBSTRUCTION_TRANSITION'),'event leaked into recent history');
 fail(sha(recent)===sha(events.slice(-64)),'recent-history alias broken');
 fail(after.history.events.length-prefix.length===suffixCount+1,'obstruction age wrong');
 const input=temporalInput(after),planner=base.oracle(shadow),rank=planner.rank([h.pos,h.prev,h.status,h.latched]);
 fail(rank.length&&Number.isFinite(rank[0].cost),'unreachable changed world');
 return {h,w:shadow,planner,rank,input,phase,eventAge:suffixCount,localBefore};
}
function replay(c){
 const h=c.h,rows=[],trajectory=[],initialCost=c.rank[0].cost;
 while(!h.done){
  fail(rows.length<180,'nonterminating recovery');
  const before=host.observe(h),rank=c.planner.rank([h.pos,h.prev,h.status,h.latched]),best=rank[0];
  fail(Number.isFinite(best.cost),'unreachable oracle state');
  const input=temporalInput(before);
  const receipt=host.apply(h,best.action);fail(receipt.accepted,'obstructed host refused oracle');
  const after=h.done?0:c.planner.rank([h.pos,h.prev,h.status,h.latched])[0].cost;
  fail(best.cost===1+after,'not shortest-successful suffix');
  rows.push({input,action:best.action,weight:1});
  trajectory.push([before.position,best.action,receipt.at]);
 }
 fail(rows.length===initialCost,'suffix length != oracle');
 fail(host.correct(c.w,h.status).wrong===0,'uncorrected contamination');
 return {rows,trajectory,success:h.done,steps:rows.length};
}
function pick(w,split,g,v){
 const found=[],controls=[];
 for(let arm=0;arm<4;arm++){
  const a=caseAt(w,split,g,v,arm,'CLEAR'),b=caseAt(w,split,g,v,arm,'BLOCKED');
  const stripped=forgetObstruction;
  fail(sha(stripped(a.input))===sha(stripped(b.input)),'obstruction-blind input leaked');
  const wrongA=a.rank.find(x=>x.action===b.rank[0].action),wrongB=b.rank.find(x=>x.action===a.rank[0].action);
  const contrast={arm,clear_action:a.rank[0].action,blocked_action:b.rank[0].action,
   clear_regret:wrongA?wrongA.cost-a.rank[0].cost:null,blocked_regret:wrongB?wrongB.cost-b.rank[0].cost:null};
  const pair={a,b,contrast};
  if(a.rank[0].action!==b.rank[0].action && contrast.clear_regret>=2 && contrast.blocked_regret>=2)found.push(pair);
  else controls.push(pair);
 }
 return {found,controls};
}
function save(dest,name,rows){const buf=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(dest,name),buf,{flag:'wx'});return {rows:rows.length,bytes:buf.length,sha256:sha(buf)};}
function generate(dest,{quick=false}={}){
 fail(!fs.existsSync(dest),'refuse overwrite');fs.mkdirSync(dest,{recursive:true});
 const splits={},shards={},geometryOwners=new Map(),modelOwners=new Map();
 for(const [split,num] of Object.entries(SPLITS)){
  let candidatePairs=0,positivePairs=0,negativePairs=0,competing=0,terminal=0,zeroWeight=0;
  const decisions=[],challenges=[],seen=new Map(),rejects=[],geometries=[];
  for(let g=0;g<(quick?1:num);g++)for(let v=0;v<8;v++){
   const w=makeWorld(split,g,v),previous=geometryOwners.get(w.geometry_sha256);
   fail(!previous||previous===split,'cross-split topology leakage');geometryOwners.set(w.geometry_sha256,split);
   const {found,controls}=pick(w,split,g,v);candidatePairs+=4;
   if(!found.length){rejects.push([g,v,'no-strict-contrast']);continue;}
   geometries.push(w.geometry_sha256);
   const chosen=[...found,...controls.slice(0,1)];
   for(const p of chosen){
    const positive=found.includes(p);if(positive)positivePairs++;else negativePairs++;
    const replays=[replay(p.a),replay(p.b)];
    fail(replays.every(r=>r.success),'nonterminal');terminal+=2;
    if(positive){
     competing++;
     fail(p.a.input.legal.includes(p.b.rank[0].action)&&p.b.input.legal.includes(p.a.rank[0].action),'first action only differs via legality');
    }
    challenges.push({split,g,v,arm:p.contrast.arm,world_sha256:w.world_sha256,geometry_sha256:w.geometry_sha256,positive,
      ...p.contrast,steps_clear:replays[0].steps,steps_blocked:replays[1].steps,
      obstruction_ablation_alias_sha256:sha(forgetObstruction(p.a.input)),obstruction_event_age_actions:p.a.eventAge,synthetic_not_native:true});
    for(const replayed of replays)for(const row of replayed.rows){
      const k=sha(row.input),prior=modelOwners.get(k),seenAction=seen.get(k);
      fail(!prior||prior===split,'cross split input overlap');modelOwners.set(k,split);
      fail(seenAction===undefined||seenAction===row.action,'conflicting input label');
      if(seenAction!==undefined&&split==='TRAIN'){row.weight=0;zeroWeight++;}
      else seen.set(k,row.action);
      decisions.push(row);
    }
   }
  }
  fail(positivePairs>0,'no positive obstacle contrasts '+split);
  shards[split+'_policy.jsonl']=save(dest,split+'_policy.jsonl',decisions);
  if(split!=='TRAIN')shards[split+'_challenge.jsonl']=save(dest,split+'_challenge.jsonl',challenges);
  splits[split]={candidate_pairs:candidatePairs,positive_pairs:positivePairs,negative_controls:negativePairs,
    paired_legal_first_action_contrasts:competing,episodes:terminal,decisions:decisions.length,zero_weight:zeroWeight,
    unique_geometries:new Set(geometries).size,rejected_worlds:rejects};
 }
 const dependencies=['causal_shortcut_v32.js','causal_gate_v26.js','outcome_forks_v25.js','bounded_view_v30.js'];
 const manifest={schema:'driv3r/corpus_factory/delayed-obstruction-v40/manifest',synthetic_not_native_CyberSand:true,
   generator:'delayed_obstruction_v40.js',generator_sha256:sha(fs.readFileSync(__filename)),dependencies:dependencies.map(file=>({file,sha256:sha(fs.readFileSync(path.join(__dirname,file)))})),
   model_contract:'Only input goes to learner. action is teacher label; weight=0 excludes duplicate TRAIN records. challenge is evaluator-only.',
   environmental_intervention:'host changes remote east junction edges before 160 accepted legal movement actions; past bulletin remains in durable event history but never in last-64 events; the alternative detour stays reachable',
   caveats:['Synthetic symbolic host, not native CyberSand','Host-dispatched remote actuator, not learned agent action','Deterministic north-root ageing loops; symbolic event interpretation can solve challenge','Fixed remote obstruction junction','No named SLM run or latency/RSS'],quick,splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,pick,replay,caseAt,makeWorld,obstruct,temporalInput,forgetObstruction,sha};
if(require.main===module){const [out,flag]=process.argv.slice(2);fail(out,'usage: node delayed_obstruction_v40.js OUT [--quick]');console.log(JSON.stringify(generate(path.resolve(out),{quick:flag==='--quick'}).splits,null,2));}
