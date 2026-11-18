'use strict';
// Supplemental v39: remote obstruction announcements that change optimal
// legal first actions despite identical immediate root geometry.
// Synthetic game Host only; no neural policy, no CyberSand native runtime.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const fail=(x,s)=>{if(!x)throw Error(s);};
const SPLITS={TRAIN:4,DEV:2,HELDOUT:2},PHASES=['CLEAR','BLOCKED'];
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
function caseAt(w,split,g,v,arm,phase){
 const initial=base.prepare(w,split,g,v,arm).h;initial.remaining=1200;
 const before=host.observe(initial),shadow=obstruct(w,phase);
 const surveyed=initial.history.events[0];fail(surveyed.kind==='REMOTE_SURVEY','lost initial host survey');
 initial.w=shadow;
 // Environmental transition injected by synthetic host at same moment in both forks.
 initial.history.events.push({kind:'OBSTRUCTION_TRANSITION',accepted:true,edges:closures,blocked:phase==='BLOCKED',cause:'REMOTE_ACTUATOR'});
 const after=host.observe(initial);
 fail(JSON.stringify(before.legal_actions)===JSON.stringify(after.legal_actions),'root legal actions diverged');
 fail(JSON.stringify(before.local_view)===JSON.stringify(after.local_view),'root local geometry diverged');
 const input=view.compress(after);input.obstruction={kind:'public_remote_edge',edges:closures.map(edge=>edge.map(n=>w.nidx.get(n))),blocked:phase==='BLOCKED'};
 const planner=base.oracle(shadow),rank=planner.rank([initial.pos,initial.prev,initial.status,initial.latched]);
 fail(rank.length&&Number.isFinite(rank[0].cost),'unreachable changed world');
 return {h:initial,w:shadow,planner,rank,input,phase};
}
function replay(c){
 const h=c.h,rows=[],trajectory=[],initialCost=c.rank[0].cost;
 while(!h.done){
  fail(rows.length<180,'nonterminating recovery');
  const before=host.observe(h),rank=c.planner.rank([h.pos,h.prev,h.status,h.latched]),best=rank[0];
  fail(Number.isFinite(best.cost),'unreachable oracle state');
  const input=view.compress(before);input.obstruction=c.input.obstruction;
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
  const stripped=x=>{const y=JSON.parse(JSON.stringify(x));delete y.obstruction;return y;};
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
      obstruction_ablation_alias_sha256:sha((()=>{let x={...p.a.input};delete x.obstruction;return x;})()),synthetic_not_native:true});
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
 const manifest={schema:'driv3r/corpus_factory/obstruction-replan-v39/manifest',synthetic_not_native_CyberSand:true,
   generator:'obstruction_replan_v39.js',generator_sha256:sha(fs.readFileSync(__filename)),dependencies:dependencies.map(file=>({file,sha256:sha(fs.readFileSync(path.join(__dirname,file)))})),
   model_contract:'Only input goes to learner. action is teacher label; weight=0 excludes duplicate TRAIN records. challenge is evaluator-only.',
   environmental_intervention:'same history/prelude and local root legal view; host changes remote east junction edges (1,0)-(2,0) and (1,0)-(1,1) before decision, public obstruction bulletin; real remote detour remains available',
   caveats:['Synthetic symbolic host, not native CyberSand','Host-dispatched remote actuator, not learned agent action','Graph motif still short deterministic loops','No named SLM run or latency/RSS'],quick,splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,pick,replay,caseAt,makeWorld,obstruct,sha};
if(require.main===module){const [out,flag]=process.argv.slice(2);fail(out,'usage: node obstruction_replan_v39.js OUT [--quick]');console.log(JSON.stringify(generate(path.resolve(out),{quick:flag==='--quick'}).splits,null,2));}
