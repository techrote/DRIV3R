'use strict';
/* DRIV3R Corpus Factory v45: host-executed partial-observation map discovery,
 * delayed remote-gate evidence and multi-stage wrong-paint recovery.
 * Research-only SYNTHETIC graph host, NOT native CyberSand or a trained policy. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(c,msg)=>{if(!c)throw Error(msg)};
const DIRS=[['N',0,-1],['E',1,0],['S',0,1],['W',-1,0]];
const SPLITS={TRAIN:64,DEV:24,HELDOUT:24};
const xy=(g,i)=>g.coords[i], key=p=>p.join(','), co=(a,b)=>[a,b].sort((p,q)=>p.localeCompare(q)).join('|');
const direction=(a,b)=>{const d=DIRS.find(x=>x[1]===b[0]-a[0]&&x[2]===b[1]-a[1]);ok(d,'non-cardinal move');return 'MOVE_'+d[0]};
const reverse=a=>({MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_E:'MOVE_W',MOVE_W:'MOVE_E'})[a];
function graphRoute(g,from,to,excluded=-1){const q=[from],parent=new Map([[from,null]]);
 for(let i=0;i<q.length&&!parent.has(to);i++)for(const [v,e] of g.adj[q[i]])if(e!==excluded&&!parent.has(v)){parent.set(v,q[i]);q.push(v)}
 if(!parent.has(to))return null;const nodes=[];for(let v=to;v!==from;v=parent.get(v))nodes.push(v);nodes.push(from);nodes.reverse();return nodes.slice(1).map((v,i)=>direction(xy(g,nodes[i]),xy(g,v)))}
function travel(h,actions){for(const a of actions)ok(host.apply(h,a).accepted,'survey move rejected '+a)}
function survey(h,w){const g=h.g,s=w.s,open=host.ranks(g,s,w.goal,-1)[0],closed=host.ranks(g,s,w.goal,w.pivotal)[0];
 ok(open.action!==closed.action,'source lacks strict counterfactual');
 for(const first of [open,closed]){const route=[first.action,...graphRoute(g,first.neighbor,w.goal,-1)];travel(h,route);
  if(h.painted[w.goal]===undefined)ok(host.apply(h,'PAINT_B').accepted,'wrong-paint prelude failed');
  travel(h,route.slice().reverse().map(reverse));ok(h.pos===s&&h.stage===0,'survey must return to original start');}
}
function prelude(w,kind,closed){const h=host.start(w,kind,false);h.gate=kind==='positive'?w.pivotal:w.control;
 survey(h,w);const neighbor=h.g.adj[h.pos].find(([,e])=>e!==h.gate);
 ok(neighbor,'no safe ageing arm');const outward=direction(xy(h.g,h.pos),xy(h.g,neighbor[0])),back=reverse(outward);
 for(let i=0;i<192;i++){
  if(i===8)external(h,'EDGE_GATE',!closed);
  if(i===24||i===70)external(h,'SENSOR_DIAGNOSTIC',(w.seed+i)%97);
  if(i===48)external(h,'EDGE_GATE',closed);
  ok(host.apply(h,i%2===0?outward:back).accepted,'ageing move refused');
 }
 ok(h.pos===w.s&&h.closed===closed,'fork not restored');return h;
}
// The v44 external function is deliberately unexported; public event structure
// is maintained here without any native host/game modifications.
function external(h,kind,value){if(kind==='EDGE_GATE'){h.closed=!!value;h.events.push({kind,accepted:true,edge:h.gate,value:h.closed?1:0,at:h.accepted});}
 else if(kind==='SENSOR_DIAGNOSTIC')h.events.push({kind,accepted:true,code:value,at:h.accepted});else throw Error('event type');}
function observe(h){const g=h.g,visitedIndices=[...new Set(h.positions)];const visited=new Set(visitedIndices);const nodes=new Map(),edges=new Map();
 for(const v of visitedIndices){nodes.set(key(xy(g,v)),xy(g,v));for(const [b] of g.adj[v]){nodes.set(key(xy(g,b)),xy(g,b));const a=xy(g,v),z=xy(g,b);edges.set(co(key(a),key(z)),[a,z].sort((c,d)=>key(c).localeCompare(key(d))));}}
 const p=xy(g,h.pos),local=DIRS.map(([d,dx,dy])=>{const found=g.adj[h.pos].find(([b])=>xy(g,b)[0]-p[0]===dx&&xy(g,b)[1]-p[1]===dy);return [d,found?xy(g,found[0]):null,!!found&&(found[1]!==h.gate||!h.closed)]});
 const durable=h.events.filter(e=>e.kind==='EDGE_GATE'||e.kind==='SENSOR_DIAGNOSTIC'||e.kind==='GOAL_REVISION').map(e=>e.kind==='EDGE_GATE'?[e.at,'EDGE_GATE',...g.links[e.edge].map(z=>xy(g,z)),e.value]:e.kind==='SENSOR_DIAGNOSTIC'?[e.at,'SENSOR_DIAGNOSTIC',e.code]:[e.at,'GOAL_REVISION',xy(g,e.goal),e.material]);
 const target=h.stage===0?[xy(g,h.w.goal),'A']:h.stage===1?[xy(g,h.w.goal2),'B']:null;
 return {v:'synthetic-v45-partial-survey',p,target,stage:h.stage,legal:host.legal(h),
   local,survey:{visited:visitedIndices.map(z=>xy(g,z)).sort((a,b)=>key(a).localeCompare(key(b))),nodes:[...nodes.values()].sort((a,b)=>key(a).localeCompare(key(b))),edges:[...edges.values()].sort((a,b)=>co(key(a[0]),key(a[1])).localeCompare(co(key(b[0]),key(b[1]))))},
   painted:Object.entries(h.painted).map(([i,mat])=>[xy(g,+i),mat]).sort((a,b)=>key(a[0]).localeCompare(key(b[0]))),
   recent:h.events.filter(e=>e.kind==='ACTION_ACK').slice(-32).map(e=>[e.action,xy(g,e.pos)]),
   recent_positions:h.positions.slice(-32).map(z=>xy(g,z)),durable};}
function ablateEvent(inp){const z=JSON.parse(JSON.stringify(inp));z.durable=z.durable.filter(a=>a[1]!=='EDGE_GATE');return z}
function pathOnSurvey(i,goal){const positions=new Map(i.survey.nodes.map(x=>[key(x),x]));const links=new Map([...positions.keys()].map(x=>[x,[]]));
 const gates=i.durable.filter(e=>e[1]==='EDGE_GATE');const latest=gates[gates.length-1];const blocked=latest&&latest[4]===1?co(key(latest[2]),key(latest[3])):null;
 for(const [a,b] of i.survey.edges){const x=key(a),y=key(b);if(co(x,y)===blocked)continue;links.get(x).push(y);links.get(y).push(x)}
 const source=key(i.p),q=[source],d=new Map([[source,0]]),parent=new Map();
 for(let k=0;k<q.length;k++)for(const n of links.get(q[k])||[])if(!d.has(n)){d.set(n,d.get(q[k])+1);parent.set(n,q[k]);q.push(n)}
 const target=key(goal),visited=new Set(i.survey.visited.map(key));let pick=null;
 if(d.has(target)){pick=target;}else{
  const frontier=q.filter(z=>!visited.has(z));if(!frontier.length)return null;
  frontier.sort((a,b)=>{const A=positions.get(a),B=positions.get(b),heuristic=x=>Math.abs(x[0]-goal[0])+Math.abs(x[1]-goal[1]);
    return (heuristic(A)*2+d.get(a))-(heuristic(B)*2+d.get(b))||a.localeCompare(b)});pick=frontier[0];}
 if(pick===source)return null;let next=pick;while(parent.get(next)!==source){next=parent.get(next);ok(next,'broken surveyed path')}
 return direction(i.p,positions.get(next));
}
function teacher(i){if(i.stage===2)return 'STOP';const [goal,mat]=i.target;
 if(key(i.p)===key(goal)){const painted=i.painted.find(([c])=>key(c)===key(goal));return painted&&painted[1]!==mat?'ERASE':'PAINT_'+mat;}
 const a=pathOnSurvey(i,goal);ok(a,'no reachable discovered frontier/goal');ok(i.legal.includes(a),'planner chose illegal menu action '+a);return a;}
function replay(h,limit=256){const rows=[];for(let n=0;n<limit&&!h.done;n++){const i=observe(h),a=teacher(i);
  ok(i.legal.includes(a),'illegal supervised action');rows.push({input:i,action:a,weight:1});ok(host.apply(h,a).accepted,'host refused teacher');}
 ok(h.done&&h.success,'did not finish host objective '+h.w.split+':'+h.w.index+':'+h.source);return rows;}
function evidence(w,kind){const a=prelude(w,kind,false),b=prelude(w,kind,true),ia=observe(a),ib=observe(b);
 ok(H(ablateEvent(ia))===H(ablateEvent(ib)),'pair leaks closed-state outside old gate bulletin');
 ok(H(ia)!==H(ib),'pair has no visible old memory difference');
 ok(H(ia.recent)===H(ib.recent)&&H(ia.local)===H(ib.local)&&H(ia.legal)===H(ib.legal),'local/recent leak');
 const aa=teacher(ia),bb=teacher(ib),positive=kind==='positive';
 if(positive)ok(aa!==bb,'partial survey eliminated causal routing contrast');else ok(aa===bb,'irrelevant gate changed decision');
 // Structural lower-bound regret independently computed on the full hidden graph.
 const open=host.ranks(a.g,a.pos,w.goal,-1),closed=host.ranks(b.g,b.pos,w.goal,b.gate);
 const wrongOpen=open.find(z=>z.action===bb),wrongClosed=closed.find(z=>z.action===aa);
 if(positive)ok(aa===open[0].action&&bb===closed[0].action&&wrongOpen&&wrongClosed&&wrongOpen.cost-open[0].cost>=2&&wrongClosed.cost-closed[0].cost>=2,'partial-map route fails exact-best causal first-action gate');
 else ok(aa===open[0].action&&bb===closed[0].action,'control routes must be true-optimal');
 const visited=new Set(a.positions).size;ok(visited<a.g.coords.length,'not partial observability');
 const traces=[replay(a),replay(b)];
 return {traces,meta:{kind,positive,split:w.split,source_world_index:w.index,source_seed:w.seed,
   geometry_sha256:w.geometry_sha256,goal:w.goal,goal2:w.goal2,gate_edge:kind==='positive'?w.pivotal:w.control,
   initial_survey_visited:visited,initial_world_nodes:w.g.coords.length,initial_coverage_fraction:+(visited/w.g.coords.length).toFixed(5),
   memory_ablated_input_sha256:H(ablateEvent(ia)),first_open:aa,first_closed:bb,
   wrong_action_regret_open:positive?wrongOpen.cost-open[0].cost:0,
   wrong_action_regret_closed:positive?wrongClosed.cost-closed[0].cost:0,
   episode_steps:traces.map(x=>x.length),teacher_bound:'visible_survey_and_durable_events_only',synthetic:true}};
}
function candidate(split,i){for(let trial=0;trial<72;trial++){
  const w=host.scenario(split,5000+i*211+trial);
  try{const p=evidence(w,'positive'),c=evidence(w,'control');
    if(p.meta.initial_coverage_fraction>.82)continue;
    return {w,p,c,trial};
  }catch(e){if(e.message.includes('no qualified topology'))throw e;}
 }throw Error('no viable partial-survey counterfactual for '+split+':'+i)}
function write(out,name,rows){const b=Buffer.from(rows.map(z=>JSON.stringify(z)).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)}}
function generate(out,{quick=false}={}){ok(!fs.existsSync(out),'output exists');fs.mkdirSync(out,{recursive:true});
 const splits={},shards={},geometry=new Map(),inputs=new Map();let all=0;
 for(const [split,size] of Object.entries(SPLITS)){
  const count=quick?Math.min(2,size):size,policy=[],challenge=[];let decisions=0,zero=0,coverage=0,terminals=0,frontierLabels=0,unknownGoalLabels=0;
  for(let index=0;index<count;index++){
   const v=candidate(split,index),owner=geometry.get(v.w.geometry_sha256);ok(!owner||owner===split,'cross-split geometry leakage');geometry.set(v.w.geometry_sha256,split);
   for(const e of [v.p,v.c]){challenge.push(e.meta);coverage+=e.meta.initial_coverage_fraction;terminals+=2;
    for(const trace of e.traces)for(const row of trace){const k=H(row.input),prev=inputs.get(k);ok(!prev||prev.split===split,'cross-split policy overlap');ok(!prev||prev.action===row.action,'conflicting trainer action');
      if(prev&&split==='TRAIN'){row.weight=0;zero++}else inputs.set(k,{split,action:row.action});
      if(row.action.startsWith('MOVE_')&&row.input.target){const goal=key(row.input.target[0]);
        const observed=new Set(row.input.survey.nodes.map(key));if(!observed.has(goal))unknownGoalLabels++;
        if(!row.input.survey.visited.some(p=>key(p)===goal))frontierLabels++;}
      policy.push(row);}}
  }
  shards[split+'_policy.jsonl']=write(out,split+'_policy.jsonl',policy);
  shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=write(out,split+'_challenge_EVALUATOR_ONLY.jsonl',challenge);
  splits[split]={worlds:count,terminal_episodes:terminals,positive_pairs:count,negative_control_pairs:count,decisions:policy.length,zero_weight:zero,move_labels_before_target_visit:frontierLabels,move_labels_with_goal_unseen:unknownGoalLabels,initial_coverage_mean:+(coverage/(2*count)).toFixed(5)};all+=policy.length;
 }
 const manifest={schema:'driv3r/corpus_factory/partial-survey-memory-v45/manifest',date:'2026-11-18',synthetic_not_native_CyberSand:true,quick,
  generator:'frontier_memory_v45.js',generator_sha256:H(fs.readFileSync(__filename)),parent:'unseen_topology_v44.js',parent_sha256:H(fs.readFileSync(require.resolve('./unseen_topology_v44.js'))),
  note:'Deterministic visible-survey frontier teacher; labels are NOT clairvoyant shortest-path optimals in unseen areas. Gate event is old; observed static edges require physical survey. Per-world paired remote-gate causality plus irrelevant-gate controls.',
  input_contract:'Feed ONLY policy.input to model. policy.action is supervised label; policy.weight is training loss weight. evaluator-only challenge metadata MUST remain sealed.',
  train_test_separation:'Different v45 seeds and grid sizes per split. v45 scenarios use fresh v44 scenario indices >=5000; do not mix v44 HELDOUT topology into training.',
  stats:{total_policy_decisions:all,unique_topologies:geometry.size},splits,shards,caveats:['Synthetic host, not native CyberSand','Full route survey is a scripted accepted-action prelude; exploration after fork uses only observed edges','Frontier planner is an oracle labeler/control, not learned policy','Remote global bulletin conveys gate state; no real sensor latency or physics','No target SLM trained; model inference latency/RSS not measured','Small random-graph family, not multi-genre native maps']};
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
module.exports={SPLITS,candidate,prelude,observe,teacher,pathOnSurvey,ablateEvent,replay,evidence,generate,H};
if(require.main===module){const [dir,quick]=process.argv.slice(2);ok(dir,'usage: node frontier_memory_v45.js NEW_OUTPUT [--quick]');console.log(JSON.stringify(generate(path.resolve(dir),{quick:quick==='--quick'}).splits,null,2))}
