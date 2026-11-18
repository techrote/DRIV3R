'use strict';
/* DRIV3R Corpus Factory v44: synthetic closed-loop unseen-topology/remote-gate
 * navigation and sequential painting. NOT native CyberSand. Data, not a trainer. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const check=(c,m)=>{if(!c)throw Error(m)};
const cp=x=>JSON.parse(JSON.stringify(x));
const seed=x=>parseInt(H(x).slice(0,8),16)>>>0;
function random(s){let a=s>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296}}
const DIRS=[['N',0,-1],['E',1,0],['S',0,1],['W',-1,0]];
const SPLITS={TRAIN:72,DEV:24,HELDOUT:24};
const ns={TRAIN:[7,8],DEV:[9],HELDOUT:[10,11]};
function shuffle(a,r){for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
function graph(s,n,p){const r=random(s),available=[];
 for(let y=0;y<n;y++)for(let x=0;x<n;x++)if(r()<.87)available.push([x,y]);
 let index=new Map(available.map((a,i)=>[a.join(','),i]));let edges=[];
 for(let i=0;i<available.length;i++)for(const [,dx,dy] of DIRS.slice(1,3)){
   const b=index.get([available[i][0]+dx,available[i][1]+dy].join(','));if(b!==undefined)edges.push([i,b]);}
 shuffle(edges,r);const parent=available.map((_,i)=>i);const find=i=>{while(parent[i]!==i)i=parent[i]=parent[parent[i]];return i};
 const kept=[];for(const [a,b] of edges){const aa=find(a),bb=find(b);if(aa!==bb){parent[aa]=bb;kept.push([a,b]);}else if(r()<p)kept.push([a,b]);}
 const groups=new Map();for(let i=0;i<available.length;i++){const root=find(i);if(!groups.has(root))groups.set(root,[]);groups.get(root).push(i)}
 const biggest=[...groups.values()].sort((a,b)=>b.length-a.length)[0]||[];
 if(biggest.length<n*n*.65)return null;
 biggest.sort((a,b)=>a-b);const renumber=new Map(biggest.map((x,i)=>[x,i]));
 const coords=biggest.map(i=>available[i]),links=kept.filter(([a,b])=>renumber.has(a)&&renumber.has(b)).map(([a,b])=>[renumber.get(a),renumber.get(b)].sort((x,y)=>x-y)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 const occupied=new Set(coords.map(c=>c.join(',')));const protectedCells=[];
 for(let y=0;y<n;y++)for(let x=0;x<n;x++)if(!occupied.has([x,y].join(',')))protectedCells.push([x,y]);
 const adj=coords.map(()=>[]);links.forEach(([a,b],k)=>{adj[a].push([b,k]);adj[b].push([a,k])});adj.forEach(a=>a.sort((a,b)=>a[0]-b[0]));
 return {coords,links,protected:protectedCells,adj,n};
}
function dist(g,goal,blocked){const ds=Array(g.coords.length).fill(Infinity);ds[goal]=0;const q=[goal];
 for(let i=0;i<q.length;i++){const a=q[i];for(const [b,e] of g.adj[a])if(e!==blocked&&ds[b]===Infinity){ds[b]=ds[a]+1;q.push(b)}}return ds}
function ranks(g,s,goal,blocked){const ds=dist(g,goal,blocked);return g.adj[s].filter(([,edge])=>edge!==blocked).map(([neighbor,edge])=>({action:move(g,s,neighbor),cost:1+ds[neighbor],neighbor,edge})).sort((a,b)=>a.cost-b.cost||a.action.localeCompare(b.action))}
function move(g,a,b){const dx=g.coords[b][0]-g.coords[a][0],dy=g.coords[b][1]-g.coords[a][1];const d=DIRS.find(([,x,y])=>dx===x&&dy===y);check(d,'noncardinal edge');return 'MOVE_'+d[0]}
function remote(g,s,edge){return g.links[edge].every(i=>Math.abs(g.coords[i][0]-g.coords[s][0])+Math.abs(g.coords[i][1]-g.coords[s][1])>=3)}
function challenge(g,r){const indices=g.coords.map((_,i)=>i);shuffle(indices,r);
 for(const s of indices.slice(0,Math.min(20,indices.length))){if(g.adj[s].length<2)continue;
  const goals=shuffle(g.coords.map((_,i)=>i).filter(i=>i!==s),r).slice(0,20);
  for(const goal of goals){let open=ranks(g,s,goal,-1);if(open.length<2||open[0].cost<5||!Number.isFinite(open[0].cost)||open[1].cost-open[0].cost<2)continue;
   let path=[],cur=s,ds=dist(g,goal,-1);
   while(cur!==goal&&path.length<g.coords.length){const next=g.adj[cur].find(([i])=>ds[i]===ds[cur]-1);if(!next)break;path.push(next[1]);cur=next[0]}
   if(cur!==goal)continue;
   for(const pivotal of shuffle(path.filter(edge=>remote(g,s,edge)),r)){
    const blocked=ranks(g,s,goal,pivotal);
    if(!blocked.length||!Number.isFinite(blocked[0].cost)||blocked[0].action===open[0].action||blocked.length<2||blocked[1].cost-blocked[0].cost<2)continue;
    const ca=open.find(z=>z.action===blocked[0].action),cb=blocked.find(z=>z.action===open[0].action);
    if(!ca||!cb||ca.cost-open[0].cost<2||cb.cost-blocked[0].cost<2)continue;
    const ctrl=shuffle(g.links.map((_,e)=>e).filter(e=>e!==pivotal&&remote(g,s,e)),r).find(e=>{
      let q=ranks(g,s,goal,e);return q.length&&Number.isFinite(q[0].cost)&&q[0].action===open[0].action&&q[0].cost===open[0].cost&&q.length>1&&q[1].cost-q[0].cost>=2;});
    if(ctrl===undefined)continue;
    const candidates=shuffle(g.coords.map((_,i)=>i).filter(i=>i!==s&&i!==goal),r);
    const goal2=candidates.find(i=>dist(g,i,pivotal)[goal]>=4&&Number.isFinite(dist(g,i,pivotal)[goal])&&Number.isFinite(dist(g,i,ctrl)[goal]));
    if(goal2===undefined)continue;
    return {s,goal,goal2,pivotal,control:ctrl,first_open:open[0].action,first_closed:blocked[0].action,regret_open:ca.cost-open[0].cost,regret_closed:cb.cost-blocked[0].cost};
   }
  }
 }return null;
}
function scenario(split,i){const sizes=ns[split];check(sizes,'invalid split');
 for(let attempt=0;attempt<160;attempt++){
  const id=seed('v44/'+split+'/'+i+'/'+attempt),n=sizes[id%sizes.length],r=random(id);
  const g=graph(id,n,split==='HELDOUT'?.26:split==='DEV'?.32:.40);if(!g)continue;
  const c=challenge(g,r);if(c){const w={g: {coords:g.coords,links:g.links,protected:g.protected,n:g.n},...c,seed:id,split,index:i,attempt,geometry_sha256:H({coords:g.coords,links:g.links,protected:g.protected})};return w}
 }throw Error('no qualified topology '+split+':'+i);
}
function hydrate(w){const g=cp(w.g);g.adj=g.coords.map(()=>[]);g.links.forEach(([a,b],e)=>{g.adj[a].push([b,e]);g.adj[b].push([a,e])});g.adj.forEach(a=>a.sort((x,y)=>x[0]-y[0]));return g}
function start(w,kind,closed){const g=hydrate(w),gate=kind==='positive'?w.pivotal:w.control;
 return {w,g,gate,closed:false,desired:!!closed,pos:w.s,stage:0,painted:{},events:[],positions:[w.s],accepted:0,done:false,success:false,source:kind};}
function legal(h){const a=h.g.adj[h.pos].filter(([,e])=>e!==h.gate||!h.closed).map(([b])=>move(h.g,h.pos,b));
 if(h.painted[h.pos]===undefined)a.push('PAINT_A','PAINT_B');else a.push('ERASE');a.push('STOP');return [...new Set(a)].sort();}
function external(h,kind,value){if(kind==='EDGE_GATE'){h.closed=!!value;h.events.push({kind,accepted:true,edge:h.gate,value:h.closed?1:0,at:h.accepted})}
 else if(kind==='SENSOR_DIAGNOSTIC')h.events.push({kind,accepted:true,code:value,at:h.accepted});else throw Error('event')}
function apply(h,action){check(!h.done,'already done');const before=legal(h);if(!before.includes(action))return {accepted:false,action};
 let next=h.pos;if(action.startsWith('MOVE_')){const d=DIRS.find(x=>'MOVE_'+x[0]===action);next=h.g.adj[h.pos].map(([b,e])=>({b,e})).find(({b,e})=>e!==h.gate||!h.closed? h.g.coords[b][0]-h.g.coords[h.pos][0]===d[1]&&h.g.coords[b][1]-h.g.coords[h.pos][1]===d[2]:false)?.b;
  check(next!==undefined,'legal neighbor mismatch');h.pos=next;
 }else if(action==='ERASE'){delete h.painted[h.pos]}
 else if(action==='PAINT_A'||action==='PAINT_B'){h.painted[h.pos]=action.slice(-1);
  if(h.stage===0&&h.pos===h.w.goal&&action==='PAINT_A'){h.stage=1;}
  else if(h.stage===1&&h.pos===h.w.goal2&&action==='PAINT_B'){h.stage=2;}
 }else if(action==='STOP'){h.done=true;h.success=h.stage===2&&h.painted[h.w.goal]==='A'&&h.painted[h.w.goal2]==='B'&&Object.entries(h.painted).every(([k,v])=>(+k===h.w.goal&&v==='A')||(+k===h.w.goal2&&v==='B'));}
 h.accepted++;h.events.push({kind:'ACTION_ACK',accepted:true,action,at:h.accepted,pos:h.pos});h.positions.push(h.pos);
 if(action==='PAINT_A'&&h.stage===1&&h.pos===h.w.goal)h.events.push({kind:'GOAL_REVISION',accepted:true,goal:h.w.goal2,material:'B',at:h.accepted});
 return {accepted:true,action,pos:h.pos};}
function shortest(g,s,t,blocked){const ds=dist(g,t,blocked);check(Number.isFinite(ds[s]),'disconnected target');const actions=[];let at=s;
 while(at!==t){const next=g.adj[at].filter(([,e])=>e!==blocked).find(([b])=>ds[b]===ds[at]-1);check(next,'no shortening move');actions.push(move(g,at,next[0]));at=next[0];check(actions.length<=g.coords.length,'loop')}
 return actions}
function prepare(w,kind,closed){const h=start(w,kind,closed),rng=random(seed('prelude/'+w.seed)),bad=w.index%3;
 if(bad){const target=bad===1?w.goal:w.goal2, wrong=bad===1?'PAINT_B':'PAINT_A';
  const forward=shortest(h.g,h.pos,target,-1);for(const a of forward)check(apply(h,a).accepted,'prelude move');check(apply(h,wrong).accepted,'prelude paint');
  for(const a of [...forward].reverse()){const d=DIRS.find(z=>'MOVE_'+z[0]===a);const reverse=DIRS.find(z=>z[1]===-d[1]&&z[2]===-d[2]);check(apply(h,'MOVE_'+reverse[0]).accepted,'prelude return');}
  check(h.pos===w.s&&h.stage===0,'bad prelude state');}
 const neighbor=h.g.adj[h.pos].find(([,e])=>e!==h.gate);check(neighbor,'no safe neighbor');const go=move(h.g,h.pos,neighbor[0]);
 const d=DIRS.find(z=>'MOVE_'+z[0]===go);const back='MOVE_'+DIRS.find(z=>z[1]===-d[1]&&z[2]===-d[2])[0];
 for(let j=0;j<192;j++){
  if(j===8)external(h,'EDGE_GATE',!closed);
  if(j===24)external(h,'SENSOR_DIAGNOSTIC',Math.floor(rng()*100));
  if(j===48)external(h,'EDGE_GATE',closed);
  if(j===70)external(h,'SENSOR_DIAGNOSTIC',Math.floor(rng()*100));
  check(apply(h,j%2===0?go:back).accepted,'ageing host move rejected');}
 check(h.pos===w.s&&h.closed===closed,'initial fork state');return h;
}
function input(h){const g=h.g,local=DIRS.map(([d,dx,dy])=>{const found=g.adj[h.pos].find(([b])=>g.coords[b][0]-g.coords[h.pos][0]===dx&&g.coords[b][1]-g.coords[h.pos][1]===dy);return [d,found?found[0]:null,!!found&&(found[1]!==h.gate||!h.closed)]});
 const durable=h.events.filter(e=>e.kind==='EDGE_GATE'||e.kind==='SENSOR_DIAGNOSTIC'||e.kind==='GOAL_REVISION').map(e=>e.kind==='EDGE_GATE'?[e.at,'EDGE_GATE',e.edge,e.value]:e.kind==='SENSOR_DIAGNOSTIC'?[e.at,'SENSOR_DIAGNOSTIC',e.code]:[e.at,'GOAL_REVISION',e.goal,e.material]);
 return {v:'synthetic-v44-unseen-topology',p:h.pos,stage:h.stage,goal:h.stage===0?[h.w.goal,'A']:h.stage===1?[h.w.goal2,'B']:null,legal:legal(h),geometry:{coords:g.coords,links:g.links,protected:g.protected,gate:g.links[h.gate]},local,painted:Object.entries(h.painted).map(([k,v])=>[+k,v]).sort((a,b)=>a[0]-b[0]),recent:h.events.filter(e=>e.kind==='ACTION_ACK').slice(-32).map(e=>[e.action,e.pos]),recent_positions:h.positions.slice(-32),durable};}
function ablate(inp){const z=cp(inp);z.durable=z.durable.filter(e=>e[1]!=='EDGE_GATE');return z}
function oracle(h){if(h.stage===2)return 'STOP';const target=h.stage===0?h.w.goal:h.w.goal2,mat=h.stage===0?'A':'B';
 if(h.pos===target){if(h.painted[target]!==undefined&&h.painted[target]!==mat)return 'ERASE';return 'PAINT_'+mat}
 return shortest(h.g,h.pos,target,h.closed?h.gate:-1)[0];}
function recover(h){const rows=[];for(let i=0;i<160;i++){if(h.done)break;
 const obs=input(h),action=oracle(h);check(obs.legal.includes(action),'oracle illegal');rows.push({input:obs,action,weight:1});check(apply(h,action).accepted,'host rejected teacher');}
 check(h.done&&h.success,'bad terminal '+h.w.split+':'+h.w.index);return rows}
function counterfactual(w,kind){const a=prepare(w,kind,false),b=prepare(w,kind,true),ia=input(a),ib=input(b);
 check(H(ablate(ia))===H(ablate(ib)),'counterfactual ablation leak');check(H(ia)!==H(ib),'identical full gate histories');check(H(ia.recent)===H(ib.recent)&&H(ia.local)===H(ib.local)&&H(ia.legal)===H(ib.legal),'local or recent leaks');
 const open=ranks(a.g,a.pos,w.goal,-1),closed=ranks(b.g,b.pos,w.goal,b.gate),op=open[0],cl=closed[0];
 const opWrong=open.find(x=>x.action===cl.action),clWrong=closed.find(x=>x.action===op.action);
 const positive=kind==='positive';if(positive)check(op.action!==cl.action&&opWrong&&clWrong&&opWrong.cost-op.cost>=2&&clWrong.cost-cl.cost>=2,'weak positive');
 else check(op.action===cl.action&&op.cost===cl.cost,'false negative control');
 const snapshots=[a,b],rows=snapshots.map(recover);
 return {rows,meta:{kind,positive,split:w.split,world_index:w.index,seed:w.seed,attempt:w.attempt,geometry_sha256:w.geometry_sha256,gate_edge:kind==='positive'?w.pivotal:w.control,goal:w.goal,goal2:w.goal2,ablate_sha256:H(ablate(ia)),open_input_sha256:H(ia),closed_input_sha256:H(ib),first_open:op.action,first_closed:cl.action,regret_open:opWrong?opWrong.cost-op.cost:null,regret_closed:clWrong?clWrong.cost-cl.cost:null,steps_open:rows[0].length,steps_closed:rows[1].length}};
}
function save(out,name,rows){const b=Buffer.from(rows.map(x=>JSON.stringify(x)).join('\n')+'\n');const file=path.join(out,name);fs.writeFileSync(file,b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:H(b)}}
function generate(dir,{quick=false}={}){check(!fs.existsSync(dir),'destination exists');fs.mkdirSync(dir,{recursive:true});const shards={},splits={},topOwners=new Map(),inputOwners=new Map();
 for(const [split,full] of Object.entries(SPLITS)){const count=quick?Math.min(2,full):full,policy=[],challenges=[],topologies=new Set();let terminals=0,positives=0,controls=0,zero=0;
  for(let i=0;i<count;i++){const w=scenario(split,i);check(!topOwners.has(w.geometry_sha256)||topOwners.get(w.geometry_sha256)===split,'cross-split topology collision');topOwners.set(w.geometry_sha256,split);topologies.add(w.geometry_sha256);
   for(const kind of ['positive','control']){const test=counterfactual(w,kind);(kind==='positive'?positives++:controls++);terminals+=2;challenges.push(test.meta);
    for(const trace of test.rows)for(const row of trace){const key=H(row.input),prev=inputOwners.get(key);check(!prev||prev.split===split,'cross-split input collision');check(!prev||prev.action===row.action,'conflicting label');if(prev&&split==='TRAIN'){row.weight=0;zero++;}else inputOwners.set(key,{split,action:row.action});policy.push(row);}}
  }
  shards[split+'_policy.jsonl']=save(dir,split+'_policy.jsonl',policy);shards[split+'_challenge_EVALUATOR_ONLY.jsonl']=save(dir,split+'_challenge_EVALUATOR_ONLY.jsonl',challenges);
  splits[split]={worlds:count,unique_topologies:topologies.size,positive_pairs:positives,negative_control_pairs:controls,terminal_episodes:terminals,decisions:policy.length,zero_weight:zero};
 }
 const mf={schema:'driv3r/corpus_factory/unseen-topology-v44/manifest',date:'2026-11-18',synthetic_not_native_CyberSand:true,generator:'unseen_topology_v44.js',generator_sha256:H(fs.readFileSync(__filename)),design:'Procedurally generated randomized planar graph junctions with remote edge-gate counterfactuals, long host-accepted history ageing, alternating goal/material painting and wrong-paint recovery. Strict positive reroute vs irrelevant-gate negative controls.',distribution:{TRAIN:'7-8 grid, extra-edge probability 0.40',DEV:'9 grid, extra-edge probability 0.32',HELDOUT:'10-11 grid, extra-edge probability 0.26'},model_contract:'ONLY policy.input to model; policy.action supervised label; policy.weight for loss; *_challenge_EVALUATOR_ONLY never training.',quick,splits,shards,caveats:['Synthetic graph host and event dispatcher, not native CyberSand','Oracle knows environment dynamics that policy can infer from full old event receipts; no native dynamics','No model weights trained or evaluated','Full static graph provided after synthetic survey; not unexplored frontier discovery']};
 fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
module.exports={scenario,hydrate,start,prepare,apply,legal,input,ablate,oracle,recover,counterfactual,ranks,dist,generate,H,SPLITS};
if(require.main===module){const [d,quick]=process.argv.slice(2);check(d,'usage: node unseen_topology_v44.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(d),{quick:quick==='--quick'}).splits,null,2))}
