'use strict';
// DRIV3R Corpus Factory v25 — suffix-outcome-proven branch/retrace contrasts.
// Synthetic research host, not the canonical native CyberSand interface.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:8,DEV:4,HELDOUT:4};
const GROUP_IDS={TRAIN:[0,1,2,3,4,5,6,7],DEV:[1,2,3,4],HELDOUT:[1,2,3,4]};
const DIRS=[[0,-1,'MOVE_N'],[1,0,'MOVE_E'],[0,1,'MOVE_S'],[-1,0,'MOVE_W']];
const P=[1,3,9,27,81];
const id=(x,y)=>`${x},${y}`;
function world(split,g,config){
 const seed={TRAIN:23,DEV:61,HELDOUT:107}[split]+g*7;
 const harder={TRAIN:0,DEV:2,HELDOUT:5}[split];
 const southern=2+(seed%3)+Math.floor(harder/3),western=4+(Math.floor(seed/3)%3)+harder+Math.floor(g/5),eastern=3+(Math.floor(seed/5)%3)+harder+g%3,northern=3+(Math.floor(seed/7)%2)+Math.floor(harder/2);
 const cells=new Set(),edges=new Map();
 const add=(x,y)=>{const k=id(x,y);cells.add(k);if(!edges.has(k))edges.set(k,new Set());return k;};
 const join=(a,b)=>{add(...a.split(',').map(Number));add(...b.split(',').map(Number));edges.get(a).add(b);edges.get(b).add(a);};
 const chain=arr=>{for(let i=0;i<arr.length;i++){add(...arr[i]);if(i)join(id(...arr[i-1]),id(...arr[i]));}};
 chain([[0,0],[-1,0],[-2,0]]);
 chain([[0,0],[0,-1],[0,-2]]);
 if(!config.blocked){const chainNorth=[[0,-2]];for(let y=-3;y>=-northern;y--)chainNorth.push([0,y]);chain(chainNorth);}
 chain([[-1,0],[-1,-1],[-1,-2]]);
 if(config.upper_link)join(id(0,-2),id(-1,-2));
 const south=[[-1,0]];for(let y=1;y<=southern;y++)south.push([-1,y]);south.push([0,southern],[1,southern]);for(let y=southern-1;y>=0;y--)south.push([1,y]);south.push([0,0]);chain(south);
 const west=[[-2,0]];for(let x=-3;x>=-western;x--)west.push([x,0]);chain(west);
 const east=[[1,0]];for(let x=2;x<=eastern;x++)east.push([x,0]);chain(east);
 const choice=[id(0,-2),id(-1,-2),id(-western,0),id(eastern,0),id(0,southern)];
 const target=choice.filter((_,i)=>(config.target_mask&(1<<i))!==0);
 const recipes=target.map((_,i)=>((i+config.phase)%2===0?'A':'B'));
 for(const p of target)if(!cells.has(p))throw Error('invalid target '+p);
 const nodes=[...cells].sort(),nidx=new Map(nodes.map((n,i)=>[n,i]));
 const adjacency=nodes.map(n=>[...edges.get(n)].sort().map(x=>nidx.get(x)));
 const material=Object.fromEntries(target.map((t,i)=>[t,recipes[i]]));
 return {split,group:g,shape:{southern,western,eastern,northern},nodes,adjacency,nidx,root:nidx.get('0,0'),previous:nidx.get('-1,0'),exit:nidx.get('-2,0'),target,recipes,material,quota:2,config,graph_sha256:sha({nodes,adjacency}),remote_sha256:sha({nodes,adjacency,target,recipes})};
}
function getDigit(status,i){return Math.floor(status/P[i])%3;}
function putDigit(status,i,n){return status+(n-getDigit(status,i))*P[i];}
function correctCount(w,status){let count=0,wrong=0;for(let j=0;j<w.target.length;j++){const d=getDigit(status,j),want=w.recipes[j]==='A'?1:2;if(d===want)count++;else if(d!==0)wrong++;}return {count,wrong};}
function terminal(w,p,status){const q=correctCount(w,status);return p===w.exit&&q.count>=w.quota&&q.wrong===0;}
function key(p,prev,status){return `${p}|${prev}|${status}`;}
function options(w,s){const [p,prev,status]=s,acts=[];
 for(const n of w.adjacency[p]){
  let action;if(n===prev)action='BACKTRACK';else {const [x,y]=w.nodes[p].split(',').map(Number),[a,b]=w.nodes[n].split(',').map(Number);action=DIRS.find(d=>d[0]===a-x&&d[1]===b-y)?.[2];}
  if(!action)throw Error('non-orthogonal edge');acts.push({action,next:[n,p,status]});
 }
 const targetIndex=w.target.indexOf(w.nodes[p]);
 if(targetIndex>=0){const v=getDigit(status,targetIndex);if(v===0)for(let m=1;m<=2;m++)acts.push({action:m===1?'PAINT_A':'PAINT_B',next:[p,prev,putDigit(status,targetIndex,m)]});else acts.push({action:'ERASE',next:[p,prev,putDigit(status,targetIndex,0)]});}
 if(terminal(w,p,status))acts.push({action:'STOP',next:null});
 return acts.sort((a,b)=>a.action.localeCompare(b.action));
}
// Exhaustive suffix distances for every node, predecessor and paint state.
// This derives action targets from terminal outcomes, never a geometric proxy.
function planner(w){
 const reverse=new Map(),distance=new Map(),queue=[];
 const addPred=(from,to)=>{const k=key(...to);if(!reverse.has(k))reverse.set(k,[]);reverse.get(k).push(key(...from));};
 for(let p=0;p<w.nodes.length;p++)for(const prev of w.adjacency[p])for(let status=0;status<81;status++){
  const s=[p,prev,status],k=key(...s);
  if(terminal(w,p,status)){distance.set(k,1);queue.push(k);}
  for(const a of options(w,s))if(a.next)addPred(s,a.next);
 }
 for(let i=0;i<queue.length;i++){
  const k=queue[i],d=distance.get(k);
  for(const pred of reverse.get(k)||[])if(!distance.has(pred)){distance.set(pred,d+1);queue.push(pred);}
 }
 function rank(s){return options(w,s).map(a=>({action:a.action,cost:a.next?1+(distance.get(key(...a.next))??Infinity):1})).sort((a,b)=>a.cost-b.cost||a.action.localeCompare(b.action));}
 return {rank,reachable:distance.size,total_states:reverse.size};
}
function candidates(split,group){
 const pool=[];for(let blocked of [false,true])for(let upper_link of [false,true])for(let target_mask=1;target_mask<32;target_mask++)for(let phase of [0,1]){
  if(target_mask.toString(2).replace(/0/g,'').length!==3)continue;
  const config={blocked,upper_link,target_mask,phase},w=world(split,group,config),p=planner(w),ranking=p.rank([w.root,w.previous,0]);
  if(!Number.isFinite(ranking[0].cost)||ranking[0].cost===ranking[1].cost)continue;
  if(!['BACKTRACK','MOVE_N'].includes(ranking[0].action))continue;
  if(ranking[0].cost>80)continue;
  const other=ranking.find(a=>a.action===(ranking[0].action==='MOVE_N'?'BACKTRACK':'MOVE_N'));
  if(!other||!Number.isFinite(other.cost)||other.cost-ranking[0].cost<2)continue;
  pool.push({w,ranking,p,config,first:ranking[0].action,regret:other.cost-ranking[0].cost,optimal_cost:ranking[0].cost,alternative_cost:other.cost});
 }
 return pool;
}
function family(split,group){
 const pool=candidates(split,group),out=[];
 // Each family crosses blocked/open distal geometry with move/retrace outcomes.
 // Prefer different omit indices to rule out a single visible local cue.
 for(const blocked of [false,true])for(const action of ['BACKTRACK','MOVE_N']){
  const subset=pool.filter(x=>x.config.blocked===blocked&&x.first===action);
  subset.sort((a,b)=>b.regret-a.regret||a.config.target_mask-b.config.target_mask||Number(a.config.upper_link)-Number(b.config.upper_link)||a.config.phase-b.config.phase);
  if(subset.length<2)continue;out.push(subset[0]);out.push(subset.find(x=>x.config.target_mask!==subset[0].config.target_mask)||subset[1]);
 }
 if(out.length!==8)throw Error(`insufficient counterfactual examples ${split}:${group} pool=${pool.length} selected=${out.map(x=>x.first)}`);
 const base=observe(new Host(out[0].w,split,group,0));
 for(let i=0;i<out.length;i++){
  const c=out[i],o=observe(new Host(c.w,split,group,i));
  if(JSON.stringify(o.local_view)!==JSON.stringify(base.local_view)||JSON.stringify(o.legal_actions)!==JSON.stringify(base.legal_actions)||JSON.stringify(o.history.events.slice(-64))!==JSON.stringify(base.history.events.slice(-64)))throw Error('alias mismatch');
 }
 return out;
}
function prefix(w){
 const events=[{kind:'REMOTE_SURVEY',visibility:'remote',tiles:w.nodes.map((n,i)=>({cell:n,blocked:false,neighbors:w.adjacency[i].map(j=>w.nodes[j])})),paint_targets:w.target.map((n,i)=>({cell:n,material:w.recipes[i]})),goal:{paint_quota:w.quota,exit:w.nodes[w.exit]}}];
 const positions=[w.nodes[w.exit]];for(let i=0;i<96;i++){const at=i%2===0?w.nodes[w.previous]:w.nodes[w.exit];events.push({kind:'MOVE_ACK',accepted:true,at});positions.push(at);}
 events.push({kind:'MOVE_ACK',accepted:true,at:w.nodes[w.root]});positions.push(w.nodes[w.root]);return {events,positions};
}
class Host{
 constructor(w,split,group,variant){this.w=w;this.pos=w.root;this.prev=w.previous;this.status=0;this.history=prefix(w);this.remaining=100;this.done=false;this.invalid=false;this.episode_id=`${split}-fork-${String(group).padStart(2,'0')}-${variant}`;}
}
function observe(h){
 const w=h.w,p=h.pos,xy=w.nodes[p].split(',').map(Number),local=[];
 for(const [dx,dy] of [[0,0],...DIRS.map(d=>d.slice(0,2))]){
  const n=id(xy[0]+dx,xy[1]+dy),j=w.nidx.get(n),reachable=j!==undefined&&w.adjacency[p].includes(j);
  const t=w.target.indexOf(n);local.push({offset:[dx,dy],reachable,paintable:t>=0&&reachable||t>=0&&dx===0&&dy===0,painted:t>=0&&reachable||t>=0&&dx===0&&dy===0?['EMPTY','A','B'][getDigit(h.status,t)]:null});
 }
 const state=[h.pos,h.prev,h.status];return {protocol:'driv3r.synthetic-outcome-fork/25',position:w.nodes[p],local_view:local,legal_actions:options(w,state).map(x=>x.action),goal:{quota:w.quota,exit:w.nodes[w.exit]},painted_correct:correctCount(w,h.status).count,remaining:h.remaining,history:clone(h.history)};
}
function apply(h,action){
 if(h.done)throw Error('already terminal');if(h.remaining<=0){h.invalid=true;throw Error('budget exhausted');}
 const a=options(h.w,[h.pos,h.prev,h.status]).find(x=>x.action===action);
 if(!a){h.history.events.push({kind:'REFUSED',action,accepted:false});h.remaining--;return {action,accepted:false};}
 if(a.next){[h.pos,h.prev,h.status]=a.next;}else h.done=true;
 h.remaining--;const receipt={kind:'ACTION_ACK',accepted:true,action,at:h.w.nodes[h.pos],painted_correct:correctCount(h.w,h.status).count};h.history.events.push(receipt);h.history.positions.push(h.w.nodes[h.pos]);return receipt;
}
function replay(c,split,group,variant,emit){
 const h=new Host(c.w,split,group,variant),header={schema:'driv3r/corpus_factory/outcome-forks-v25/episode',episode_id:h.episode_id,split,world_group:`${split}-${group}`,synthetic_host:true,geometry_sha256:c.w.graph_sha256,remote_world_sha256:c.w.remote_sha256,difficulty:{nodes:c.w.nodes.length,optimal_suffix_steps:c.optimal_cost,counterfactual_regret:c.regret},initial_history:clone(h.history),observation_keys:Object.keys(observe(h))};
 let lastE=h.history.events.length,lastP=h.history.positions.length,step=0,paints=0,retraces=0;
 while(!h.done){const input=observe(h),rank=c.p.rank([h.pos,h.prev,h.status]),target=rank[0].action;if(rank[0].cost===Infinity)throw Error('unreachable');
  const events_added=clone(input.history.events.slice(lastE)),positions_added=input.history.positions.slice(lastP);lastE=input.history.events.length;lastP=input.history.positions.length;
  const receipt=apply(h,target);if(!receipt.accepted)throw Error('refused oracle');
  emit({schema:'driv3r/corpus_factory/outcome-forks-v25/decision',episode_id:h.episode_id,step,state:Object.fromEntries(Object.entries(input).filter(([k])=>k!=='history')),events_added,positions_added,model_input_sha256:sha(input),target,receipt,training_weight:1});
  if(target.startsWith('PAINT_'))paints++;if(target==='BACKTRACK')retraces++;step++;if(step>100)throw Error('timeout');
 }
 if(step!==c.optimal_cost)throw Error('optimal suffix mismatch '+step+' != '+c.optimal_cost);
 return {...header,steps:step,paints,retraces,terminal_success:!h.invalid,first_target:c.first,alternative_cost:c.alternative_cost,optimal_cost:c.optimal_cost,counterfactual_regret:c.regret,blocked_distal_spur:c.config.blocked};
}
function fileMeta(file,rows){const b=fs.readFileSync(file);return {sha256:sha(b),bytes:b.length,rows};}
function generate(dest,{quick=false}={}){
 if(fs.existsSync(dest))throw Error('refusing overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},seenAll=new Map(),geomOwners=new Map(),sigOwners=new Map();
 for(const [split,n] of Object.entries(SPLITS)){
  const eps=[],ds=[],seen=new Map(),classCounts={BACKTRACK:0,MOVE_N:0},blockedCounts={BACKTRACK:0,MOVE_N:0},openCounts={BACKTRACK:0,MOVE_N:0},groups=quick?1:n;
  for(let g=0;g<groups;g++){
   const actualGroup=GROUP_IDS[split][g],fam=family(split,actualGroup);for(let v=0;v<fam.length;v++){
    const c=fam[v],geometry=c.w.graph_sha256,owner=geomOwners.get(geometry);if(owner&&owner!==split)throw Error('split geometry overlap');geomOwners.set(geometry,split);
    const signature=c.w.remote_sha256,sigOwner=sigOwners.get(signature);if(sigOwner&&sigOwner!==split)throw Error('split world overlap');sigOwners.set(signature,split);
    classCounts[c.first]++;(c.config.blocked?blockedCounts:openCounts)[c.first]++;
    const header=replay(c,split,actualGroup,v,d=>{
     const old=seen.get(d.model_input_sha256);if(old!==undefined){if(old!==d.target)throw Error('label conflict');if(split==='TRAIN')d.training_weight=0;}else seen.set(d.model_input_sha256,d.target);
     const prior=seenAll.get(d.model_input_sha256);if(prior&&prior!==split)throw Error('cross-split exact input leakage');seenAll.set(d.model_input_sha256,split);ds.push(d);
    });eps.push(header);
   }
  }
  if(classCounts.BACKTRACK!==classCounts.MOVE_N||blockedCounts.BACKTRACK!==blockedCounts.MOVE_N||openCounts.BACKTRACK!==openCounts.MOVE_N)throw Error('unbalanced family');
  const ef=split+'_episodes.jsonl',df=split+'_decisions.jsonl';fs.writeFileSync(path.join(dest,ef),eps.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(dest,df),ds.map(x=>JSON.stringify(x)).join('\n')+'\n');shards[ef]=fileMeta(path.join(dest,ef),eps.length);shards[df]=fileMeta(path.join(dest,df),ds.length);
  splits[split]={groups,group_ids:GROUP_IDS[split].slice(0,groups),episodes:eps.length,decisions:ds.length,unique_inputs:seen.size,duplicate_inputs:ds.length-seen.size,first_action:classCounts,blocked_distal_spur:blockedCounts,open_distal_spur:openCounts,paint_actions:eps.reduce((x,e)=>x+e.paints,0),retrace_actions:eps.reduce((x,e)=>x+e.retraces,0),min_counterfactual_regret:Math.min(...eps.map(e=>e.counterfactual_regret))};
 }
 const m={schema:'driv3r/corpus_factory/outcome-forks-v25/manifest',host:'synthetic-not-native-CyberSand',generator:'outcome_forks_v25.js',generator_sha256:sha(fs.readFileSync(__filename)),quick,oracle:'exhaustive reverse-BFS over pos,previous,paint-status with terminal quota/exit and contamination recovery',oracle_actions_entire_post_prelude:true,initial_preparatory_history_scripted:true,distal_obstruction_not_proxy_label:true,suffix_counterfactual_outcomes:true,full_last64_alias_within_family:true,split_worlds_disjoint:true,splits,shards};fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(m,null,2)+'\n');return m;
}
module.exports={SPLITS,sha,world,planner,candidates,family,Host,observe,apply,replay,generate,correctCount,key};
if(require.main===module){if(!process.argv[2])throw Error('usage: node outcome_forks_v25.js NEW_DIRECTORY [--quick]');const m=generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'});console.log(JSON.stringify(m.splits));}
