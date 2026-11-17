'use strict';
// DRIV3R Corpus Factory v32: causal re-opened gate and four-way memory on five-target
// synthetic graphs. Not native CyberSand. No learner, no policy-training code.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const hash=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:[0,1,2,3,4,5,6,7],DEV:[0,1,2,3],HELDOUT:[0,1,2,3]};
const DIR=[[0,-1],[1,0],[0,1],[-1,0]];
function world(split,g,v){
  const L=({TRAIN:4,DEV:14,HELDOUT:21}[split]+g),N=L+4+(g%3),kind=(v>>2)&1;
  const id=(x,y)=>`${x},${y}`,edges=new Map();
  const add=(x,y)=>{const key=id(x,y);if(!edges.has(key))edges.set(key,new Set());return key;};
  const join=(a,b)=>{const [x,y]=a.split(',').map(Number),[u,z]=b.split(',').map(Number);
    if(Math.abs(x-u)+Math.abs(y-z)!==1)throw Error('nonorthogonal');edges.get(a).add(b);edges.get(b).add(a);};
  add(0,0);for(const [dx,dy,n] of [[-1,0,L],[1,0,L],[0,-1,N],[0,1,L]]){
    let prev=id(0,0);for(let k=1;k<=n;k++){const next=add(k*dx,k*dy);join(prev,next);prev=next;}
  }
  // A dynamic shortcut changes after the first accepted painting operation.
  add(1,1);join(id(1,1),id(1,0));join(id(1,1),id(0,1));
  // Additional disjoint spurs to discourage a single memorized graph template.
  for(const [x,y,dy] of kind?[[2,0,1],[-2,0,-1]]:[[0,2,1],[0,-2,-1]]){
    const k=id(x,y+dy);if(edges.has(k))continue;add(x,y+dy);join(id(x,y),k);
  }
  const nodes=[...edges.keys()].sort(),nidx=new Map(nodes.map((k,i)=>[k,i]));
  const adjacency=nodes.map(k=>[...edges.get(k)].sort().map(n=>nidx.get(n)));
  const target=[id(-L,0),id(L,0),id(0,L),id(0,-L),id(0,-N)];
  const recipes=target.map((_,i)=>((Boolean((i+(v&1)+g)%2) !== (Boolean(v&2)&&i%2===0))?'B':'A'));
  // Gate is initially CLOSED, and the first accepted paint OPENS it. The
  // 3-edge bypass through (0,1)->(1,1)->(1,0) keeps the graph traversable.
  // Unlike v31's equivalent-length cross-arm link, this gate changes the
  // exact cost of reaching the east arm and can alter optimal decisions.
  const rule={kind:'FIRST_ACCEPTED_PAINT_TOGGLES_GATE',gate_edge:[id(0,0),id(1,0)],initial_open:false};
  const w={nodes,nidx,adjacency,target,recipes,quota:2,root:nidx.get(id(0,0)),previous:nidx.get(id(0,-1)),
    exit:nidx.get(id(0,-N)),gate:rule.gate_edge.map(k=>nidx.get(k)),gate_initial_open:false,rule,
    protected_cells:['-1,1','-1,-1','1,-1','2,2'],
    geometry_sha256:hash({nodes,adjacency}),world_sha256:hash({nodes,adjacency,target,recipes,rule,quota:2})};
  if(w.target.some(t=>!w.nidx.has(t))||w.gate.some(x=>x===undefined))throw Error('broken world');
  return w;
}
function survey(w){return {kind:'REMOTE_SURVEY',tiles:w.nodes.map((cell,i)=>({cell,neighbors:w.adjacency[i].map(j=>w.nodes[j])})),
    paint_targets:w.target.map((cell,i)=>({cell,material:w.recipes[i]})),protected_cells:w.protected_cells,
    goal:{quota:w.quota,exit:w.nodes[w.exit]},live_gate:w.rule};}
function digit(s,i){return Math.floor(s/3**i)%3;}
function statusWrite(s,i,value){return s+(value-digit(s,i))*3**i;}
function terminal(w,s,p){if(p!==w.exit)return false;let good=0;
  for(let i=0;i<5;i++){const d=digit(s,i);if(d && d!==(w.recipes[i]==='A'?1:2))return false;
    if(d)good++;}return good>=w.quota;}
function oracle(w,{fixed}={}){
  // Reverse BFS over (position, five ternary paint statuses). The latch is
  // permanently 1 after both host-accepted prelude paintings; previous node
  // changes action spelling but not shortest path cost.
  const n=w.nodes.length,states=243,limit=n*states,dist=new Int32Array(limit).fill(-1),q=new Int32Array(limit);
  let tail=0,head=0,terminalCount=0;
  for(let s=0;s<states;s++)if(terminal(w,s,w.exit)){const key=s*n+w.exit;dist[key]=1;q[tail++]=key;terminalCount++;}
  if(!terminalCount)throw Error('no terminal states');
  const targetAt=new Map(w.target.map((k,i)=>[w.nidx.get(k),i]));
  function enqueue(s,p,d){const key=s*n+p;if(dist[key]>=0)return;dist[key]=d;q[tail++]=key;}
  while(head<tail){const key=q[head++],s=Math.floor(key/n),p=key%n,d=dist[key]+1;
    for(const prev of w.adjacency[p])if(host.edgeAllowed(w,p,prev,1,fixed))enqueue(s,prev,d);
    const i=targetAt.get(p);if(i!==undefined){const old=digit(s,i);
      if(old===0){enqueue(statusWrite(s,i,1),p,d);enqueue(statusWrite(s,i,2),p,d);}
      else enqueue(statusWrite(s,i,0),p,d);
    }
  }
  const rank=state=>host.options(w,state,fixed).map(a=>({action:a.action,cost:a.next?
    (dist[a.next[2]*n+a.next[0]]<0?Infinity:dist[a.next[2]*n+a.next[0]]+1):1}))
    .sort((a,b)=>a.cost-b.cost||a.action.localeCompare(b.action));
  return {rank,visited:tail,terminalCount,states:limit};
}
function go(h,to){
  // BFS on the actual host-accessible graph, choosing only accepted legal
  // movement operations. No teleportation or synthetic state mutations.
  const w=h.w,queue=[h.pos],pred=new Map([[h.pos,-1]]);
  for(let i=0;i<queue.length&&!pred.has(to);i++)for(const n of w.adjacency[queue[i]]){
    if(!pred.has(n)&&host.edgeAllowed(w,queue[i],n,h.latched)){pred.set(n,queue[i]);queue.push(n);}}
  if(!pred.has(to))throw Error('route blocked');
  const route=[];for(let p=to;p!==h.pos;p=pred.get(p))route.push(p);route.reverse();
  for(const n of route){const a=host.options(w,[h.pos,h.prev,h.status,h.latched]).find(x=>x.next?.[0]===n);
    if(!a||!host.apply(h,a.action).accepted)throw Error('host refused scripted movement');}
  return route.length;
}
function prepare(w,split,g,v,arm){
  const h=new host.Host(w,split,g,v);h.remaining=1200;h.history={events:[survey(w)],positions:[w.nodes[h.pos]]};
  // Four possible older wrong-material locations, one identical latest wrong
  // PAINT on the common goal. Equal route lengths equalize all public counters
  // and the absolute index of the most recent paint receipt.
  const wrong=i=>w.recipes[i]==='A'?'PAINT_B':'PAINT_A';let accepted=0;
  accepted+=go(h,w.nidx.get(w.target[arm]));
  if(arm!==1){ // Equalize the two extra legal steps used by the closed east gate.
    const targetIdx=h.pos;const predecessor=h.prev;
    accepted+=go(h,predecessor);accepted+=go(h,targetIdx);
  }
  if(!host.apply(h,wrong(arm)).accepted)throw Error('wrong paint rejected');accepted++;
  accepted+=go(h,w.root);accepted+=go(h,w.nidx.get(w.target[4]));
  if(!host.apply(h,wrong(4)).accepted)throw Error('shared paint rejected');accepted++;
  accepted+=go(h,w.root);
  const north=w.nidx.get('0,-1');for(let i=0;i<80;i++){accepted+=go(h,north);accepted+=go(h,w.root);}
  if(host.correct(w,h.status).wrong!==2||h.latched!==1||h.pos!==w.root||h.prev!==north)throw Error('wrong scripted state');
  const obs=host.observe(h),recent=obs.history.events.slice(-64);
  if(recent.some(e=>e.action?.startsWith('PAINT')||e.kind==='GATE_TRANSITION'))throw Error('history not aged');
  return {h,obs,accepted};
}
function restricted(obs,kind){return view.compress(obs,kind==='last-paint'?{lastPaintOnly:true}:{memory:false});}
function qualify(w,p,split,g,v){
  const cases=Array.from({length:4},(_,arm)=>prepare(w,split,g,v,arm));
  const first=cases.map(c=>p.rank([c.h.pos,c.h.prev,c.h.status,c.h.latched])[0]);
  const expected=['MOVE_W','MOVE_E','MOVE_S','BACKTRACK'];
  if(first.some((r,i)=>!r||!Number.isFinite(r.cost)||r.action!==expected[i]))return null;
  const hs=cases.map(c=>hash(restricted(c.obs,'none'))),ls=cases.map(c=>hash(restricted(c.obs,'last-paint'))),
    rs=cases.map(c=>hash(c.obs.history.events.slice(-64))),full=cases.map(c=>hash(c.obs));
  if(new Set(hs).size!==1||new Set(ls).size!==1||new Set(rs).size!==1||new Set(full).size!==4)return null;
  if(new Set(cases.map(c=>c.accepted)).size!==1)return null;
  const regrets=cases.map((c,i)=>{const ranking=p.rank([c.h.pos,c.h.prev,c.h.status,c.h.latched]);
    return Math.min(...expected.filter((_,j)=>j!==i).map(a=>(ranking.find(x=>x.action===a)?.cost??Infinity)-first[i].cost));});
  if(regrets.some(x=>!Number.isFinite(x)||x<2))return null;
  // A frozen-closed counterfactual has exactly the same graph, recipe and
  // paint statuses, but wrongly suppresses the actual accepted gate opening.
  // Prove at least one *strict* downstream decision divergence, not simply a
  // different graph description or a nonconsequential gate receipt.
  const frozen=oracle(w,{fixed:false}),causal=[];
  for(let arm=0;arm<4;arm++){
    const c=cases[arm],dynamic=p.rank([c.h.pos,c.h.prev,c.h.status,c.h.latched]),old=frozen.rank([c.h.pos,c.h.prev,c.h.status,c.h.latched]);
    const top=old[0],alternative=dynamic.find(x=>x.action===top.action);
    const regret=alternative?alternative.cost-dynamic[0].cost:Infinity;
    causal.push({arm,dynamic_first:dynamic[0].action,frozen_first:top.action,
      frozen_first_legal_now:Boolean(alternative),strict_regret:regret});
  }
  if(!causal.some(x=>x.strict_regret>=2))return null;
  return {cases,first,regrets,causal,restricted_hash:hs[0],last_paint_hash:ls[0],recent_hash:rs[0]};
}
function replay(w,p,split,g,v,pair,arm,emit){
  const c=pair.cases[arm],h=c.h,id=`${split}-v32-g${g}-v${v}-arm${arm}`;h.id=id;
  const header={schema:'driv3r/corpus_factory/causal-shortcut-v32/episode',episode_id:id,split,
    world_group:`${split}-${g}`,pair_id:`${split}-g${g}-v${v}`,synthetic_host_not_native:true,
    world_sha256:w.world_sha256,geometry_sha256:w.geometry_sha256,
    difficulty:{nodes:w.nodes.length,target_count:5,older_remote_distance:Math.abs(Number(w.target[arm].split(',')[0]))+Math.abs(Number(w.target[arm].split(',')[1])),
      scripted_accepted_actions:c.accepted,post_paint_accepted_moves:160,first_wrong_choice_regret:pair.regrets[arm]},
    hidden_interventions:{older_wrong_target_index:arm,latest_wrong_target_index:4},
    gate_causal_control:pair.causal[arm],
    initial_history:clone(h.history),initial_full_observation_sha256:hash(c.obs),
    no_memory_alias_sha256:pair.restricted_hash,last_paint_alias_sha256:pair.last_paint_hash};
  let e=h.history.events.length,z=h.history.positions.length,steps=0,erases=0;
  while(!h.done){const o=host.observe(h),rank=p.rank([h.pos,h.prev,h.status,h.latched])[0];
    if(!rank||!Number.isFinite(rank.cost)||steps>200)throw Error('oracle suffix unavailable');
    const bounded=view.compress(o);if(JSON.stringify(bounded).length>8192)throw Error('bounded public view >8192 JSON characters');
    const record={schema:'driv3r/corpus_factory/causal-shortcut-v32/decision',episode_id:id,step:steps,
      state:Object.fromEntries(Object.entries(o).filter(([k])=>k!=='history')),
      events_added:clone(o.history.events.slice(e)),positions_added:o.history.positions.slice(z),
      model_input_sha256:hash(o),bounded_model_input:bounded,bounded_model_input_sha256:hash(bounded),
      target:rank.action,training_weight:1};
    e=o.history.events.length;z=o.history.positions.length;
    const receipt=host.apply(h,rank.action);if(!receipt.accepted)throw Error('oracle refused');
    record.receipt=receipt;emit(record,o);if(rank.action==='ERASE')erases++;steps++;
  }
  if(erases!==2||host.correct(w,h.status).wrong||host.correct(w,h.status).count<2||steps!==pair.first[arm].cost)
    throw Error('terminal recovery not shortest');
  return {...header,first_action:pair.first[arm].action,first_action_regret:pair.regrets[arm],terminal_success:true,steps,erases};
}
function shard(out,name,rows){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(out,name),b);return {rows:rows.length,bytes:b.length,sha256:hash(b)};}
function generate(dest,{quick=false}={}){
  if(fs.existsSync(dest))throw Error('refusing overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
  const splits={},shards={},inputOwners=new Map(),geometryOwners=new Map();
  for(const [split,gAll] of Object.entries(SPLITS)){
    const gIds=quick?gAll.slice(0,1):gAll,eps=[],records=[],pairs=[],rejected=[],seen=new Map();
    for(const g of gIds)for(let v=0;v<8;v++){
      const w=world(split,g,v),p=oracle(w),pair=qualify(w,p,split,g,v);
      if(!pair){rejected.push({g,v});continue;}
      const prev=geometryOwners.get(w.geometry_sha256);
      if(prev&&prev!==split)throw Error('cross-split graph geometry leakage');geometryOwners.set(w.geometry_sha256,split);
      pairs.push({pair_id:`${split}-g${g}-v${v}`,first_actions:pair.first.map(x=>x.action),regrets:pair.regrets,
        frozen_closed_strict_regret:pair.causal.map(x=>x.strict_regret),world_sha256:w.world_sha256});
      for(let arm=0;arm<4;arm++)eps.push(replay(w,p,split,g,v,pair,arm,r=>{
        const prior=seen.get(r.model_input_sha256),owner=inputOwners.get(r.model_input_sha256);
        if(prior&&prior!==r.target)throw Error('same input, conflicting oracle label');
        if(owner&&owner!==split)throw Error('cross-split exact input overlap');
        if(prior&&split==='TRAIN')r.training_weight=0;else seen.set(r.model_input_sha256,r.target);
        inputOwners.set(r.model_input_sha256,split);records.push(r);
      }));
    }
    if(pairs.length===0)throw Error(`no strict four-way worlds: ${split}`);
    for(const suffix of ['episodes','decisions']){const name=`${split}_${suffix}.jsonl`,rows=suffix==='episodes'?eps:records;
      shards[name]=shard(dest,name,rows);}
    splits[split]={groups:gIds.length,group_ids:gIds,worlds_considered:gIds.length*8,strict_four_way_worlds:pairs.length,rejected,
      episodes:eps.length,decisions:records.length,corrective_erasures:eps.reduce((n,e)=>n+e.erases,0),
      first_action:eps.reduce((d,e)=>(d[e.first_action]=(d[e.first_action]||0)+1,d),{}),
      min_wrong_first_regret:Math.min(...eps.map(x=>x.first_action_regret)),
      strict_frozen_gate_first_disagreements:pairs.reduce((n,p)=>n+p.frozen_closed_strict_regret.filter(r=>r>=2).length,0),
      exact_duplicates:records.length-seen.size,
      zero_weight:records.filter(r=>r.training_weight===0).length,unique_full_inputs:seen.size};
  }
  const manifest={schema:'driv3r/corpus_factory/causal-shortcut-v32/manifest',synthetic_host_not_native_CyberSand:true,
    generator:'causal_shortcut_v32.js',generator_sha256:hash(fs.readFileSync(__filename)),
    dependencies:['causal_gate_v26.js','outcome_forks_v25.js','bounded_view_v30.js'].map(f=>({file:f,sha256:hash(fs.readFileSync(path.join(__dirname,f)))})),
    quick,oracle:'exact reverse BFS of five-ternary-target status x node position, post-latch, STOP included',
    challenge:'Four remote wrong-material locations require four different corrections despite identical public recent64; initial closed root-east gate opens on first accepted paint and produces measured real alternate-route regret against frozen-closed control.',
    restricted_first_action_theoretical_ceiling:0.25,last_paint_only_first_action_theoretical_ceiling:0.25,
    consumer_warning:'Use reconstructed observation or bounded_model_input only; episode metadata/hidden_interventions/split/pair_id/oracle labels are not policy inputs. JSON length cap is characters, not tokenizer tokens.',
    limitations:['Synthetic graph and scripted host-accepted prelude, no native CyberSand acceptance.','Causal gate effect is evaluated after a scripted paint; not proof of adaptive learned action.','Only five paintable targets, a single gate and simple radial corridors.','No real SLM training or latency/RSS measured.'],splits,shards};
  fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={world,oracle,prepare,qualify,replay,generate,hash,restricted};
if(require.main===module){if(!process.argv[2])throw Error('usage: node causal_shortcut_v32.js NEW_DIRECTORY [--quick]');
  console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv[3]==='--quick'}).splits));}
