'use strict';
/* Independent corpus v46 audit: no call to v46.teacher or v46.replay for labels;
 * regenerates host state and confirms every policy action in sequence. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const ok=(v,m)=>{if(!v)throw Error(m)};
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const k=x=>x.join(',');const edge=(a,b)=>[k(a),k(b)].sort().join('|');
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
function independentAction(i){const receipts=i.durable.filter(x=>x[1]==='MISSION_ASSIGN'||x[1]==='GOAL_REVISION');
 ok(receipts.length>0,'no mission history');if(i.durable.some(x=>x[1]==='MISSION_COMPLETE'))return 'STOP';
 const [,,goal,mat]=receipts.at(-1),source=k(i.p),dest=k(goal);
 if(source===dest)return i.painted.some(([p])=>k(p)===dest)?'ERASE':'PAINT_'+mat;
 const vertices=new Map(i.survey.nodes.map(p=>[k(p),p])),adj=new Map([...vertices.keys()].map(k=>[k,[]]));
 const relevantGate=i.durable.filter(x=>x[1]==='EDGE_GATE').at(-1),blocked=relevantGate&&relevantGate[4]===1?edge(relevantGate[2],relevantGate[3]):null;
 for(const [a,b] of i.survey.edges){if(edge(a,b)===blocked)continue;adj.get(k(a)).push(k(b));adj.get(k(b)).push(k(a))}
 const q=[source],cost=new Map([[source,0]]),from=new Map();for(let z=0;z<q.length;z++)for(const b of adj.get(q[z])||[])if(!cost.has(b)){cost.set(b,cost.get(q[z])+1);from.set(b,q[z]);q.push(b)}
 let target=dest;if(!cost.has(dest)){
  const visited=new Set(i.survey.visited.map(k));const choices=q.filter(x=>!visited.has(x));ok(choices.length,'no surveyed frontier');
  choices.sort((a,b)=>{const score=x=>{const p=vertices.get(x);return 2*(Math.abs(p[0]-goal[0])+Math.abs(p[1]-goal[1]))+cost.get(x)};
   return score(a)-score(b)||a.localeCompare(b)});target=choices[0];}
 let next=target;while(from.get(next)!==source){next=from.get(next);ok(next,'unreachable surveyed path')}
 const a=vertices.get(next),p=i.p;return a[0]>p[0]?'MOVE_E':a[0]<p[0]?'MOVE_W':a[1]>p[1]?'MOVE_S':'MOVE_N';}
function checkObservation(i,h){ok(i.v==='synthetic-v46-historical-missions','wrong input tag');
 ok(!('target' in i)&&!('stage' in i)&&!('goal' in i)&&!('source_seed' in i),'privileged live mission leak');
 const seen=new Set(h.positions),nodes=new Set(),edges=new Set();for(const z of seen){const a=h.g.coords[z];nodes.add(k(a));for(const [n] of h.g.adj[z]){nodes.add(k(h.g.coords[n]));edges.add(edge(a,h.g.coords[n]))}}
 ok(H([...nodes].sort())===H(i.survey.nodes.map(k).sort()),'unseen node leak');
 ok(H([...edges].sort())===H(i.survey.edges.map(([a,b])=>edge(a,b)).sort()),'unseen edge leak');
 ok(H([...seen].map(z=>k(h.g.coords[z])).sort())===H(i.survey.visited.map(k).sort()),'bad visit receipt');
 ok(i.legal.length===new Set(i.legal).size&&i.legal.every(a=>host.legal(h).includes(a)),'bad legal menu');}
function audit(dir,{quick=false}={}){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'))),files={};
 ok(mf.quick===quick&&mf.synthetic_not_native_CyberSand===true,'provenance flags');
 ok(H(fs.readFileSync(path.join(__dirname,'mission_memory_v46.js')))===mf.generator_sha256,'wrong generator pinned hash');
 for(const p of mf.parents)ok(H(fs.readFileSync(path.join(__dirname,p.file)))===p.sha256,'parent hash mismatch '+p.file);
 for(const [name,s] of Object.entries(mf.shards)){const b=fs.readFileSync(path.join(dir,name));ok(H(b)===s.sha256&&s.bytes===b.length,'corrupt shard '+name);files[name]=read(path.join(dir,name));ok(files[name].length===s.rows,'row mismatch '+name)}
 let audited=0,terminal=0,positiveMission=0,positiveGate=0,negativeStale=0,oldestAge=Infinity,goalUnknown=0;const worldOwner=new Map(),inputs=new Map();
 for(const [split,info] of Object.entries(mf.splits)){
  const policy=files[split+'_policy.jsonl'],challenge=files[split+'_challenge_EVALUATOR_ONLY.jsonl'];
  ok(challenge.length===info.worlds,'missing paired evaluator challenge '+split);
  let pointer=0,zero=0;
  for(let j=0;j<info.worlds;j++){
   const w=v.candidate(split,j).w,meta=challenge[j];ok(meta.geometry_sha256===w.geometry_sha256&&meta.source_seed===w.seed,'wrong world provenance');
   ok(!worldOwner.has(w.geometry_sha256)||worldOwner.get(w.geometry_sha256)===split,'geometry split contamination');worldOwner.set(w.geometry_sha256,split);
   const envs=[v.prepare(w,false,false),v.prepare(w,false,true),v.prepare(w,true,false),v.prepare(w,true,true),v.prepare(w,false,false,'same')];
   const initial=envs.map(v.observe),acts=initial.map(independentAction);
   ok(H(v.ablateMission(initial[0]))===H(v.ablateMission(initial[1])),'mission ablation mismatch');
   ok(acts[0]!==acts[1]&&acts[0]!==acts[2]&&acts[0]===acts[4],'counterfactual / stale negative failed');
   ok(H(v.ablateMission(initial[0]))===H(v.ablateMission(initial[4])),'stale control leaked');
   ok(initial.every(i=>i.recent.length<=32&&i.recent_positions.length<=32),'recent memory exceeded');
   const delay=envs[0].accepted-envs[0].events.filter(e=>e.kind==='MISSION_ASSIGN').at(-1).at;
   ok(delay>=176&&meta.last_mission_age_actions===delay,'historical instruction not aged');oldestAge=Math.min(oldestAge,delay);
   const base=host.ranks(envs[0].g,w.s,w.goal,-1),swapped=host.ranks(envs[1].g,w.s,w.goal2,-1);
   const wrong1=base.find(x=>x.action===acts[1]),wrong2=swapped.find(x=>x.action===acts[0]);
   ok(base[0].action===acts[0]&&swapped[0].action===acts[1]&&wrong1.cost-base[0].cost>=1&&wrong2.cost-swapped[0].cost>=1,'strict mission regret false');
   ok(H(meta.first_actions)===H(acts),'initial actions differ from challenge');
   positiveMission++;positiveGate++;negativeStale++;
   for(let arm=0;arm<envs.length;arm++){
    const h=envs[arm],len=meta.terminal_steps[arm];ok(len>0&&len<=256,'invalid replay length');
    for(let turn=0;turn<len;turn++){
     const row=policy[pointer++];ok(row&&Object.keys(row).sort().join(',')==='action,input,weight','invalid row envelope');
     const current=v.observe(h);ok(H(current)===H(row.input),'not actual host observation');checkObservation(row.input,h);
     const independent=independentAction(row.input);ok(row.action===independent,'teacher mislabel: '+turn+' '+split);
     ok(row.input.legal.includes(row.action),'illegal action');
     const hash=H(row.input),prior=inputs.get(hash);ok(!prior||prior.split===split,'cross-split policy leakage');ok(!prior||prior.action===row.action,'contradictory identical input');
     if(prior&&split==='TRAIN'){ok(row.weight===0,'duplicate nonzero TRAIN');zero++;}else{ok(row.weight===1,'unexpected weight zero');inputs.set(hash,{split,action:row.action});}
     if(row.action.startsWith('MOVE_')&&!row.input.survey.nodes.some(p=>k(p)===k(v.auditTarget(row.input)[0])))goalUnknown++;
     v.accepted(h,row.action);audited++;
    }ok(h.done&&h.success,'host terminal not successful');terminal++;
   }
  }
  ok(pointer===policy.length&&zero===info.duplicate_zero_weight,'split row accounting error');
 }
 ok(audited===mf.total_policy_decisions&&worldOwner.size===mf.unique_world_geometries,'total mismatch');
 return {verified_rows:audited,successful_host_episodes:terminal,positive_mission_pairs:positiveMission,positive_remote_gate_pairs:positiveGate,negative_stale_instruction_controls:negativeStale,worlds:worldOwner.size,minimum_last_instruction_age:oldestAge,move_labels_while_goal_node_unseen:goalUnknown};}
if(require.main===module){const [dir,second,mode]=process.argv.slice(2);ok(dir,'usage test_mission_memory_v46.js DATA [SECOND_DATA] [--quick]');const opts={quick:second==='--quick'||mode==='--quick'},report=audit(dir,opts);
 if(second&&second!=='--quick'){audit(second,opts);const as=fs.readdirSync(dir).sort(),bs=fs.readdirSync(second).sort();ok(H(as)===H(bs),'different file lists');for(const f of as)ok(H(fs.readFileSync(path.join(dir,f)))===H(fs.readFileSync(path.join(second,f))),'nonrepeatable '+f);}
 console.log(JSON.stringify(report,null,2));}
module.exports={audit,independentAction,checkObservation};
