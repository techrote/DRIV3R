'use strict';
// Independent exhaustive policy-input auditor; does not read oracle route labels.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const m=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(c,s)=>{if(!c)throw Error(s)};
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(x=>JSON.parse(x));
function independent(input){const g=input.geometry,idx=new Map(g.coords.map((x,i)=>[x.join(','),i]));const edges=g.links.map(([a,b])=>[a,b]);
 // Gate endpoints already refer to integer indexes; direct comparison is unambiguous.
 const gate2=edges.findIndex(([a,b])=>(a===g.gate[0]&&b===g.gate[1])||(b===g.gate[0]&&a===g.gate[1]));
 ok(gate2>=0,'gate not graph edge');
 const events=input.durable.filter(z=>z[1]==='EDGE_GATE');ok(events.length>=2,'missing old gate history');
 const closed=events.at(-1)[3]===1;
 const adj=g.coords.map(()=>[]);for(let e=0;e<edges.length;e++){if(e===gate2&&closed)continue;const [a,b]=edges[e];adj[a].push(b);adj[b].push(a)}
 const p=input.p;const actions=[];for(const b of adj[p]){const dx=g.coords[b][0]-g.coords[p][0],dy=g.coords[b][1]-g.coords[p][1];const d=dx===1?'MOVE_E':dx===-1?'MOVE_W':dy===1?'MOVE_S':'MOVE_N';actions.push([d,b])}
 let legal=actions.map(([d])=>d);if(input.painted.find(([k])=>k===p))legal.push('ERASE');else legal.push('PAINT_A','PAINT_B');legal.push('STOP');
 ok(H(legal.sort())===H(input.legal),'policy legal mismatch');
 const tgt=input.goal?.[0];if(input.stage===2)return {expected:'STOP',closed};
 ok(tgt!==undefined,'missing target');const d=Array(adj.length).fill(Infinity),q=[tgt];d[tgt]=0;for(let z=0;z<q.length;z++)for(const b of adj[q[z]])if(d[b]===Infinity){d[b]=d[q[z]]+1;q.push(b)}
 ok(Number.isFinite(d[p]),'unreachable objective');if(p===tgt){const mat=input.goal[1],at=input.painted.find(([i])=>i===p);return {expected:at&&at[1]!==mat?'ERASE':'PAINT_'+mat,closed}}
 const opts=actions.filter(([a,b])=>d[b]===d[p]-1).map(([a])=>a);ok(opts.length,'no shortest move');return {expected:opts,closed};
}
function audit(dir, quick=false){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'))),shards={};let n=0,wrong=0,contrasts=0,controls=0,contaminated=0;const seen=new Map(),owners=new Map();
 ok(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),'generator mismatch');
 for(const [f,info] of Object.entries(mf.shards)){const b=fs.readFileSync(path.join(dir,f));ok(H(b)===info.sha256&&b.length===info.bytes,'shard byte hash '+f);shards[f]=read(path.join(dir,f));ok(shards[f].length===info.rows,'shard count '+f)}
 for(const [split,stat] of Object.entries(mf.splits)){
  const rows=shards[split+'_policy.jsonl'],ch=shards[split+'_challenge_EVALUATOR_ONLY.jsonl'];ok(ch.length===2*stat.worlds,'challenge pairs count');ok(stat.terminal_episodes===4*stat.worlds,'terminal count');
  let nzero=0;for(const r of rows){n++;ok(Object.keys(r).sort().join(',')==='action,input,weight','teacher leaked top-level');const i=r.input;ok(!('oracle' in i)&&!('world_index' in i)&&!('seed' in i)&&!('closed' in i),'teacher field leaked input');ok(r.weight===0||r.weight===1,'bad weight');nzero+=1-r.weight;
   ok(i.legal.includes(r.action),'illegal action label');let check=independent(i);ok(Array.isArray(check.expected)?check.expected.includes(r.action):check.expected===r.action,'nonoptimal or incompatible action '+r.action);ok(i.durable.filter(e=>e[1]==='EDGE_GATE').at(-1)[0]<i.recent_positions.length+1000,'events malformed');
   const hash=H(i),prev=seen.get(hash);ok(!prev||prev.action===r.action,'label collision');ok(!prev||prev.split===split,'split leak');seen.set(hash,{split,action:r.action});}
  ok(nzero===stat.zero_weight,'weight mismatch');
  let cursor=0;for(let wi=0;wi<stat.worlds;wi++){const w=m.scenario(split,wi);for(const kind of ['positive','control'])for(const closed of [false,true]){const h=m.prepare(w,kind,closed);const c=ch[wi*2+(kind==='positive'?0:1)];const length=closed?c.steps_closed:c.steps_open;for(let z=0;z<length;z++){const row=rows[cursor++];ok(H(m.input(h))===H(row.input),'independent saved-input/host replay mismatch '+split+':'+wi+':'+z);ok(m.apply(h,row.action).accepted,'replay refused');}ok(h.done&&h.success,'replay failed terminal')}}ok(cursor===rows.length,'unconsumed policy rows');
  for(const c of ch){const own=owners.get(c.geometry_sha256);ok(!own||own===split,'graph leakage');owners.set(c.geometry_sha256,split);ok(c.regret_open===null||c.regret_open>=0,'negative regret');
   if(c.positive){contrasts++;ok(c.first_open!==c.first_closed&&c.regret_open>=2&&c.regret_closed>=2,'not causal contrast');}
   else{controls++;ok(c.first_open===c.first_closed&&c.regret_open===0&&c.regret_closed===0,'control changed first choice')}
   ok(c.steps_open>0&&c.steps_closed>0,'empty continuation');
  }
 }
 ok(contrasts>0&&controls>0,'missing controls');return {audited_decisions:n,positives:contrasts,controls,distinct_topologies:owners.size,teacher_input_leaks:wrong,shards:Object.keys(shards).length};}
if(require.main===module){const [dir,other]=process.argv.slice(2);ok(dir,'usage: node test_unseen_topology_v44.js DIR [IDENTICAL_DIR]');const report=audit(dir);if(other){const b=audit(other);ok(H(report)===H(b),'audit mismatch');for(const f of fs.readdirSync(dir)){ok(H(fs.readFileSync(path.join(dir,f)))===H(fs.readFileSync(path.join(other,f))),'nondeterminism '+f)}}console.log(JSON.stringify(report,null,2))}
module.exports={audit,independent};
