'use strict';
/* Independent source-aware audit for v45 data. No model inference. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v=require('./frontier_memory_v45.js'),host=require('./unseen_topology_v44.js');
const ok=(c,s)=>{if(!c)throw Error(s)};
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const pkey=p=>p.join(',');const ekey=(a,b)=>[pkey(a),pkey(b)].sort().join('|');
function independentlyChoose(i){if(i.stage===2)return 'STOP';const [goal,mat]=i.target,at=i.painted.find(x=>pkey(x[0])===pkey(i.p));
 if(pkey(i.p)===pkey(goal))return at&&at[1]!==mat?'ERASE':'PAINT_'+mat;
 const latest=i.durable.filter(x=>x[1]==='EDGE_GATE').at(-1);const blocked=latest&&latest[4]===1?ekey(latest[2],latest[3]):null;
 const g=new Map(i.survey.nodes.map(x=>[pkey(x),[]]));for(const [a,b] of i.survey.edges){if(ekey(a,b)===blocked)continue;g.get(pkey(a)).push(pkey(b));g.get(pkey(b)).push(pkey(a))}
 const d=new Map([[pkey(i.p),0]]),parents=new Map(),q=[pkey(i.p)];for(let z=0;z<q.length;z++)for(const k of g.get(q[z])||[])if(!d.has(k)){d.set(k,d.get(q[z])+1);parents.set(k,q[z]);q.push(k)}
 let chosen=pkey(goal);if(!d.has(chosen)){
  const visited=new Set(i.survey.visited.map(pkey)),poss=q.filter(x=>!visited.has(x));ok(poss.length>0,'frontier absent');
  poss.sort((a,b)=>{const pos=x=>x.split(',').map(Number),target=goal;
   const score=x=>{const z=pos(x);return 2*(Math.abs(z[0]-target[0])+Math.abs(z[1]-target[1]))+d.get(x)};
   return score(a)-score(b)||a.localeCompare(b)});chosen=poss[0];}
 let step=chosen;while(parents.get(step)!==pkey(i.p)){step=parents.get(step);ok(step,'invalid parent')}
 const [x,y]=step.split(',').map(Number),[px,py]=i.p;
 return x===px+1?'MOVE_E':x===px-1?'MOVE_W':y===py+1?'MOVE_S':y===py-1?'MOVE_N':'ERROR';}
function validateSurvey(i,h){const g=h.g,observed=new Set(h.positions),nodes=new Set(),edges=new Set();for(const at of observed){nodes.add(pkey(g.coords[at]));for(const [to] of g.adj[at]){nodes.add(pkey(g.coords[to]));edges.add(ekey(g.coords[at],g.coords[to]))}}
 ok(H([...nodes].sort())===H(i.survey.nodes.map(pkey).sort()),'unobserved geometry leaked');
 ok(H([...edges].sort())===H(i.survey.edges.map(x=>ekey(...x)).sort()),'wrong surveyed edges');
 ok(H([...observed].map(x=>pkey(g.coords[x])).sort())===H(i.survey.visited.map(pkey).sort()),'visited-map mismatch');
 ok(i.survey.nodes.length<g.coords.length || i.stage>0,'survey prematurely reveals complete graph');
}
function audit(dir,{quick=false}={}){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));ok(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'frontier_memory_v45.js'))),'wrong generator hash');ok(mf.parent_sha256===H(fs.readFileSync(path.join(__dirname,'unseen_topology_v44.js'))),'wrong parent hash');ok(mf.quick===quick,'quick flag');
 const files={};for(const [name,info] of Object.entries(mf.shards)){const b=fs.readFileSync(path.join(dir,name));ok(H(b)===info.sha256&&b.length===info.bytes,'shard mismatch '+name);files[name]=read(path.join(dir,name));ok(files[name].length===info.rows,'wrong row count '+name)}
 let total=0,terminal=0,pairs=0,controls=0;const labels=new Map(),worlds=new Map();
 for(const [split,stats] of Object.entries(mf.splits)){
  const policy=files[split+'_policy.jsonl'],ch=files[split+'_challenge_EVALUATOR_ONLY.jsonl'];ok(ch.length===stats.worlds*2,'wrong evaluation pair count');let pointer=0,zeros=0;
  for(let index=0;index<stats.worlds;index++){
   const candidate=v.candidate(split,index),w=candidate.w,prior=worlds.get(w.geometry_sha256);ok(!prior||prior===split,'cross-split world leakage');worlds.set(w.geometry_sha256,split);
   for(let k=0;k<2;k++){
    const kind=k?'control':'positive',meta=ch[index*2+k];ok(meta.kind===kind&&meta.source_seed===w.seed&&meta.geometry_sha256===w.geometry_sha256,'wrong source index');
    if(meta.positive){pairs++;ok(meta.first_open!==meta.first_closed&&meta.wrong_action_regret_open>=2&&meta.wrong_action_regret_closed>=2,'weak contrast')}
    else{controls++;ok(meta.first_open===meta.first_closed,'control misclassified')}
    ok(meta.initial_coverage_fraction<=.82&&meta.initial_coverage_fraction>0,'not a partial survey');
    const a=v.prelude(w,kind,false),b=v.prelude(w,kind,true),ia=v.observe(a),ib=v.observe(b);
    ok(H(v.ablateEvent(ia))===H(v.ablateEvent(ib)),'pair leak after memory ablation');
    ok(H(ia.recent)===H(ib.recent)&&H(ia.legal)===H(ib.legal)&&H(ia.local)===H(ib.local),'current/recent leak');
    const last=ia.durable.filter(x=>x[1]==='EDGE_GATE').at(-1);ok(a.accepted-last[0]>=144,'event not sufficiently aged');
    for(let arm=0;arm<2;arm++){
     const h=arm?b:a,len=meta.episode_steps[arm];ok(len>0&&len<=256,'episode size');
     for(let z=0;z<len;z++){
      const row=policy[pointer++];ok(row,'shard too short');ok(Object.keys(row).sort().join(',')==='action,input,weight','teacher metadata leaked');
      const i=row.input;ok(H(v.observe(h))===H(i),'observation not exact host replay');
      ok(row.action===independentlyChoose(i),'bad observer-only label');ok(i.legal.includes(row.action),'illegal label');
      ok(!Object.keys(i).some(z=>['seed','world_index','geometry','oracle','teacher','closed','hidden','ground_truth'].includes(z)),'privileged metadata in training');
      validateSurvey(i,h);const kk=H(i),prev=labels.get(kk);ok(!prev||prev.split===split,'cross-split policy leakage');ok(!prev||prev.action===row.action,'conflicting policy labels');
      if(prev&&split==='TRAIN')ok(row.weight===0,'repeated training row not downweighted');else {ok(row.weight===1,'new row weight 0');labels.set(kk,{split,action:row.action})}
      zeros+=1-row.weight;ok(host.apply(h,row.action).accepted,'host refused action '+row.action);total++;
     }
     ok(h.done&&h.success,'incomplete terminal');terminal++;
    }
   }
  }
  ok(pointer===policy.length&&zeros===stats.zero_weight,'row accounting mismatch '+split);
 }
 ok(total===mf.stats.total_policy_decisions&&terminal===4*worlds.size,'total mismatch');return {audited_decisions:total,terminal_successes:terminal,positive_pairs:pairs,negative_controls:controls,distinct_worlds:worlds.size,verified: 'independent input-only routing, full physical host replay, survey nonleak, event ablation, split isolation'};}
if(require.main===module){const [dir,other,quick]=process.argv.slice(2);ok(dir,'usage node test_frontier_memory_v45.js DIR [BYTE_IDENTICAL_DIR] [--quick]');const opts={quick:quick==='--quick'||other==='--quick'},r=audit(dir,opts);if(other&&other!=='--quick'){audit(other,opts);for(const f of fs.readdirSync(dir))ok(H(fs.readFileSync(path.join(dir,f)))===H(fs.readFileSync(path.join(other,f))),'not byte identical '+f)}console.log(JSON.stringify(r,null,2))}
module.exports={audit,independentlyChoose,validateSurvey};
