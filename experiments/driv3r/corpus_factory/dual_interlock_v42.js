'use strict';
// DRIV3R Corpus Factory v42: two independently operated remote interlocks,
// old causal memory and matched negative controls. Synthetic host, not native CyberSand.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js');
const old=require('./delayed_obstruction_v41.js'),view=require('./bounded_view_v30.js');
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const check=(p,msg)=>{if(!p)throw Error(msg);};
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:4,DEV:2,HELDOUT:2};
const SOURCE='synthetic-v42-dual-interlock';
function withoutB(input){const x=clone(input);x.durable=x.durable.filter(e=>e[1]!=='REMOTE_INTERLOCK_B');return x;}
function withoutInterlocks(input){const x=clone(input);x.durable=x.durable.filter(e=>!['REMOTE_INTERLOCK_A','REMOTE_INTERLOCK_B'].includes(e[1]));return x;}
function inputOf(obs){
 const x=view.compress(obs);
 const events=obs.history.events.flatMap((e,i)=>e.kind==='INTERLOCK_TRANSITION'&&e.accepted?[[i,'REMOTE_INTERLOCK_'+e.id,e.value]]:
 e.kind==='SENSOR_DIAGNOSTIC'&&e.accepted?[[i,'REMOTE_SENSOR',e.code]]:[]);
 check(events.filter(x=>x[1]==='REMOTE_INTERLOCK_A').length>=2,'missing A transitions');
 check(events.filter(x=>x[1]==='REMOTE_INTERLOCK_B').length>=2,'missing B transitions');
 x.v=SOURCE;x.durable.push(...events);x.durable.sort((a,b)=>a[0]-b[0]);return x;
}
function plan(variant,finalB,finalA){
 // 160 accepted, host-executed movements; final B is 112 moves old,
 // last A is 80 moves old. A=1 and B=1 are BOTH needed to close remote graph.
 const a=[0,'A',1],last=[80,'A',finalA],sense=[96,'S',3];
 const sched=[
  [a,[32,'B',1-finalB],[48,'B',finalB],last,sense],
  [a,[24,'S',5],[32,'B',1-finalB],[48,'B',finalB],last,sense],
  [[0,'B',1-finalB],[24,'A',1],[48,'B',finalB],last,sense],
  [a,[16,'B',1-finalB],[32,'S',7],[48,'B',finalB],last,sense]
 ][variant];
 check(sched,'unknown schedule');return sched;
}
function caseAt(w,split,g,v,arm,finalB,finalA=1,variant=0){
 const original=base.prepare(w,split,g,v,arm).h;
 const steps=160,ev=original.history.events,pos=original.history.positions;
 const tail=ev.slice(-steps),prefix=clone(ev.slice(0,-steps)),prefixPos=pos.slice(0,-steps);
 check(tail.length===steps&&tail.every(e=>e.kind==='ACTION_ACK'&&e.accepted&&!e.action.startsWith('PAINT')&&e.action!=='ERASE'),'not accepted movement-only ageing');
 check(prefixPos.at(-1)==='0,0'&&prefixPos.length>2,'start not safe root');
 const h=new host.Host(old.obstruct(w,'CLEAR'),split,g,v);
 h.pos=h.w.nidx.get(prefixPos.at(-1));h.prev=h.w.nidx.get(prefixPos.at(-2));
 h.status=original.status;h.latched=original.latched;h.remaining=original.remaining+steps;
 h.history={events:prefix,positions:prefixPos.slice()};
 check(h.latched===1,'prelude has not opened gate');
 let A=0,B=0,physicalChanges=0,bChanges=0;const schedule=plan(variant,finalB,finalA);
 const lastById={};
 for(let i=0;i<steps;i++){
   for(const [at,id,value] of schedule)if(at===i){
     if(id==='S'){
       h.history.events.push({kind:'SENSOR_DIAGNOSTIC',accepted:true,source:'REMOTE_SENSOR',code:value});
     }else{
       const before=A&B;
       if(id==='A')A=value;else B=value;
       const after=A&B;
       const next=old.obstruct(w,after?'BLOCKED':'CLEAR');
       check(next.phase===(after?'BLOCKED':'CLEAR'),'state/physical mismatch');
       h.w=next;
       h.history.events.push({kind:'INTERLOCK_TRANSITION',accepted:true,id,value,cause:'REMOTE_ACTUATOR'});
       if(before!==after){physicalChanges++;if(id==='B')bChanges++;}
       lastById[id]=i;
     }
   }
   const e=tail[i],r=host.apply(h,e.action);
   check(r.accepted&&r.action===e.action&&r.at===e.at&&r.painted_correct===e.painted_correct,'host replay diverged '+i);
 }
 check(A===finalA&&B===finalB&&h.w.phase===(finalA&&finalB?'BLOCKED':'CLEAR'),'final interlocks wrong');
 check(h.pos===original.pos&&h.prev===original.prev&&h.status===original.status&&h.latched===original.latched&&h.remaining===original.remaining,'intervention modified shared state');
 const obs=host.observe(h);
 check(hash(obs.history.events.slice(-64))===hash(ev.slice(-64)),'recent event leak');
 check(hash(obs.history.positions.slice(-64))===hash(pos.slice(-64)),'recent location leak');
 check(steps-lastById.A>64&&steps-lastById.B>64,'interlock not durable');
 const input=inputOf(obs),planner=base.oracle(h.w),rank=planner.rank([h.pos,h.prev,h.status,h.latched]);
 check(rank.length>0&&Number.isFinite(rank[0].cost),'unreachable oracle');
 return {h,w:h.w,input,rank,planner,arm,variant,finalA,finalB,
   ageA:steps-lastById.A,ageB:steps-lastById.B,physicalChanges,bChanges};
}
function recover(c){
 const h=c.h,rows=[],initial=c.rank[0].cost;
 while(!h.done){
   check(rows.length<180,'not terminal');
   const obs=host.observe(h),rank=c.planner.rank([h.pos,h.prev,h.status,h.latched]),best=rank[0];
   check(best&&Number.isFinite(best.cost),'no live suffix');
   const input=inputOf(obs),receipt=host.apply(h,best.action);
   check(receipt.accepted,'oracle refused');
   const remain=h.done?0:c.planner.rank([h.pos,h.prev,h.status,h.latched])[0].cost;
   check(best.cost===remain+1,'not optimal accepted-action suffix');
   rows.push({input,action:best.action,weight:1});
 }
 check(rows.length===initial&&host.correct(c.w,h.status).wrong===0,'bad terminal');
 return rows;
}
function pairAt(w,split,g,v,arm,variant,finalA){
 const a=caseAt(w,split,g,v,arm,0,finalA,variant),b=caseAt(w,split,g,v,arm,1,finalA,variant);
 check(hash(withoutB(a.input))===hash(withoutB(b.input)),'B ablation leaked non-B difference');
 check(hash(a.input)!==hash(b.input),'full context alias');
 check(hash(a.input.recent)===hash(b.input.recent),'recent actions differ');
 check(a.input.legal.join(',')===b.input.legal.join(','),'first difference is legality');
 const wrongA=a.rank.find(x=>x.action===b.rank[0].action),wrongB=b.rank.find(x=>x.action===a.rank[0].action);
 const regretA=wrongA?wrongA.cost-a.rank[0].cost:null,regretB=wrongB?wrongB.cost-b.rank[0].cost:null;
 return {a,b,contrast:finalA===1&&a.rank[0].action!==b.rank[0].action&&regretA>=2&&regretB>=2,
   meta:{arm,variant,finalA,first_b0:a.rank[0].action,first_b1:b.rank[0].action,
   regret_b0:regretA,regret_b1:regretB,ageA:a.ageA,ageB:a.ageB,
   physical_b0:a.physicalChanges,physical_b1:b.physicalChanges,b_changed_physical_b0:a.bChanges,b_changed_physical_b1:b.bChanges}};
}
function save(dest,file,rows){const b=Buffer.from(rows.map(r=>JSON.stringify(r)).join('\n')+'\n');fs.writeFileSync(path.join(dest,file),b,{flag:'wx'});return {rows:rows.length,bytes:b.length,sha256:hash(b)};}
function generate(dest,{quick=false}={}){
 check(!fs.existsSync(dest),'no overwrite');fs.mkdirSync(dest,{recursive:true});
 const splits={},shards={},owners=new Map(),geometry=new Map();
 for(const [split,num] of Object.entries(SPLITS)){
   const out=[],challenges=[],seen=new Map(),groups=quick?1:num;
   let candidates=0,positives=0,controls=0,episodes=0,zeroWeight=0,terminal=0;const splitGeometries=new Set();
   for(let g=0;g<groups;g++)for(let v=0;v<8;v++){
     const w=old.makeWorld(split,g,v),priorGeo=geometry.get(w.geometry_sha256);splitGeometries.add(w.geometry_sha256);
     check(!priorGeo||priorGeo===split,'split graph overlap');geometry.set(w.geometry_sha256,split);
     const pairList=[];
     for(let variant=0;variant<4;variant++)for(let arm=0;arm<4;arm++){
       const p=pairAt(w,split,g,v,arm,variant,1);candidates++;
       if(p.contrast)pairList.push(p);
     }
     check(pairList.length===4,'missing four strict counterfactual schedules');
     const control=pairAt(w,split,g,v,1,0,0);
     check(!control.contrast&&control.a.rank[0].action===control.b.rank[0].action,'interlock-negative control changed optimal decision');
     for(const p of [...pairList,control]){
       const positive=p!==control;positive?positives++:controls++;
       const trajectories=[recover(p.a),recover(p.b)];terminal+=2;episodes+=2;
       if(positive)check(p.a.rank[0].action==='MOVE_E'&&p.b.rank[0].action==='MOVE_S','unexpected strict route contrast');
       challenges.push({split,g,v,world_sha256:w.world_sha256,geometry_sha256:w.geometry_sha256,positive,synthetic_not_native:true,
          ...p.meta,ablate_B_alias_sha256:hash(withoutB(p.a.input)),full_a_sha256:hash(p.a.input),full_b_sha256:hash(p.b.input),
          steps_b0:trajectories[0].length,steps_b1:trajectories[1].length});
       for(const trace of trajectories)for(const row of trace){
         const key=hash(row.input),owner=owners.get(key),prior=seen.get(key);
         check(!owner||owner===split,'cross-split input');owners.set(key,split);
         check(prior===undefined||prior===row.action,'input contradictory label');
         if(prior!==undefined&&split==='TRAIN'){row.weight=0;zeroWeight++;}else seen.set(key,row.action);
         out.push(row);
       }
     }
   }
   check(positives&&controls&&terminal===episodes,'empty or nonterminal challenges');
   shards[split+'_policy.jsonl']=save(dest,split+'_policy.jsonl',out);
   if(split!=='TRAIN')shards[split+'_challenge.jsonl']=save(dest,split+'_challenge.jsonl',challenges);
   splits[split]={candidate_pairs:candidates,strict_positive_pairs:positives,negative_controls:controls,episodes,oracle_terminal_successes:terminal,decisions:out.length,zero_weight:zeroWeight,unique_graphs:splitGeometries.size};
 }
 const dependencies=['delayed_obstruction_v41.js','causal_shortcut_v32.js','causal_gate_v26.js','outcome_forks_v25.js','bounded_view_v30.js'];
 const manifest={schema:'driv3r/corpus_factory/dual-interlock-v42/manifest',synthetic_not_native_CyberSand:true,
  generator:'dual_interlock_v42.js',generator_sha256:hash(fs.readFileSync(__filename)),
  dependencies:dependencies.map(file=>({file,sha256:hash(fs.readFileSync(path.join(__dirname,file)))})),
  design:'Independent A/B remote actuator receipts; path blocked iff both are 1. Matched B=0/1 pairs have identical last A, recent events, local view, legality; different optimal actions only if A=1. Negative A=0 controls.',
  labels:'Host-legal accepted-action suffix oracle. Remote actuator transitions are scripted synthetic host events, not agent actions.',
  model_contract:'Only input is policy-visible. action is teacher label and weight=0 excludes duplicated TRAIN examples. DEV/HELDOUT challenges are evaluator-only.',
  quick,splits,shards,caveats:['One hand-built symbolic remote junction with two logical interlocks','Scripted, accepted host events; not native CyberSand','Symbolic two-bit logic can solve challenge','No SLM inference or training undertaken']};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={caseAt,pairAt,withoutB,withoutInterlocks,inputOf,recover,generate,hash};
if(require.main===module){const [dest,flag]=process.argv.slice(2);check(dest,'usage: node dual_interlock_v42.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(dest),{quick:flag==='--quick'}).splits,null,2));}
