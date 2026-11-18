'use strict';
/* Independent v53 synthetic-host replay: reconstruct observations/receipts and
 * teacher labels through the independent v52 auditor, not producer's rollout.
 * This is not a learned-policy test or a native CyberSand test.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const gen=require('./deferred_surface_v53.js'),aud=require('./test_realert_v52.js');
const v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(x,m)=>{if(!x)throw Error('INDEPENDENT_V53 '+m)};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const cp=x=>JSON.parse(JSON.stringify(x));
function independentB(h,arm,signal,step){if(!arm.when||h.auditB.sent)return false;
 if((arm.when==='initial'&&step===0)||(arm.when==='second'&&signal==='again')){
  ok(h.stage===0,'B warning stage A');ok(h.painted[h.siteB]===undefined,'B safe site occupied');
  if(arm.flipB)h.laws.B=h.laws.B==='NORMAL'?'SWAPPED':'NORMAL';
  h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});
  h.auditB.sent=true;h.auditB.at=h.accepted;return true;
 }return false;
}
function externalObservation(h){const o=aud.observation(h);o.v='synthetic-v53-deferred-cross-surface';return o;}
function validate(dir,split,mf,geom,input){const policy=fs.readFileSync(path.join(dir,split+'_policy.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 const challenges=fs.readFileSync(path.join(dir,split+'_challenge_EVALUATOR_ONLY.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 ok(challenges.length===mf.splits[split].worlds&&policy.length===mf.splits[split].policy_decisions,'manifest rows');
 let pos=0,episodes=0,probes=0,dupes=0,causalPairs=0,negativePairs=0,maxAge=0,hiddenFlipFailures=0;
 for(let ix=0;ix<challenges.length;ix++){
  const w=v46.candidate(split,700+ix).w,m=challenges[ix],old=geom.get(w.geometry_sha256);
  ok(m.geometry_sha256===w.geometry_sha256,'source topology identity');ok(!old||old===split,'cross-split topology leak');geom.set(w.geometry_sha256,split);
  const traces=[];
  for(let aidx=0;aidx<gen.ARMS.length;aidx++){
   const arm=gen.ARMS[aidx],h=aud.prepare(w,gen.armParams(arm));h.auditB={sent:false,at:null};
   let bNotices=0,aNotices=0,nprobe=0,goals=0,age=null,second=null,probeB=null,finalB=null,firstA=null;
   for(let t=0;t<m.episode_steps[aidx];t++){
    const signal=aud.inject(h,gen.armParams(arm));if(signal)aNotices++;
    if(independentB(h,arm,signal,t))bNotices++;
    const obs=externalObservation(h),row=policy[pos++];ok(row,'missing policy row');
    ok(same(obs,row.input),'host observation diverged '+split+'/'+ix+'/'+aidx+'/'+t);
    ok(row.action===aud.action(obs),'independent teacher differs '+split+'/'+ix+'/'+aidx+'/'+t);
    ok(obs.legal.includes(row.action),'illegal label');
    ok(!/"(?:laws|auditB|flipB|againFlip|futureB|arm|protected_damage)"/.test(JSON.stringify(obs)),'hidden host metadata leaked');
    const key=H(obs),prior=input.get(key);ok(!prior||prior.split===split,'cross-split input leak');
    ok(!prior||prior.action===row.action,'label conflict');
    if(split==='TRAIN'&&prior){ok(row.weight===0,'duplicate TRAIN not masked');dupes++;}
    if(!prior)input.set(key,{split,action:row.action});
    if(signal==='again')second={input:cp(obs),action:row.action};
    if(h.stage===1&&h.pos===h.w.goal2&&age===null&&h.auditB.sent)age=h.accepted-h.auditB.at;
    const active=h.stage===0?'A':'B',at=h.pos,doneGoal=active==='A'?h.w.goal:h.w.goal2;
    const r=aud.apply(h,row.action,gen.armParams(arm));
    if(row.action.startsWith('PAINT_')){
     if(at===doneGoal){ok(r.material===active,'protected material wrong');goals++;
      if(active==='A')firstA={input:cp(obs),action:row.action};else finalB={input:cp(obs),action:row.action};
     }else{ok(row.action==='PAINT_A','probe command not A');nprobe++;probes++;
      if(active==='B')probeB={input:cp(obs),material:r.material,action:row.action};}
    }
   }
   ok(h.done&&h.success&&!h.protected_damage,'host terminal failed '+split+'/'+ix+'/'+arm.id);
   ok(nprobe===m.physical_probe_paints[aidx]&&goals===2,'probe/goal counts');
   ok(aNotices===Number(!!arm.alertA)+Number(!!arm.again),'A alert missing');
   ok(bNotices===Number(!!arm.when),'B alert missing');
   ok(!Object.entries(h.painted).some(([k])=>+k!==h.w.goal&&+k!==h.w.goal2),'safe site not erased');
   if(arm.when){ok(age!==null&&age>0,'B warning not deferred');maxAge=Math.max(maxAge,age);}
   if(arm.when&&arm.flipB)hiddenFlipFailures++;
   episodes++;traces.push({probeB,finalB,firstA,second});
  }
  for(const [a,b] of [[1,2],[3,4],[6,7],[9,10],[12,13],[14,15],[16,17]]){
   const x=traces[a],y=traces[b];ok(same(x.probeB.input,y.probeB.input),'B hidden drift leak');
   ok(x.probeB.material!==y.probeB.material&&x.finalB.action!==y.finalB.action,'B physical causality failed');
   const abl=o=>{const z=cp(o),warn=z.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]==='B').at(-1);z.durable=z.durable.filter(e=>!(e[1]==='PAINT_RESULT'&&e[4]==='B'&&e[0]>warn[0]));return z;};
   ok(same(abl(x.finalB.input),abl(y.finalB.input)),'B post-receipt ablation mismatch');causalPairs++;
  }
  for(const [a,b] of [[0,1],[5,6],[8,9],[11,12]]){
   ok(traces[a].firstA.action===traces[b].firstA.action,'irrelevant B changed A goal material');negativePairs++;
  }
  for(const [a,b] of [[8,9],[11,12]]){
   const x=traces[a].second,y=traces[b].second;ok(x&&y&&x.action===y.action,'coincident warning changed A action');
   const z=cp(y.input);z.durable=z.durable.filter(e=>!(e[1]==='SURFACE_ALERT'&&e[2]==='B'));
   ok(same(x.input,z),'coincident observation changed other than B warning');
  }
 }
 ok(pos===policy.length,'orphaned policy rows');
 ok(episodes===mf.splits[split].synthetic_host_successes&&probes===mf.splits[split].host_probe_paints,'aggregate count mismatch');
 ok(causalPairs===mf.splits[split].deferred_B_physical_causal_pairs&&negativePairs===mf.splits[split].irrelevant_B_warning_negative_pairs,'counterfactual counts');
 ok(maxAge===mf.splits[split].max_deferred_warning_age_actions,'deferred age bound');
 for(const [file,info] of Object.entries(mf.shards))if(file.startsWith(split+'_')){
  const b=fs.readFileSync(path.join(dir,file));ok(b.length===info.bytes&&H(b)===info.sha256,'shard SHA or size');
  ok(info.rows===(file.includes('challenge')?challenges.length:policy.length),'shard row count');
 }
 return {worlds:challenges.length,host_successes:episodes,policy_decisions:pos,physical_probes:probes,
  B_causal_pairs:causalPairs,irrelevant_B_negative_pairs:negativePairs,max_future_warning_age_actions:maxAge,
  exact_duplicates_masked_train:dupes,stale_tool_law_wrong_goal_if_B_warning_ignored: hiddenFlipFailures};
}
function main(){const dir=path.resolve(process.argv[2]);const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 ok(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'deferred_surface_v53.js'))),'generator hash');
 for(const p of mf.parents)ok(p.sha256===H(fs.readFileSync(path.join(__dirname,p.file))),'parent hash '+p.file);
 const results={},geometry=new Map(),inputs=new Map(),target=process.argv[3];
 for(const split of target?[target]:['TRAIN','DEV','HELDOUT']){ok(['TRAIN','DEV','HELDOUT'].includes(split),'unknown split');
  results[split]=validate(dir,split,mf,geometry,inputs);}
 console.log(JSON.stringify({status:'PASS independent host replay, counterfactual and negative controls, accepted actions, integrity and split isolation',results},null,2));
}
if(require.main===module)main();module.exports={validate};