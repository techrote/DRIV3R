'use strict';
/* v49 independent audit: never calls v49 teacher, rollout, world, generate or
 * command to decide a policy action; reconstructs host commands and materials. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v49=require('./contextual_probe_v49.js'),v46Audit=require('./test_mission_memory_v46.js');
const v46=require('./mission_memory_v46.js'),host=require('./unseen_topology_v44.js');
const ok=(v,m)=>{if(!v)throw Error(m)};
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const read=f=>fs.readFileSync(f,'utf8').trim().split('\n').map(JSON.parse);
const clean=i=>{const {surface,...x}=i;return {...x,v:'synthetic-v46-historical-missions',durable:i.durable.filter(e=>e[1]!=='PAINT_RESULT')};};
function independentAction(i){const action=v46Audit.independentAction(clean(i));if(!action.startsWith('PAINT_'))return action;
 const evidence=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[2]==='PAINT_A'&&e[4]===i.surface);
 ok(evidence.length===2,'missing contextual calibration');const final=evidence.at(-1)[3];ok(['A','B'].includes(final),'bad physical outcome');
 return final==='A'?action:'PAINT_'+(action.endsWith('A')?'B':'A');}
function step(h,command){ok(host.legal(h).includes(command),'host refuses proposed legal command');
 const prior=h.stage,at=h.accepted,surface=h.surfaces[h.pos]||'A',law=h.laws[surface];
 const actual=command.startsWith('PAINT_')&&law==='SWAPPED'?'PAINT_'+(command.endsWith('A')?'B':'A'):command;
 ok(host.apply(h,actual).accepted&&h.accepted===at+1,'host failed update');
 h.events.filter(e=>e.kind==='ACTION_ACK').at(-1).action=command;
 if(prior!==2&&h.stage===2)h.events.push({kind:'MISSION_COMPLETE',accepted:true,at:h.accepted});
}
function inspectPrelude(h,arm,w){const receipts=h.events.filter(e=>e.kind==='PAINT_RESULT');ok(receipts.length===4,'must perform two probes at each surface');
 const by={A:[],B:[]};for(const e of receipts){ok(['A','B'].includes(e.surface)&&['A','B'].includes(e.material)&&e.issued==='PAINT_A','invalid calibrated observation');
  const real=e.surface==='A'?w.s:h.siteB;
  ok(H(e.site)===H(h.g.coords[real]),'probe site not physical');
  const ack=h.events.find(x=>x.kind==='ACTION_ACK'&&x.at===e.at),erase=h.events.find(x=>x.kind==='ACTION_ACK'&&x.at===e.at+1);
  ok(ack&&ack.action==='PAINT_A'&&ack.pos===real&&erase&&erase.action==='ERASE'&&erase.pos===real,'host PAINT_A/ERASE missing');
  by[e.surface].push(e.material);}
 ok(by.A.length===2&&by.B.length===2,'uneven probe distribution');
 const outcome=law=>law==='NORMAL'?'A':'B';
 ok(by.A[0]===outcome(arm.oldA)&&by.A[1]===outcome(arm.A),'A calibration does not match host law');
 ok(by.B[0]===outcome(arm.oldB)&&by.B[1]===outcome(arm.B),'B calibration does not match host law');
 ok(h.accepted-receipts.at(-1).at>=81,'calibration not delayed');
 ok(h.painted[w.s]===undefined&&h.painted[h.siteB]===undefined,'calibration left paint');
}
function inspectInput(i,h){ok(i.v==='synthetic-v49-dual-surface-physical-probes','wrong protocol');
 for(const secret of ['tool_mode','laws','tool_law','target','stage','source_seed','source_world_index','goal','oracle','action','weight'])ok(!(secret in i),'privileged '+secret);
 ok(i.surface===(h.surfaces[h.pos]||'A'),'current surface differs from accepted host state');
 ok(i.legal.length===host.legal(h).length&&i.legal.every(a=>host.legal(h).includes(a)),'legal menu mismatch');
 const mapped=h.events.filter(e=>e.kind==='PAINT_RESULT').map(e=>[e.at,'PAINT_RESULT',e.issued,e.material,e.surface,e.site]);
 ok(H(i.durable.filter(e=>e[1]==='PAINT_RESULT'))===H(mapped),'fake material receipt');
 v46Audit.checkObservation(clean(i),h);
}
function audit(dir,{quick=false}={}){
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));ok(mf.schema==='driv3r/corpus_factory/contextual-probe-v49/manifest'&&mf.quick===quick,'manifest schema/quick');
 ok(H(fs.readFileSync(path.join(__dirname,mf.generator)))===mf.generator_sha256,'generator hash mismatch');
 for(const p of mf.parents)ok(H(fs.readFileSync(path.join(__dirname,p.file)))===p.sha256,'parent source mismatch '+p.file);
 for(const [name,entry] of Object.entries(mf.shards)){const b=fs.readFileSync(path.join(dir,name));
  ok(b.length===entry.bytes&&H(b)===entry.sha256,'shard corrupt '+name);ok(b.toString('utf8').split('\n').length-1===entry.rows,'row count bad '+name);}
 let episodes=0,actions=0,worlds=0,calibrations=0,firstErrors=0,lastErrors=0,rawPaint=0,staleCtrls=0,revCtrls=0;
 const geometry=new Map(),inputs=new Map(),nearNavigation=new Map();
 for(const [split,info] of Object.entries(mf.splits)){
  const policy=read(path.join(dir,split+'_policy.jsonl')),challenge=read(path.join(dir,split+'_challenge_EVALUATOR_ONLY.jsonl'));
  ok(challenge.length===info.worlds&&info.successful_synthetic_episodes===info.worlds*9,'challenge counts');
  let cursor=0,duplicate=0,nearDuplicate=0;
  for(let j=0;j<info.worlds;j++){
   if(j%2===0)console.error('AUDIT PROGRESS',Date.now(),split,j,'/',info.worlds,'rss',Math.round(process.memoryUsage().rss/1048576));
   const w=v46.candidate(split,j).w,meta=challenge[j];
   ok(meta.source_split===split&&meta.source_seed===w.seed&&meta.geometry_sha256===w.geometry_sha256,'source provenance mismatch');
   ok(!geometry.has(w.geometry_sha256)||geometry.get(w.geometry_sha256)===split,'cross-split geometry');geometry.set(w.geometry_sha256,split);worlds++;
   const hs=v49.arms.map(arm=>v49.prepare(w,arm));calibrations+=hs.length*4;
   for(let arm=0;arm<hs.length;arm++)inspectPrelude(hs[arm],v49.arms[arm],w);
   const start=hs.map(v49.observe),initial=start.map(independentAction);
   ok(H(initial)===H(meta.first_actions),'initial routing actions differ');
   ok(initial[0]===initial[1]&&initial[0]===initial[2]&&initial[0]!==initial[4]&&initial[0]!==initial[5],'route causal controls wrong');
   ok(H(v49.ablateMission(start[0]))===H(v49.ablateMission(start[5])),'mission info leak');
   ok(H(v49.ablateGate(start[0]))===H(v49.ablateGate(start[4])),'gate info leak');
   const paints=[];
   for(let arm=0;arm<hs.length;arm++){
    const h=hs[arm],paint=[];
    for(let t=0;t<meta.episode_steps[arm];t++){
     const row=policy[cursor++];ok(row&&Object.keys(row).sort().join(',')==='action,input,weight','bad trainer record shape');
     const now=v49.observe(h);ok(H(now)===H(row.input),'host-emitted observation not replayable');
     if(t===0||row.action.startsWith('PAINT_')||t%16===0||t===meta.episode_steps[arm]-1)inspectInput(now,h);
     const expected=independentAction(now);ok(expected===row.action,'independent observable-only action disagrees');
     const key=H(now),prior=inputs.get(key);ok(!prior||prior.split===split,'cross-split policy input leakage');ok(!prior||prior.action===row.action,'contradictory label');
     let expectedWeight=prior&&split==='TRAIN'?0:1;if(prior&&split==='TRAIN')duplicate++;else inputs.set(key,{split,action:row.action});
     if(split==='TRAIN'&&row.action.startsWith('MOVE_')){const sig=H(v49.ablateAllProbes(now)),other=nearNavigation.get(sig);
      ok(!other||other===row.action,'near-duplicate navigation label conflict');
      if(other){if(expectedWeight!==0)nearDuplicate++;expectedWeight=0;}else nearNavigation.set(sig,row.action);}
     ok(row.weight===expectedWeight,'incorrect exact/near-duplicate training loss weight');
     if(row.action.startsWith('PAINT_')){
      paint.push({input:now,action:row.action});rawPaint++;
      const physical=v46Audit.independentAction(clean(now));
      const global=now.durable.filter(e=>e[1]==='PAINT_RESULT').at(-1)[3];
      const naive=global==='A'?physical:'PAINT_'+(physical.endsWith('A')?'B':'A');
      if(naive!==row.action){lastErrors++;if(paint.length===1)firstErrors++;}
     }
     step(h,row.action);actions++;
    }
    ok(h.done&&h.success&&paint.length===2,'no correct synthetic terminal completion');
    ok(paint[0].action===meta.first_paint_commands[arm]&&paint[1].action===meta.second_paint_commands[arm],'paint action metadata drift');
    paints.push(paint);episodes++;
   }
   // Counterfactual inputs become identical when only the selected cause is removed.
   ok(H(v49.mask(paints[0][0].input,'A'))===H(v49.mask(paints[1][0].input,'A'))&&paints[0][0].action!==paints[1][0].action,'first-surface causal pair invalid');
   ok(H(v49.mask(paints[0][1].input,'B'))===H(v49.mask(paints[2][1].input,'B'))&&paints[0][1].action!==paints[2][1].action,'second-surface causal pair invalid');
   ok(H(v49.mask(paints[0][0].input,'A','oldest'))===H(v49.mask(paints[6][0].input,'A','oldest')),'obsolete A leaked');
   ok(H(v49.mask(paints[0][1].input,'B','oldest'))===H(v49.mask(paints[7][1].input,'B','oldest')),'obsolete B leaked');
   ok(paints[0][0].action===paints[6][0].action&&paints[0][1].action===paints[7][1].action,'obsolete event altered action');
   ok(paints[0][0].action===paints[8][0].action&&paints[0][1].action===paints[8][1].action,'reversing probe order altered action');
   staleCtrls+=2;revCtrls++;
  }
  ok(cursor===policy.length&&duplicate===info.duplicate_train_zero_weight&&nearDuplicate===info.train_near_duplicate_navigation_zero_weight,'split policy length/dedup mismatch');
 }
 ok(actions===mf.total_policy_decisions&&worlds===mf.source_worlds,'world/decision totals differ');
 return {source_worlds:worlds,successful_synthetic_host_episodes:episodes,independently_verified_policy_decisions:actions,
  host_executed_calibration_paints:calibrations,host_executed_calibration_erases:calibrations,
  causal_surface_A_pairs:worlds,causal_surface_B_pairs:worlds,negative_stale_surface_controls:staleCtrls,
  reverse_order_controls:revCtrls,total_material_paint_labels:rawPaint,
  naive_global_latest_probe_wrong_decisions:lastErrors,naive_global_latest_probe_wrong_first_paint:firstErrors,
  cross_split_leaks:0,contradictory_labels:0};
}
if(require.main===module){const [a,b,flag]=process.argv.slice(2),quick=b==='--quick'||flag==='--quick';ok(a,'usage node test_contextual_probe_v49.js DATA_DIR [SECOND_DATA_DIR] [--quick]');
 const result=audit(a,{quick});if(b&&b!=='--quick'){
  const x=fs.readdirSync(a).sort(),y=fs.readdirSync(b).sort();ok(H(x)===H(y),'non-deterministic filenames');
  for(const file of x)ok(H(fs.readFileSync(path.join(a,file)))===H(fs.readFileSync(path.join(b,file))),'non-deterministic bytes '+file);}
 console.log(JSON.stringify(result,null,2));}
module.exports={audit,independentAction,step,inspectInput,inspectPrelude};
