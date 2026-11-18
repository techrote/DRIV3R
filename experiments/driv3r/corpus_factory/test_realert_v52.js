'use strict';
/* Independent synthetic-host replay and counterfactual audit. No learned policy. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v52=require('./realert_v52.js'),v50=require('./online_recovery_v50.js');
const v49=require('./contextual_probe_v49.js'),v46=require('./mission_memory_v46.js');
const v45=require('./frontier_memory_v45.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error('INDEPENDENT_V52 '+m)};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function surface(h){return h.stage===0?'A':h.stage===1?'B':null;}
function target(h,s){return s==='A'?h.w.goal:h.w.goal2;}
function site(h,s){return s==='A'?h.w.s:h.siteB;}
function prepare(w,arm){const h=v50.prep(w,arm);h.audit={sentFirst:{},sentAgain:false,erasedFirst:false,movesAfterErase:0};return h;}
function inject(h,arm){const s=surface(h);
 if(s&&((s==='A'&&arm.alertA)||(s==='B'&&arm.alertB))&&!h.audit.sentFirst[s]&&h.pos===target(h,s)&&h.painted[h.pos]===undefined){
  h.audit.sentFirst[s]=true;if(s==='A'?arm.flipA:arm.flipB)h.laws[s]=h.laws[s]==='NORMAL'?'SWAPPED':'NORMAL';
  h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:s,site:h.g.coords[site(h,s)]});return 'first';
 }
 if(arm.again&&s===arm.again&&!h.audit.sentAgain&&h.audit.erasedFirst&&h.audit.movesAfterErase>=(arm.delay||1)&&h.pos!==site(h,s)){
  h.audit.sentAgain=true;if(arm.againFlip)h.laws[s]=h.laws[s]==='NORMAL'?'SWAPPED':'NORMAL';
  ok(h.painted[site(h,s)]===undefined,'calibration residue after erase');
  h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:s,site:h.g.coords[site(h,s)]});return 'again';
 }return null;
}
function observation(h){const x=v50.observe(h);x.v='synthetic-v52-temporal-probe-invalidation';
 x.durable.push(...h.events.filter(e=>e.kind==='SURFACE_ALERT').map(e=>[e.at,'SURFACE_ALERT',e.surface,e.site]));
 x.durable.sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));return x;
}
function action(i){if(i.durable.some(e=>e[1]==='MISSION_COMPLETE'))return 'STOP';
 const mission=i.durable.filter(e=>e[1]==='MISSION_ASSIGN'||e[1]==='GOAL_REVISION').at(-1);ok(mission,'no mission');
 const [coord,want]=[mission[2],mission[3]];
 const warnings=i.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===want);
 if(warnings.length){const a=warnings.at(-1),safe=a[3];
  const fresh=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===want&&e[0]>a[0]&&e[2]==='PAINT_A'&&same(e[5],safe));
  const residue=i.painted.some(([p])=>same(p,safe));
  if(!fresh.length){if(!same(i.p,safe))return v45.pathOnSurvey(i,safe);return residue?'ERASE':'PAINT_A';}
  if(residue&&same(i.p,safe))return 'ERASE';
 }
 if(!same(i.p,coord))return v45.pathOnSurvey(i,coord);
 if(i.painted.some(([p])=>same(p,coord)))return 'ERASE';
 const receipts=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===i.surface),last=receipts.at(-1);
 ok(last,'no physical calibration');
 const normal=last[2].at(-1)===last[3];return 'PAINT_'+(normal?want:(want==='A'?'B':'A'));
}
function apply(h,a,arm){const before=h.stage,s=surface(h),at=h.pos,wasPainted=h.painted[at]!==undefined,previous=h.accepted;
 const r=v49.command(h,a);ok(h.accepted===previous+1,'host failed accepted action');
 if(a.startsWith('PAINT_')){ok(r.material==='A'||r.material==='B','paint has no material receipt');
  h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:a,material:r.material,surface:v49.label(h),site:h.g.coords[at]});
  if(s&&h.audit.sentFirst[s]&&at===target(h,s)&&r.material!==s){h.done=true;h.success=false;h.protected_damage=true;}
 }
 if(arm.again&&s===arm.again&&at===site(h,s)&&a==='ERASE'&&wasPainted&&!h.audit.erasedFirst){
  const initial=h.events.find(e=>e.kind==='SURFACE_ALERT'&&e.surface===s);
  const physical=h.events.filter(e=>e.kind==='PAINT_RESULT'&&e.surface===s&&e.at>initial?.at&&same(e.site,h.g.coords[at]));
  if(physical.length){h.audit.erasedFirst=true;h.audit.movesAfterErase=0;}
 }
 if(arm.again&&s===arm.again&&h.audit.erasedFirst&&a.startsWith('MOVE_')&&!h.audit.sentAgain)h.audit.movesAfterErase++;
 return r;
}
function filterNewestWarning(i,s){const x=JSON.parse(JSON.stringify(i));const relevant=x.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===s);
 if(relevant.length>1){const newest=relevant.at(-1);x.durable=x.durable.filter(e=>e!==newest);}return x;}
function naiveRollout(w,arm){const h=prepare(w,arm);let steps=0,wrong=0,secondWarnings=0;
 for(;steps<350&&!h.done;steps++){
  const x=inject(h,arm);if(x==='again')secondWarnings++;
  const i=observation(h),effective=arm.again?filterNewestWarning(i,arm.again):i,a=action(effective);
  ok(i.legal.includes(a),'naive illegal');const successBefore=h.success;apply(h,a,arm);if(h.protected_damage&&!successBefore)wrong++;
 }
 ok(h.done,'naive rollout exceeded budget');return {success:!!h.success&&!h.protected_damage,damage:wrong,steps,secondWarnings};
}
function validate(dir,split,mf,geomSeen,inputSeen){const records=fs.readFileSync(path.join(dir,split+'_policy.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 const worlds=fs.readFileSync(path.join(dir,split+'_challenge_EVALUATOR_ONLY.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 ok(worlds.length===mf.splits[split].worlds&&records.length===mf.splits[split].policy_decisions,'manifest counts');
 let pos=0,done=0,second=0,probes=0,duplicates=0,naiveFailures=0,naiveSuccesses=0;
 for(let ix=0;ix<worlds.length;ix++){
  const w=v46.candidate(split,500+ix).w,m=worlds[ix],old=geomSeen.get(w.geometry_sha256);
  ok(m.geometry_sha256===w.geometry_sha256,'challenge geometry');ok(!old||old===split,'cross split geometry');geomSeen.set(w.geometry_sha256,split);
  const probesByArm=[];
  for(let armIndex=0;armIndex<v52.ARMS.length;armIndex++){
   const arm=v52.ARMS[armIndex],h=prepare(w,arm);ok(m.arms[armIndex]===arm.id,'arm id mismatch');let alerts=0,nprobes=0,paints=0;
   for(let st=0;st<m.episode_steps[armIndex];st++){
    const msg=inject(h,arm);if(msg){alerts++;if(msg==='again')second++;}
    const i=observation(h),row=records[pos++];ok(row,'missing model record');
    ok(same(row.input,i),'observation mismatched '+split+'/'+ix+'/'+arm.id+'/'+st);
    ok(row.action===action(i),'independently reconstructed teacher mismatch '+split+'/'+ix+'/'+arm.id+'/'+st);
    ok(i.legal.includes(row.action),'teacher illegal');
    ok(!/"(?:laws|sentAgain|sentFirst|againFlip|flipA|flipB|protected_damage|arm|future_material)"/.test(JSON.stringify(i)),'privileged metadata leaked');
    const hash=H(i),prior=inputSeen.get(hash);ok(!prior||prior.split===split,'identical sample leaked across splits');
    ok(!prior||prior.action===row.action,'contradictory sample label');
    if(split==='TRAIN'&&prior){ok(row.weight===0,'duplicate TRAIN not masked');duplicates++;}
    if(!prior)inputSeen.set(hash,{split,action:row.action});
    const stage=surface(h),at=h.pos;const r=apply(h,row.action,arm);
    if(row.action.startsWith('PAINT_')){if(at===target(h,stage)){ok(r.material===stage,'protected target incorrect');paints++;}else{nprobes++;probes++;ok(row.action==='PAINT_A','safe probe command wrong');}}
   }
   ok(h.done&&h.success&&!h.protected_damage,'terminal not successful');
   ok(paints===2&&nprobes===m.active_probe_paints[armIndex],'probe/target count');
   ok(alerts===Number(!!arm.alertA)+Number(!!arm.alertB)+Number(!!arm.again),'notice count');
   done++;probesByArm.push(nprobes);
   const negative=naiveRollout(w,arm);if(negative.success)naiveSuccesses++;else{naiveFailures++;ok(arm.againFlip,'naive failed without hidden flip');}
  }
  ok(probesByArm[0]===0&&probesByArm[1]===1&&probesByArm[2]===2,'control probe count');
 }
 ok(pos===records.length,'orphaned policy rows');
 ok(done===mf.splits[split].successful_host_episodes&&probes===mf.splits[split].accepted_safe_probe_paints,'per-split totals');
 ok(second===mf.splits[split].repeated_alerts_while_returning,'per-split repeat count');
 for(const [file,info] of Object.entries(mf.shards))if(file.startsWith(split+'_')){const b=fs.readFileSync(path.join(dir,file));
  ok(info.bytes===b.length&&info.sha256===H(b)&&info.rows===(file.includes('challenge')?worlds.length:records.length),'shard integrity');}
 return {worlds:worlds.length,episodes:done,policy_rows:pos,accepted_safe_probes:probes,repeated_warnings:second,
  train_duplicates_zero_weight:duplicates,naive_stale_alert_success:naiveSuccesses,naive_stale_alert_failure:naiveFailures};
}
function main(){const dir=path.resolve(process.argv[2]),mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 ok(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'realert_v52.js'))),'generator hash');
 for(const p of mf.parents)ok(p.sha256===H(fs.readFileSync(path.join(__dirname,p.file))),'parent pin '+p.file);
 const stats={},geometry=new Map(),inputs=new Map();const requested=process.argv[3];const splits=requested?[requested]:['TRAIN','DEV','HELDOUT'];ok(splits.every(s=>['TRAIN','DEV','HELDOUT'].includes(s)),'unknown split');for(const split of splits)stats[split]=validate(dir,split,mf,geometry,inputs);
 console.log(JSON.stringify({status:'PASS independent host replay, teacher labels, leakage, physical receipts, controls and hashes',stats},null,2));
}
if(require.main===module)main();
module.exports={prepare,inject,observation,action,apply,naiveRollout,validate};
