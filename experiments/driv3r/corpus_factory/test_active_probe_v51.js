'use strict';
/* Independent observable-label and host replay audit, no policy training. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v51=require('./active_probe_v51.js'),v50=require('./online_recovery_v50.js');
const v49=require('./contextual_probe_v49.js'),v46=require('./mission_memory_v46.js'),v45=require('./frontier_memory_v45.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const ok=(v,m)=>{if(!v)throw Error('INDEPENDENT_V51_AUDIT '+m)};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function independentLabel(i){
 if(i.durable.some(e=>e[1]==='MISSION_COMPLETE'))return 'STOP';
 const missions=i.durable.filter(e=>e[1]==='MISSION_ASSIGN'||e[1]==='GOAL_REVISION');ok(missions.length,'no mission');
 const m=missions.at(-1),goal=m[2],want=m[3];
 const alerts=i.durable.filter(e=>e[1]==='SURFACE_ALERT'&&e[2]===want);
 if(alerts.length){
  const a=alerts.at(-1),site=a[3];
  const measured=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===want&&e[0]>a[0]&&e[2]==='PAINT_A'&&same(e[5],site));
  const residue=i.painted.some(([p])=>same(p,site));
  if(!measured.length){if(!same(i.p,site))return v45.pathOnSurvey(i,site);return residue?'ERASE':'PAINT_A';}
  if(residue&&same(i.p,site))return 'ERASE';
 }
 if(!same(i.p,goal))return v45.pathOnSurvey(i,goal);
 if(i.painted.some(([p])=>same(p,goal)))return 'ERASE';
 const q=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===i.surface);ok(q.length>=2,'missing physical calibration');
 const last=q.at(-1),issued=last[2].slice(-1),physical=last[3];
 ok(['A','B'].includes(physical)&&['A','B'].includes(issued),'invalid probe outcome');
 return 'PAINT_'+(issued===physical?want:(want==='A'?'B':'A'));
}
function independentAlert(h,arm){const surf=h.stage===0?'A':h.stage===1?'B':null;
 if(!surf||h.pos!==(surf==='A'?h.w.goal:h.w.goal2)||h.painted[h.pos]!==undefined)return;
 if(!((surf==='A')?arm.alertA:arm.alertB))return;
 if(h.auditAlerted?.[surf])return;
 if(!h.auditAlerted)h.auditAlerted={};h.auditAlerted[surf]=true;
 if((surf==='A')?arm.flipA:arm.flipB)h.laws[surf]=h.laws[surf]==='NORMAL'?'SWAPPED':'NORMAL';
 const site=surf==='A'?h.w.s:h.siteB;
 h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:surf,site:h.g.coords[site]});
}
function independentObservation(h){const x=v50.observe(h);x.v='synthetic-v51-active-safe-probe';
 x.durable.push(...h.events.filter(e=>e.kind==='SURFACE_ALERT').map(e=>[e.at,'SURFACE_ALERT',e.surface,e.site]));
 x.durable.sort((a,b)=>a[0]-b[0]||String(a[1]).localeCompare(String(b[1])));return x;
}
function independentCommand(h,a){const stage=h.stage,at=h.pos,prev=h.accepted;
 const r=v49.command(h,a);ok(h.accepted===prev+1,'host rejected action');
 if(a.startsWith('PAINT_'))h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:a,material:r.material,surface:v49.label(h),site:h.g.coords[at]});
 const surf=stage===0?'A':stage===1?'B':null;
 if(surf&&h.auditAlerted?.[surf]&&a.startsWith('PAINT_')&&at===(surf==='A'?h.w.goal:h.w.goal2)&&r.material!==surf){
  h.done=true;h.success=false;h.protected_damage=true;
 }
}
function auditSplit(dir,split,mf){const pol=fs.readFileSync(path.join(dir,split+'_policy.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 const chal=fs.readFileSync(path.join(dir,split+'_challenge_EVALUATOR_ONLY.jsonl'),'utf8').trimEnd().split('\n').map(JSON.parse);
 ok(pol.length===mf.splits[split].policy_decisions,'policy record count');ok(chal.length===mf.splits[split].worlds,'world count');
 const hashes=new Map();let pos=0,terminals=0,probes=0,zero=0,protectedCorrect=0;
 for(let ix=0;ix<chal.length;ix++){
  const w=v46.candidate(split,300+ix).w,m=chal[ix];ok(w.geometry_sha256===m.geometry_sha256,'geometry hash');
  for(let a=0;a<v51.ARMS.length;a++){
   const arm=v51.ARMS[a],h=v51.prep(w,arm);ok(m.arms[a]===arm.id,'arm identity');let saw=0,goalPaints=0;
   for(let j=0;j<m.decisions[a];j++){
    independentAlert(h,arm);const expected=independentObservation(h),row=pol[pos++];ok(row,'missing row');
    ok(same(row.input,expected),'host observation mismatch '+split+'/'+ix+'/'+arm.id+'/'+j);
    ok(row.action===independentLabel(expected),'teacher label does not follow observation '+split+'/'+ix+'/'+arm.id+'/'+j);
    ok(expected.legal.includes(row.action),'illegal action');
    ok(!/"(?:laws|alerted|auditAlerted|flipA|flipB|protected_damage|arm|expected_material)"/.test(JSON.stringify(expected)),'privileged metadata leaked');
    const hk=H(expected),prior=hashes.get(hk);ok(!prior||prior===row.action,'same-input label contradiction');
    if(prior&&split==='TRAIN')ok(row.weight===0,'exact train repetition unmasked');
    if(!prior)hashes.set(hk,row.action);if(row.weight===0)zero++;
    const before=h.stage,at=h.pos;independentCommand(h,row.action);
    if(row.action.startsWith('PAINT_')){
     if((before===0&&at===h.w.goal)||(before===1&&at===h.w.goal2)){
      ok(h.painted[at]===(before===0?'A':'B'),'protected goal incorrect');goalPaints++;protectedCorrect++;
     }else{saw++;probes++;ok(row.action==='PAINT_A','safe probe command wrong');}
    }
   }
   ok(h.done&&h.success&&!h.protected_damage,'host terminal failure');
   ok(saw===m.active_probe_paints[a],'per-arm probe count');
   ok(goalPaints===m.protected_goal_correct_paints[a]&&goalPaints===2,'per-arm goal counts');
   terminals++;
  }
 }
 ok(pos===pol.length,'orphaned rows');
 ok(terminals===mf.splits[split].successful_synthetic_episodes,'terminal total');
 ok(probes===mf.splits[split].active_host_probe_paints,'probe total');
 ok(protectedCorrect===mf.splits[split].protected_correct_goal_paints,'correct physical paint count');
 for(const [file,d] of Object.entries(mf.shards)){if(!file.startsWith(split+'_'))continue;
  const bytes=fs.readFileSync(path.join(dir,file));ok(bytes.length===d.bytes&&H(bytes)===d.sha256,'shard hash '+file);}
 return {worlds:chal.length,episodes:terminals,rows:pos,active_paints:probes,protected_correct:protectedCorrect,zero_weight:zero};
}
function main(){const dir=path.resolve(process.argv[2]);const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 ok(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'active_probe_v51.js'))),'generator SHA');
 for(const p of mf.parents)ok(p.sha256===H(fs.readFileSync(path.join(__dirname,p.file))),'parent SHA '+p.file);
 const stats={};for(const split of ['TRAIN','DEV','HELDOUT'])stats[split]=auditSplit(dir,split,mf);
 console.log(JSON.stringify({status:'PASS independent observer-only labels, host actions, physical outcomes, source/hash and all split records',stats},null,2));}
if(require.main===module)main();
module.exports={independentLabel,independentObservation,independentCommand,auditSplit};