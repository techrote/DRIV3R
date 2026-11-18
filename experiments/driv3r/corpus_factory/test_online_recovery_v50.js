'use strict';
/* Independent data auditor for v50, not a model/policy trainer. No external APIs. */
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const v50=require('./online_recovery_v50.js'),v49=require('./contextual_probe_v49.js');
const v46=require('./mission_memory_v46.js'),v45=require('./frontier_memory_v45.js'),v48=require('./empirical_tool_v48.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const assert=(x,m)=>{if(!x)throw Error('v50 independent audit: '+m)};
const key=x=>JSON.stringify(x);
const splits=['TRAIN','DEV','HELDOUT'];
function independentlyLabel(i){
 const completed=i.durable.some(z=>z[1]==='MISSION_COMPLETE');if(completed)return 'STOP';
 const missions=i.durable.filter(z=>z[1]==='MISSION_ASSIGN'||z[1]==='GOAL_REVISION');
 assert(missions.length,'no mission history');const active=missions.at(-1),coord=active[2],want=active[3];
 if(key(i.p)!==key(coord))return v45.pathOnSurvey(i,coord);
 const actual=i.painted.find(([p])=>key(p)===key(coord));if(actual)return 'ERASE';
 const receipts=i.durable.filter(z=>z[1]==='PAINT_RESULT'&&z[4]===i.surface);
 assert(receipts.length>=2,'no contextual physical history');
 const latest=receipts.at(-1),issued=latest[2].at(-1),got=latest[3];
 assert(['A','B'].includes(issued)&&['A','B'].includes(got)&&['A','B'].includes(want),'invalid receipts');
 return 'PAINT_'+(issued===got?want:(want==='A'?'B':'A'));
}
function replayPhysical(h,arm,action){
 // Independent dispatcher; the parent adapter applies the exact same synthetic host.
 const before=h.accepted,coord=h.pos;
 if(action.startsWith('PAINT_'))h.tool_mode=h.laws[v49.label(h)];
 const res=v48.accepted(h,action);assert(h.accepted===before+1,'host did not accept action');
 if(action.startsWith('PAINT_'))h.events.push({kind:'PAINT_RESULT',at:h.accepted,accepted:true,issued:action,
   material:res.material,surface:v49.label(h),site:h.g.coords[coord]});
}
function driftPhysical(h,arm){
 const where=h.stage===0?h.w.goal:h.stage===1?h.w.goal2:null;
 const surf=h.stage===0?'A':h.stage===1?'B':null;
 if(where===null||h.pos!==where||h.painted[h.pos]!==undefined)return;
 const enabled=surf==='A'?arm.flipA:arm.flipB,field=surf==='A'?'auditFlipA':'auditFlipB';
 if(enabled&&!h[field]){h[field]=true;h.laws[surf]=h.laws[surf]==='NORMAL'?'SWAPPED':'NORMAL';}
}
function verifySplit(dir,split,mf){
 const raw=fs.readFileSync(path.join(dir,split+'_policy.jsonl'),'utf8');
 const data=raw.trimEnd().split('\n').map(s=>JSON.parse(s));
 const chal=fs.readFileSync(path.join(dir,split+'_challenge_EVALUATOR_ONLY.jsonl'),'utf8').trimEnd().split('\n').map(s=>JSON.parse(s));
 assert(data.length===mf.splits[split].policy_decisions,'policy count');
 assert(chal.length===mf.splits[split].worlds,'world count');
 let pos=0,done=0,wrong=0,erases=0,zero=0,paintRows=0,wrongInput=0;
 const fingerprints=new Map();
 for(let ix=0;ix<chal.length;ix++){
  const w=v46.candidate(split,200+ix).w,m=chal[ix];
  assert(m.geometry_sha256===w.geometry_sha256&&m.source_seed===w.seed,'world source identity');
  for(let k=0;k<m.arms.length;k++){
   const arm=v50.arms[k],h=v50.prep(w,arm),length=m.decision_counts[k];
   assert(arm.id===m.arms[k],'arm order mismatch');
   let bad=0,erase=0;
   for(let step=0;step<length;step++){
    driftPhysical(h,arm);
    const got=data[pos++];assert(got,'missing policy row');
    const independent=v49.observe(h);independent.v='synthetic-v50-online-contextual-recovery';
    assert(key(independent)===key(got.input),'saved policy input does not match independent host state at '+split+'/'+ix+'/'+k+'/'+step);
    const expected=independentlyLabel(got.input);
    assert(got.action===expected,'oracle label not independently reproducible: '+got.action+'/'+expected);
    assert(got.input.legal.includes(got.action),'illegal action');
    const forbidden=key(got.input);assert(!/"(?:flipA|flipB|didFlipA|didFlipB|laws|drift_flags_EVALUATOR_ONLY|expected|arm)"/.test(forbidden),'private drift metadata exposed');
    assert(!/SWAPPED|NORMAL|TOOL_LAW/.test(forbidden),'privileged tool law exposed');
    const f=H(got.input),prev=fingerprints.get(f);assert(!prev||prev.action===got.action,'conflicting label');
    if(prev&&split==='TRAIN')assert(got.weight===0,'unmasked repeated TRAIN input');
    if(!prev)fingerprints.set(f,{action:got.action});
    if(got.weight===0)zero++;
    const before=h.stage,at=h.pos;
    replayPhysical(h,arm,got.action);
    if(got.action.startsWith('PAINT_')){
      paintRows++;const want=before===0?'A':'B';assert(at===(before===0?h.w.goal:h.w.goal2),'paint happened outside active target');
      const p=h.painted[at];if(p!==want){bad++;wrong++;}
    }
    if(got.action==='ERASE'&&before<2&&at===(before===0?h.w.goal:h.w.goal2))erase++;
   }
   assert(h.done&&h.success,'host terminal not successful');
   assert(bad===m.unexpected_wrong_paint[k],'wrong physical paint count changed');
   assert(erase>=m.recovery_erases[k],'erasure count below drift correction');
   erases+=m.recovery_erases[k];done++;
  }
 }
 assert(pos===data.length,'orphaned policy records');
 for(const [name,desc]of Object.entries(mf.shards))if(name.startsWith(split+'_')){
   const bytes=fs.readFileSync(path.join(dir,name));assert(bytes.length===desc.bytes&&H(bytes)===desc.sha256,'shard hash mismatch '+name);
 }
 assert(done===mf.splits[split].synthetic_terminal_episodes,'terminal episode mismatch');
 assert(wrong===mf.splits[split].wrong_physical_paints,'incorrect wrong-paint count');
 assert(erases===mf.splits[split].host_erases_after_wrong_paint,'incorrect recovery erase count');
 return {worlds:chal.length,episodes:done,policy_rows:pos,material_paint_rows:paintRows,wrong_physical_paints:wrong,drift_recovery_erases:erases,zero_weight_rows:zero};
}
function main(){const dir=path.resolve(process.argv[2]||'');assert(fs.existsSync(dir),'pass generated corpus dir');
 const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
 assert(mf.generator_sha256===H(fs.readFileSync(path.join(__dirname,'online_recovery_v50.js'))),'generator hash');
 for(const p of mf.parents)assert(p.sha256===H(fs.readFileSync(path.join(__dirname,p.file))),'parent hash '+p.file);
 let total=0;const results={};for(const split of splits){results[split]=verifySplit(dir,split,mf);total+=results[split].policy_rows;}
 assert(total===mf.total_policy_decisions,'total rows');
 console.log(JSON.stringify({status:'PASS independent full host replay and reference observable-only labels',total,results},null,2));
}
if(require.main===module)main();
module.exports={independentlyLabel,verifySplit};
