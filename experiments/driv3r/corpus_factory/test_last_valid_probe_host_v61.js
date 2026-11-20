#!/usr/bin/env node
'use strict';
/* Independent v61 executor: does not import the v61 generator or rely on
 * its hidden-state certificates to accept action/outcome labels. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js');
const v58=require('./offpolicy_repair_v58.js'),v46=require('./mission_memory_v46.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const ok=(v,s)=>{if(!v)throw Error('V61 AUDIT '+s)};
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
const cache=new Map();
function world(s,ix){const k=s+':'+ix;if(!cache.has(k))cache.set(k,v46.candidate(s,700+ix).w);return cache.get(k);}
function replay(split,cert,episode){
 const w=world(split,cert.source_index),arm=v53.ARMS[cert.old_swapped?2:1],pars=v53.armParams(arm);
 ok(w.seed===cert.seed&&w.geometry_sha256===cert.geometry_sha256,'geometry/provenance mismatch');
 const h=v53.prep(w,arm);let prior=[],probes=[],alerts=0,critical=null,forced=0;
 for(let t=0;t<episode.steps.length;t++){
  const st=episode.steps[t],n=v52.notice(h,pars);v53.injectB(h,arm,n,t);
  if(t===cert.second_alert_step){
   const old=h.events.filter(e=>e.kind==='PAINT_RESULT'&&e.surface==='B'&&e.at>h.v53.futureB.at&&e.site.join(',')===h.g.coords[h.siteB].join(','));
   ok(old.length===1&&h.painted[h.siteB]===undefined&&h.stage===1&&h.pos!==h.siteB,'second alert not after erased physical first probe');
   h.laws.B=cert.new_swapped?'SWAPPED':'NORMAL';
   h.events.push({kind:'SURFACE_ALERT',accepted:true,at:h.accepted,surface:'B',site:h.g.coords[h.siteB]});alerts++;
  }
  const obs=v53.observe(h),input=v58.streamify(obs,prior);
  ok(eq(input,st.input),'incorrect exact host observation '+split+':'+cert.episode+':'+t);
  ok(obs.legal.includes(st.action),'illegal action '+split+':'+cert.episode+':'+t);
  ok(!('durable' in input)&&!('laws' in input)&&!('old_swapped' in input)&&!('new_swapped' in input),'hidden host law exposed');
  const oldStage=h.stage,oldPos=h.pos;
  if(t>=cert.critical_step-cert.forced_detour_actions&&t<cert.critical_step){
   ok(st.weight===0&&st.focus_loss_weight===0&&st.action.startsWith('MOVE_'),'detour teaching forbidden');forced++;
  }
  if(t===cert.critical_step){
   ok(st.weight===1&&st.focus_loss_weight===8&&oldStage===1&&oldPos===w.goal2,'critical supervised target absent');
   ok(probes.length===2,'critical with fewer than two probes');
   ok(!input.events_delta.some(e=>e[1]==='PAINT_RESULT'&&e[4]==='B'),'receipt leaked into current event delta');
   const negative=structuredClone(h),opposite=st.action==='PAINT_A'?'PAINT_B':'PAINT_A';
   ok(obs.legal.includes(opposite),'negative command illegal');
   const wrong=v52.command(negative,opposite,pars);
   ok(wrong.material==='A'&&negative.stage===1&&!negative.success,'wrong command did not fail physical goal');
   critical={input,action:st.action,age:(t+1)-probes.at(-1).at,old:probes[0].material,
    latest:probes[1].material};
  }
  prior=obs.durable;
  const r=v52.command(h,st.action,pars);
  if(oldStage===1&&oldPos===h.siteB&&st.action==='PAINT_A'){
   ok(r.material==='A'||r.material==='B','unphysical probe result');
   probes.push({at:t+1,material:r.material});
   if(probes.length===1)ok(t+1===cert.first_probe_step,'first probe age wrong');
   if(probes.length===2)ok(t+1===cert.second_probe_step,'second probe age wrong');
  }
  if(oldStage===1&&oldPos===w.goal2&&st.action.startsWith('PAINT_')){
   ok(r.material==='B'&&h.stage===2,'goal paint did not physically complete B');
  }
 }
 ok(h.done&&h.success&&!h.protected_damage,'host terminal failure');
 ok(alerts===1&&probes.length===2&&critical&&forced===cert.forced_detour_actions,'missing two-probe challenge');
 const bAlerts=h.events.filter(e=>e.kind==='SURFACE_ALERT'&&e.surface==='B');
 ok(bAlerts.length===2,'not exactly two B warnings');
 ok(probes[0].material===cert.first_material&&probes[1].material===cert.last_material,'physical outcome certificate mismatch');
 ok(critical.action===cert.final_action&&critical.age===cert.calibration_age,'critical certificate mismatch');
 ok(critical.age>=cert.forced_detour_actions,'latest physical memory not aged');
 ok((critical.latest==='A'?'PAINT_B':'PAINT_A')===critical.action,'latest physical evidence not used');
 ok(H(critical.input)===cert.paired_current_input_sha256,'current snapshot SHA mismatch');
 return critical;
}
function audit(dir){
 const raw=fs.readFileSync(path.join(dir,'manifest.json')),manifest=JSON.parse(raw);
 ok(manifest.schema==='driv3r/corpus_factory/last-valid-physical-probe-v61/1','manifest schema');
 for(const [name,m] of Object.entries(manifest.files)){
  const b=fs.readFileSync(path.join(dir,name));ok(H(b)===m.sha256&&b.length===m.bytes,'data shard SHA/size mismatch '+name);
 }
 const geometry=new Map();let episodes=0,quartets=0,decisions=0,wrongStale=0;
 for(const s of ['TRAIN','DEV','HELDOUT']){
  const name=s+(s==='TRAIN'?'_last_probe_stream.jsonl':'_EVALUATOR_ONLY_last_probe_stream.jsonl');
  const data=read(path.join(dir,name)),certs=read(path.join(dir,s+'_last_probe_certificates_EVALUATOR_ONLY.jsonl'));
  ok(data.length===certs.length&&data.length===manifest.splits[s].episodes&&data.length%4===0,'split episode count');
  let localDecisions=0;
  for(let i=0;i<certs.length;i+=4){
   const group=certs.slice(i,i+4),expected=[[false,false],[false,true],[true,false],[true,true]];
   ok(group.every(c=>c.pair_id===group[0].pair_id),'quartet pairing');
   const res=[];
   for(let k=0;k<4;k++){
    const c=group[k];ok(c.episode===i+k&&c.split===s,'episode order');
    ok(c.old_swapped===expected[k][0]&&c.new_swapped===expected[k][1],'not balanced OLD x NEW design');
    const g=geometry.get(c.geometry_sha256);ok(!g||g===s,'cross-split geometry leakage');geometry.set(c.geometry_sha256,s);
    res.push(replay(s,c,data[i+k]));localDecisions+=data[i+k].steps.length;
   }
   ok(res.every(x=>eq(x.input,res[0].input)),'quartet instantaneous observations differ');
   ok(res[0].action===res[2].action&&res[1].action===res[3].action,'old law changed correct command');
   ok(res[0].action!==res[1].action,'latest law not discriminative');
   ok(res[0].old===res[1].old&&res[2].old===res[3].old,'first probe not controlled');
   ok(res[0].latest===res[2].latest&&res[1].latest===res[3].latest,'last probe not controlled');
   wrongStale+=res.filter(x=>x.old!==x.latest).length;quartets++;
  }
  ok(localDecisions===manifest.splits[s].decisions,'decision total mismatch '+s);
  decisions+=localDecisions;episodes+=data.length;
 }
 ok(decisions===manifest.total_decisions&&quartets===manifest.total_quartets&&geometry.size===manifest.distinct_geometries,'global totals');
 ok(wrongStale===episodes/2,'stale-probe negative balance not 50%');
 const result={pass:true,synthetic_host_only:true,independent_full_host_replay:true,episodes,quartets,decisions,
  world_geometries:geometry.size,misleading_first_calibrations:wrongStale,
  memoryless_critical_accuracy_ceiling:0.5,stale_first_probe_only_accuracy_ceiling:0.5,
  latest_receipt_symbolic_accuracy:1.0};
 console.log(JSON.stringify(result));return result;
}
if(require.main===module){ok(process.argv[2],'usage: node test_last_valid_probe_host_v61.js CORPUS_DIR');audit(process.argv[2]);}
module.exports={audit};
