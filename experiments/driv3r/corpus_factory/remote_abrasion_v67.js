#!/usr/bin/env node
'use strict';
/* DRIV3R Corpus Factory v67: physical remote paint erosion and cost-sensitive
 * reinspection. Separate synthetic v44 host extension, NOT native CyberSand.
 * DATA ONLY: no policy or model training. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const assert=(v,s)=>{if(!v)throw Error('V67 '+s)};
const splits={TRAIN:12,DEV:6,HELDOUT:6};
const seeds={TRAIN:84000,DEV:94000,HELDOUT:104000};
const sequences=[[],['A'],['B'],['A','A'],['B','B'],['A','B'],['B','A'],['A','B','A'],['B','A','B'],['A','B','B']];
function shaFile(p){return H(fs.readFileSync(p))}
function cp(x){return JSON.parse(JSON.stringify(x))}
function d(g,a,b){return host.dist(g,b,-1)[a]}
function pathTo(h,t){const out=[];let p=h.pos,limit=0;while(p!==t){const ds=host.dist(h.g,t,-1);const next=h.g.adj[p].filter(([n])=>ds[n]===ds[p]-1).sort((a,b)=>a[0]-b[0])[0];assert(next,'no path');const [x,y]=h.g.coords[p], [xx,yy]=h.g.coords[next[0]];const dir=xx>x?'E':xx<x?'W':yy>y?'S':'N';out.push('MOVE_'+dir);p=next[0];assert(++limit<h.g.coords.length,'route loop');}return out}
function selectWorld(split,j,exclude){for(let trial=0;trial<75;trial++){
 const index=seeds[split]+j*171+trial*3;let w;try{w=host.scenario(split,index);}catch{continue}
 const g=host.hydrate(w),h=host.start(w,'positive',false);
 if(d(g,w.s,w.goal)<5||d(g,w.s,w.goal2)<5||pathTo(h,w.goal)[0]===pathTo(h,w.goal2)[0]||exclude.has(w.geometry_sha256))continue;
 return {w,source_index:index,trial};
 }throw Error('V67 no distinct divergent-goal world '+split+'/'+j)}
function begin(w){const h=host.start(w,'positive',false);assert(h.pos!==w.goal&&h.pos!==w.goal2,'start at goal');return h}
function makeSense(h,pending){const obs=host.input(h),[sx,sy]=h.g.coords[h.pos];
 // Prevent privileged global paint-table visibility: only nearby painted cells
 // (Manhattan radius 1) appear; other cells must be inferred/reinspected.
 obs.painted=obs.painted.filter(([i])=>{const [x,y]=h.g.coords[i];return Math.abs(x-sx)+Math.abs(y-sy)<=1});
 obs.v='synthetic-v67-local-paint';obs.events_delta=pending.splice(0);return obs}
function rec(ctx,action,weight){const inp=makeSense(ctx.h,ctx.pending);assert(inp.legal.includes(action),'illegal '+action);ctx.steps.push({input:inp,action,weight,focus_loss_weight:weight});assert(host.apply(ctx.h,action).accepted,'rejected '+action);}
function route(ctx,t,weight){for(const a of pathTo(ctx.h,t))rec(ctx,a,weight)}
function shock(ctx,mat,source){const t=source==='COUPON'?ctx.h.pos:mat==='A'?ctx.h.w.goal:ctx.h.w.goal2;let removed=false;
 if(source==='COUPON') {assert(ctx.h.painted[t]==='A','probe not painted');if(ctx.threshold===1){delete ctx.h.painted[t];removed=true;}}
 else {ctx.count[mat]++;if(ctx.count[mat]===ctx.threshold&&ctx.h.painted[t]===mat){delete ctx.h.painted[t];removed=true;}}
 ctx.pending.push([ctx.h.accepted,source==='COUPON'?'LOCAL_ABRASION_TEST':'DISTANT_ABRASION',source==='COUPON'?'A':mat]);return removed}
function continuation(ctx,broken,{record=true,always=false}={}){const goals=always?['A','B']:broken;const w=ctx.h.w;
 if(!goals.length){rec(ctx,'STOP',1);return}
 const targets=goals.map(m=>m==='A'?w.goal:w.goal2),orders=goals.length===2?[[0,1],[1,0]]:[[0]];
 const best=orders.map(order=>({order,cost:order.reduce((a,v,k)=>a+d(ctx.h.g,k?targets[order[k-1]]:ctx.h.pos,targets[v]),0)})).sort((a,b)=>a.cost-b.cost||a.order.join().localeCompare(b.order.join()))[0];
 for(const i of best.order){const mat=goals[i],target=targets[i];route(ctx,target,1);if(ctx.h.painted[target]!==mat){assert(ctx.h.painted[target]===undefined,'unexpected foreign paint');rec(ctx,'PAINT_'+mat,1);}}
 rec(ctx,'STOP',1);
}
function episode(w,threshold,sequence){const h=begin(w),ctx={h,threshold,sequence,pending:[],steps:[],count:{A:0,B:0}};
 // Actual host PAINT at a sacrificial local test cell, physically abrade it,
 // observe locally, then use a declared synthetic test-fixture cleanup to restore the sample after moving away.
 rec(ctx,'PAINT_A',0);const removed=shock(ctx,'A','COUPON');
 const n=h.g.adj[h.pos][0][0],step=pathTo(h,n)[0];rec(ctx,step,0);
 delete h.painted[w.s];ctx.pending.push([h.accepted,'COUPON_RESET','A']);
 const reverse={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E',MOVE_E:'MOVE_W'}[step];rec(ctx,reverse,0);
 assert(h.painted[w.s]===undefined&&h.pos===w.s,'unclean coupon');
 route(ctx,w.goal,0);rec(ctx,'PAINT_A',0);route(ctx,w.goal2,0);rec(ctx,'PAINT_B',0);assert(h.stage===2,'mission painting incomplete');
 route(ctx,w.s,0);assert(d(h.g,h.pos,w.goal)>=5&&d(h.g,h.pos,w.goal2)>=5,'nonremote disturbance');
 const neighbor=h.g.adj[h.pos][0][0],action=pathTo(h,neighbor)[0],back={MOVE_N:'MOVE_S',MOVE_S:'MOVE_N',MOVE_W:'MOVE_E',MOVE_E:'MOVE_W'}[action];
 for(let k=0;k<104;k++){
  if([8,21,34].includes(k)){const idx=[8,21,34].indexOf(k),mat=sequence[idx];if(mat)shock(ctx,mat,'REMOTE');else ctx.pending.push([h.accepted,'BACKGROUND_VIBRATION','NONE']);}
  rec(ctx,k%2===0?action:back,0);
 }
 assert(h.pos===w.s&&ctx.pending.length===0,'not ready');
 const critical=ctx.steps.length,criticalInput=makeSense(h,[]),broken=['A','B'].filter(m=>h.painted[m==='A'?w.goal:w.goal2]!==m);
 // Only use context before first supervised action; remote physical truth
 // is never present in model input or in its local paint view.
 const snapshot=cp(h);continuation(ctx,broken);assert(h.done&&h.success,'teacher terminal failure');
 const teacher=ctx.steps.slice(critical),first=teacher[0].action;
 const control=(always)=>{const c={h:cp(snapshot),pending:[],steps:[]};continuation(c,always?['A','B']:[],{always});return {success:c.h.success,steps:c.steps.length}};
 const stop=control(false),revisit=control(true);
 assert(revisit.success,'always revisit failed');assert(stop.success===(broken.length===0),'stop control invalid');
 assert(H(ctx.steps[critical].input)===H(criticalInput),'critical input mismatch');
 return {episode:{steps:ctx.steps},certificate:{threshold,sequence,broken,first_action:first,critical_step:critical,critical_input_sha256:H(criticalInput),teacher_actions:teacher.length,stop_only_success:stop.success,always_revisit_success:revisit.success,always_revisit_actions:revisit.steps,always_revisit_overhead:revisit.steps-teacher.length,host_terminal_success:h.success,physical_probe_eroded:removed,source:'synthetic-v44-plus-v67-physical-erasure-extension',sample_mat:'A',remote_shocks:sequence.length}};
}
function fname(s,type){return s+(type==='stream'?(s==='TRAIN'?'_remote_abrasion_stream.jsonl':'_EVALUATOR_ONLY_remote_abrasion_stream.jsonl'):'_remote_abrasion_certificates_EVALUATOR_ONLY.jsonl')}
function writeRows(dir,name,rows,mf){const b=Buffer.from(rows.map(JSON.stringify).join('\n')+'\n');fs.writeFileSync(path.join(dir,name),b,{flag:'wx'});mf.files[name]={sha256:H(b),rows:rows.length,bytes:b.length};}
function generate(out,{quick=false}={}){assert(!fs.existsSync(out),'destination exists');fs.mkdirSync(out,{recursive:true});const mf={schema:'driv3r/corpus_factory/remote-abrasion-v67/1',date:'2026-11-19',synthetic_not_native_CyberSand:true,generator_sha256:shaFile(__filename),host_sha256:shaFile(path.join(__dirname,'unseen_topology_v44.js')),quick,
 scenario:'Physical remote erasure of formerly completed painted goals after local abrasion calibration, material-specific shock history and long distractor detours. Agent must trade pointless reinspection against irreversible STOP failure.',
 observation_contract:'Local painted radius 1 only, unmodified legal actions/geometry/stage, incremental physical abrasion events and ordinary host movement/painting receipts. No distant paint truth, threshold, counts or certificate fields in input. Event names are invented synthetic symbols, NOT native CyberSand ABI.',
 supervision:'Forced calibration/pre-goal/remote-detour steps weight 0. Only post-intervention critical continuation and necessary reinspection/repainting weight 1. DEV, HELDOUT and every certificate evaluator-only. Same-sequence 2-threshold arms share topology, goals and remote shock events; current critical observations identical within each world.',
 provenance:'Novel disjoint v44 world indices by split; exclude any supplied older geometry hashes when generating. All physical paint/erase/movement via v44 host.apply; erosion physically removes host painted state via explicit v67 external intervention.',
 known_limitations:'Synthetic host extension (not native CyberSand), full graph known, teacher-forced visits, deterministic 1/2 abrasion threshold can be solved by symbolic event counter; no neural training, model or GPU. Some paths may share prefixes. Always-reinspect succeeds but wastes travel.',files:{},splits:{}};
 let sum=0,tot=0,overhead=0,never=0,groups=0;const seen=new Map(),prior=new Set();
 for(const split of Object.keys(splits)){
  const count=quick?Math.min(2,splits[split]):splits[split],streams=[],certs=[],seenInput=new Map();let decisionCount=0,localOverhead=0,stopSuccess=0,maskedDuplicate=0;const summaries=[];
  for(let j=0;j<count;j++){
   const {w,source_index,trial}=selectWorld(split,j,prior);assert(!seen.has(w.geometry_sha256),'geometry duplicate');seen.set(w.geometry_sha256,split);prior.add(w.geometry_sha256);groups++;
   const samples=[];
   for(const threshold of [1,2])for(const sequence of sequences){const {episode:ep,certificate:c}=episode(w,threshold,sequence);
    for(let i=0;i<ep.steps.length;i++)if(split==='TRAIN'&&i!==c.critical_step&&ep.steps[i].weight===1){const sig=H([ep.steps[i].input,ep.steps[i].action]);if(seenInput.has(sig)){ep.steps[i].weight=0;ep.steps[i].focus_loss_weight=0;maskedDuplicate++;}else seenInput.set(sig,true);}
    // Never mask the critical decision even if it repeats an earlier row.
    assert(ep.steps[c.critical_step].weight===1,'critical label masked');
    c.split=split;c.episode=streams.length;c.world_index=j;c.source_world_index=source_index;c.scenario_trial=trial;c.geometry_sha256=w.geometry_sha256;
    c.topology_seed=w.seed;c.world_goal=w.goal;c.world_goal2=w.goal2;c.start=w.s;
    c.episode_input_action_sha256=H(ep.steps.map(x=>[x.input,x.action,x.weight]));
    streams.push(ep);certs.push(c);samples.push(c);decisionCount+=ep.steps.length;localOverhead+=c.always_revisit_overhead;stopSuccess+=Number(c.stop_only_success);
   }
   assert(samples.every(c=>c.critical_input_sha256===samples[0].critical_input_sha256),'current-state leak across 20 arms');
   // Distinct correct first actions must exist: STOP, A-route, B-route.
   assert(new Set(samples.map(c=>c.first_action)).size===3,'weak critical action variety');
   summaries.push({world:j,geometry_sha256:w.geometry_sha256,source_world_index:source_index});
  }
  writeRows(out,fname(split,'stream'),streams,mf);writeRows(out,fname(split,'cert'),certs,mf);
  mf.splits[split]={worlds:count,episodes:streams.length,decisions:decisionCount,host_terminal_successes:streams.length,critical_decisions:streams.length,stop_only_successes:stopSuccess,always_revisit_successes:streams.length,mean_always_revisit_extra_steps:localOverhead/streams.length,masked_duplicate_train_followups:maskedDuplicate,world_geometry_sha256:summaries};
  sum+=decisionCount;tot+=streams.length;never+=stopSuccess;overhead+=localOverhead;
 }
 mf.total_episodes=tot;mf.total_decisions=sum;mf.worlds=groups;mf.stop_only_accuracy=never/tot;mf.always_revisit_success_rate=1;mf.mean_extra_steps_always_revisit=overhead/tot;
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(mf,null,2)+'\n',{flag:'wx'});return mf;
}
if(require.main===module){assert(process.argv[2],'usage: node remote_abrasion_v67.js NEW_DIR [--quick]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),{quick:process.argv.includes('--quick')}).splits,null,2))}
module.exports={generate,episode,selectWorld,makeSense,shock,begin,rec,route,continuation,fname,shaFile,cp,sequences,splits,H};
