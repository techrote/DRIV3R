#!/usr/bin/env node
'use strict';
/* Independent v67 validation: reconstruct host states from actions and physical
 * intervening events, check observations, legal moves, terminal state, and
 * current-state counterfactual integrity. No trainer or learned-policy claims. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const host=require('./unseen_topology_v44.js');
const assert=(x,m)=>{if(!x)throw Error('V67 AUDIT '+m)};
const H=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const read=p=>fs.readFileSync(p,'utf8').trim().split('\n').map(JSON.parse);
const files={TRAIN:'TRAIN_remote_abrasion_stream.jsonl',DEV:'DEV_EVALUATOR_ONLY_remote_abrasion_stream.jsonl',HELDOUT:'HELDOUT_EVALUATOR_ONLY_remote_abrasion_stream.jsonl'};
function audit(dir){const mf=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));assert(mf.schema==='driv3r/corpus_factory/remote-abrasion-v67/1'&&mf.synthetic_not_native_CyberSand,'schema');
 for(const [f,m] of Object.entries(mf.files)){const b=fs.readFileSync(path.join(dir,f));assert(H(b)===m.sha256&&b.length===m.bytes,'manifest hash '+f)}
 const seen=new Map();let decisions=0,success=0,firstActions=new Map(),stop=0,revisit=0,overhead=0;
 for(const split of ['TRAIN','DEV','HELDOUT']){
  const rows=read(path.join(dir,files[split])),certs=read(path.join(dir,split+'_remote_abrasion_certificates_EVALUATOR_ONLY.jsonl'));
  assert(rows.length===certs.length&&rows.length%20===0,'split size '+split);
  for(let i=0;i<rows.length;i++){
   const c=certs[i],steps=rows[i].steps;assert(c.split===split&&c.episode===i,'cert identity');
   const w=host.scenario(split,c.source_world_index);assert(w.geometry_sha256===c.geometry_sha256&&w.s===c.start&&w.goal===c.world_goal&&w.goal2===c.world_goal2,'world provenance');
   if(!seen.has(c.geometry_sha256))seen.set(c.geometry_sha256,split);assert(seen.get(c.geometry_sha256)===split,'split geometry overlap');
   const h=host.start(w,'positive',false),count={A:0,B:0};let physicalErasures=0,critical=null,probe=0,shockCount=0,lastRemoteAt=-1;
   for(let k=0;k<steps.length;k++){
    const r=steps[k],inp=r.input;assert(Array.isArray(inp.events_delta),'missing incremental events');
    for(const event of inp.events_delta){assert(event[0]===h.accepted,'event acceptance timestamp');
     if(event[1]==='LOCAL_ABRASION_TEST'){
      assert(probe++===0&&event[2]==='A'&&h.painted[w.s]==='A','probe event bad');
      if(c.threshold===1){delete h.painted[w.s];physicalErasures++;}
     }else if(event[1]==='COUPON_RESET'){
      assert(event[2]==='A','bad coupon reset');delete h.painted[w.s];
     }else if(event[1]==='DISTANT_ABRASION'){
      assert(shockCount<3&&event[2]===c.sequence[shockCount],'remote shock mismatch');const target=event[2]==='A'?w.goal:w.goal2;
      count[event[2]]++;if(count[event[2]]===c.threshold&&h.painted[target]===event[2]){delete h.painted[target];physicalErasures++;}
      shockCount++;lastRemoteAt=k;
     }else if(event[1]==='BACKGROUND_VIBRATION'){assert(event[2]==='NONE','invalid background event');lastRemoteAt=k;}
     else throw Error('unexpected synthetic external event '+event[1]);
    }
    const raw=host.input(h),[sx,sy]=h.g.coords[h.pos];raw.painted=raw.painted.filter(([idx])=>{const [x,y]=h.g.coords[idx];return Math.abs(x-sx)+Math.abs(y-sy)<=1});
    raw.v='synthetic-v67-local-paint';raw.events_delta=inp.events_delta;
    assert(H(raw)===H(inp),'pre-action observation mismatch split='+split+' episode='+i+' step='+k);
    assert(inp.legal.includes(r.action),'illegal advertised '+r.action);assert(host.apply(h,r.action).accepted,'host rejected '+r.action);
    assert(r.focus_loss_weight===r.weight,'loss weights diverge');
    if(k===c.critical_step){critical=inp;assert(r.weight===1,'masked critical action');assert(lastRemoteAt>=0&&k-lastRemoteAt>=60,'shock too recent');
     const broken=['A','B'].filter(mat=>h.painted[mat==='A'?w.goal:w.goal2]!==mat); // r.action already applied; if immediate STOP, h unchanged painted
     assert(JSON.stringify(broken)===JSON.stringify(c.broken),'physical damage mismatch');
     assert(H(inp)===c.critical_input_sha256&&r.action===c.first_action,'critical provenance mismatch');
    }
    if(k<c.critical_step)assert(r.weight===0,'scripted intervention leaked into loss');
    decisions++;
   }
   assert(critical&&h.done&&h.success,'terminal failure');assert(probe===1&&shockCount===c.sequence.length,'probe/shock incomplete');
   assert(c.stop_only_success===(c.broken.length===0),'STOP baseline inconsistent');
   assert(c.always_revisit_success===true&&c.always_revisit_overhead>=0,'revisit control inconsistent');
   success++;stop+=Number(c.stop_only_success);revisit+=Number(c.always_revisit_success);overhead+=c.always_revisit_overhead;
   const group=split+':'+c.world_index;if(!firstActions.has(group))firstActions.set(group,{inputs:[],actions:new Set(),signatures:new Set()});
   const g=firstActions.get(group);g.inputs.push(H(critical));g.actions.add(c.first_action);g.signatures.add(H([c.threshold,c.sequence]));
  }
 }
 for(const [key,g] of firstActions){assert(new Set(g.inputs).size===1,'current observation leak '+key);assert(g.actions.size===3,'not three first actions '+key);assert(g.signatures.size===20,'missing crossed arms '+key)}
 assert(mf.total_decisions===decisions&&mf.total_episodes===success&&mf.worlds===seen.size,'manifest totals');
 assert(mf.stop_only_accuracy===stop/success&&mf.always_revisit_success_rate===revisit/success,'controls wrong');
 assert(Math.abs(mf.mean_extra_steps_always_revisit-overhead/success)<1e-9,'cost gap incorrect');
 return {PASS:true,episodes:success,legal_host_replayed_decisions:decisions,split_exclusive_worlds:seen.size,counterfactual_20_arm_worlds:firstActions.size,critical_observation_identical:true,stop_only_successes:stop,always_revisit_successes:revisit,mean_extra_revisit_actions:overhead/success};
}
if(require.main===module){try{assert(process.argv[2],'usage: node test_remote_abrasion_host_v67.js DIR');console.log(JSON.stringify(audit(path.resolve(process.argv[2])),null,2));}catch(e){console.error(e.stack);process.exitCode=1}}
module.exports={audit};
