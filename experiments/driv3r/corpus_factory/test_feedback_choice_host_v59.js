#!/usr/bin/env node
'use strict';
/* Independent exact observable host replay: read output only; never imports v59. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const v53=require('./deferred_surface_v53.js'),v52=require('./realert_v52.js'),v46=require('./mission_memory_v46.js'),v58=require('./offpolicy_repair_v58.js');
const src=process.argv[2];assert(src,'usage node test_feedback_choice_host_v59.js V59_OUTPUT_DIR');
const lines=f=>fs.readFileSync(path.join(src,f),'utf8').trim().split('\n').map(JSON.parse);
let decisions=0,success=0,correct=0,wrong=0,zero=0;
for(const split of ['TRAIN','DEV','HELDOUT']){
 const eps=lines(split+(split==='TRAIN'?'_receipt_choice_stream.jsonl':'_EVALUATOR_ONLY_receipt_choice_stream.jsonl'));
 const certs=lines(split+'_receipt_choice_certificates_EVALUATOR_ONLY.jsonl');
 assert.equal(eps.length,certs.length);
 for(let i=0;i<eps.length;i++){
  const c=certs[i],arm=v53.ARMS.find(z=>z.id===c.arm),w=v46.candidate(split,700+c.world_index).w;
  assert.equal(w.geometry_sha256,c.geometry_sha256);assert.equal(w.seed,c.source_seed);
  const par=v53.armParams(arm),h=v53.prep(w,arm);let prior=[];
  const steps=eps[i].steps;
  for(let t=0;t<steps.length;t++){
   const alert=v52.notice(h,par);v53.injectB(h,arm,alert,t);
   const obs=v53.observe(h),partial=v58.streamify(obs,prior),s=steps[t];
   assert.deepEqual(partial,s.input,'host-observed input diverged at '+split+' episode '+i+' step '+t);
   assert(obs.legal.includes(s.action),'illegal action at replay');prior=obs.durable;
   const receipt=v52.command(h,s.action,par);
   if(t===c.source_step){
    assert.equal(s.weight,0);assert.equal(s.focus_loss_weight,0);
    assert.equal(receipt.material,c.postintervention_physical_result);
    if(c.branch_type==='correct'){assert.equal(h.stage,2);correct++;}
    else{assert.equal(h.stage,1);wrong++;}
    zero++;
   }
   decisions++;
  }
  assert(h.done&&h.success&&!h.protected_damage,'synthetic completion absent');success++;
 }
}
console.log(JSON.stringify({audit:'PASS',source:'independent host action-by-action exact observation replay',
 success_episodes:success,decisions,correct,wrong,masked_interventions:zero,synthetic_not_native:true},null,2));
