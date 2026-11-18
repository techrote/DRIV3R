'use strict';
/* Deterministic evaluation control only. This is NOT a learned policy. */
const v50=require('./online_recovery_v50.js'),v46=require('./mission_memory_v46.js');
const assert=(x,msg)=>{if(!x)throw Error(msg)};
function stalePolicy(i){const physical=v46.teacher(i);if(!physical.startsWith('PAINT_'))return physical;
 const receipts=i.durable.filter(e=>e[1]==='PAINT_RESULT'&&e[4]===i.surface);
 assert(receipts.length>=2,'no baseline calibrations');
 const lastCalibration=receipts[1];
 const swapped=lastCalibration[2].slice(-1)!==lastCalibration[3];
 return swapped?'PAINT_'+(physical.endsWith('A')?'B':'A'):physical;
}
function staleRollout(w,arm){const h=v50.prep(w,arm);let n=0,mistakes=0;
 for(;n<256&&!h.done;n++){
  v50.maybeDrift(h,arm);const i=v50.observe(h),a=stalePolicy(i);assert(i.legal.includes(a),'illegal baseline');
  const before=h.stage,at=h.pos,r=v50.command(h,a);
  if(a.startsWith('PAINT_')&&at===(before===0?h.w.goal:h.w.goal2)&&r.material!==(before===0?'A':'B'))mistakes++;
 }
 return {success:h.success,done:h.done,decisions:n,wrong_paints:mistakes};}
let outcome={worlds:16,episodes:0,stale_success:0,stale_timeout:0,wrong_paints:0,drift_static_success:0,drift_arm_failure:0};
for(let i=0;i<16;i++){
 const w=v46.candidate('HELDOUT',200+i).w;
 for(const arm of v50.arms){const r=staleRollout(w,arm);outcome.episodes++;
  outcome.stale_success+=Number(!!r.success);outcome.stale_timeout+=Number(!r.done);
  outcome.wrong_paints+=r.wrong_paints;
  if(!arm.flipA&&!arm.flipB){assert(r.success,'stale policy wrongly failed on no-drift control');outcome.drift_static_success++;}
  else{assert(!r.success,'stale policy unexpectedly recovered drift');outcome.drift_arm_failure++;}
 }
}
console.log(JSON.stringify(outcome,null,2));