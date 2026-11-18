'use strict';
/* Negative control: deterministic frozen-calibration teacher ignores warning,
 * does no active sensing; synthetic-host-only, NOT a trained model. */
const v51=require('./active_probe_v51.js'),v50=require('./online_recovery_v50.js'),v46=require('./mission_memory_v46.js');
function frozen(w,arm){const h=v51.prep(w,arm);let actions=0;
 for(;actions<320&&!h.done;actions++){
  v51.alert(h,arm);const i=v51.observe(h),a=v50.teacher(i);
  if(!i.legal.includes(a))throw Error('frozen issued illegal '+a);
  v51.command(h,a);
 }
 return {success:!!h.success,protected_damage:!!h.protected_damage,actions};
}
function main(){const out={};let total=0,pass=0,damages=0;
 for(const arm of v51.ARMS){let success=0,damaged=0;
  for(let ix=0;ix<16;ix++){const w=v46.candidate('HELDOUT',300+ix).w,z=frozen(w,arm);total++;success+=Number(z.success);pass+=Number(z.success);damaged+=Number(z.protected_damage);damages+=Number(z.protected_damage);}
  out[arm.id]={episodes:16,success,protected_damage:damaged};
 }
 if(pass!==80||damages!==64)throw Error('expected 80/144 frozen successes and 64 protected-target failures, got '+pass+'/'+damages);
 console.log(JSON.stringify({synthetic_only:true,control:'frozen physical-law teacher without active safe probing',total,terminal_success:pass,protected_target_damage:damages,by_arm:out},null,2));
}
if(require.main===module)main();module.exports={frozen};
