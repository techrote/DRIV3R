/* DRIV3R Corpus Factory v19 — recovery from wrong edits, rejected moves and premature exits.
 * Synthetic host and oracle-labelled DATA ONLY. No native CyberSand ABI claims.
 * Requires pinned v18 and v17 source on the same branch. Node >=20.
 * Usage: node recovery_edits_v19.js NEW_OUTPUT_DIR [worldSeeds=4]
 */
"use strict";
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v18=require('./motif_branch_v18.js'),v17=require('./aliased_order_v17.js');
const SPLITS=['TRAIN','DEV','HELDOUT'];
const MODES={TRAIN:['spot','all','blocked'],DEV:['alternating','blocked'],HELDOUT:['cross','exit','blocked']};
const MOVES=[['MOVE_NORTH',0,-1],['MOVE_EAST',1,0],['MOVE_SOUTH',0,1],['MOVE_WEST',-1,0]];
const key=p=>p.join(','),sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function directions(points){const a=[];for(let i=1;i<points.length;i++){
 const x=points[i-1],y=points[i],d=MOVES.find(d=>x[0]+d[1]===y[0]&&x[1]+d[2]===y[1]);if(!d)throw Error('non-cardinal path');a.push(d[0]);}return a;}
function act(h,a,expect=true){const r=v18.apply(h,a);if(r.ok!==expect)throw Error(`intervention action ${a}: expected ${expect}, got ${JSON.stringify(r)}`);return r;}
function travel(h,p,onStep){const route=v18.shortest(h.w,h.position,p);if(!route)throw Error(`unreachable ${p}`);
 for(const a of directions(route)){act(h,a);if(onStep)onStep(a);}}
function workcell(w,lane,dx){return [w.cx+dx,w.jy+(lane==='NORTH'?-2:2)];}
function prepare(h,mode){const counts={accepted_paint:0,rejected:0,scripted_steps:0};const move=(a)=>{act(h,a);counts.scripted_steps++;};
 const go=p=>travel(h,p,()=>counts.scripted_steps++);
 if(['spot','all','alternating','cross'].includes(mode)){
  const indexes={spot:[2],all:[2,3,5,6,7],alternating:[2,5,7],cross:[3,6]}[mode];
  for(const lane of ['NORTH','SOUTH'])for(const dx of indexes){go(workcell(h.w,lane,dx));move('PAINT_STONE');counts.accepted_paint++;}
  go(h.w.start);
 }else if(mode==='blocked'){
  for(const lane of ['NORTH','SOUTH']){
   go(workcell(h.w,lane,3));const pos=h.position.slice(),before=h.steps;
   act(h,'MOVE_EAST',false);if(h.steps!==before)throw Error('invalid move advanced time');counts.rejected++;
   h.receipts.push({step:343+h.steps,kind:'synthetic_refusal',position:pos,action:'MOVE_EAST',accepted:false,reason:'protected_cell'});
  }
  go(h.w.start);
 }else if(mode==='exit'){go(h.w.exit);if(h.done)throw Error('premature exit succeeded');go(h.w.start);}
 else throw Error('unknown mode '+mode);
 if(key(h.position)!==key(h.w.start)||h.done)throw Error('intervention did not return to junction');
 // Make the last 64 positions/events common within each world/mode regardless of the hidden ALPHA history.
 for(let pass=0;pass<4;pass++){
  const far=[h.w.cx,h.w.jy-9];go(far);go(h.w.start);
 }
 if(h.trail.length<65)throw Error('short history');return counts;
}
function makeRecord(split,seed,orderIndex,mode){const w=v18.world(split,seed),base=v18.makeEpisode(w,v17.ORDERS_BY_SPLIT[split][orderIndex],orderIndex,{TRAIN:1,DEV:2,HELDOUT:3}[split]);
 const h=v18.createHost(base),intervention=prepare(h,mode),family=`v19/${split}/seed${seed}/${mode}`;
 const record={schema:'driv3r/corpus-factory-recovery-edits-v19',split,episode_id:`${family}/order${orderIndex}`,group_id:`v19/${split}/seed${seed}`,
  family_id:family,topology_group:`v19/${split}/length${w.length}`,difficulty:mode==='all'?'D12':'D10',
  provenance:{generator:'recovery_edits_v19.js',synthetic_only:true,v18_dependency:'motif_branch_v18.js',v17_dependency:'aliased_order_v17.js',mode},
  base_episode:base,intervention_mode:mode,scripted_intervention_counts:intervention,
  checkpoint_model_input:v18.observe(h),native_receipt:false,
  consumer_contract:'Model receives only checkpoint_model_input, then observe(host); never base_episode, intervention_mode, split, target, future labels or privileged_host.'};
 return record;
}
function createHost(record){const h=v18.createHost(record.base_episode);prepare(h,record.intervention_mode);
 const expected=sha(JSON.stringify(record.checkpoint_model_input)),actual=sha(JSON.stringify(v18.observe(h)));
 if(actual!==expected)throw Error('checkpoint mismatch');return h;
}
function recovery(h,emit){const firstInvalid=h.invalid,base=h.episode,chosen=base.privileged_host.lane,other=chosen==='NORTH'?'SOUTH':'NORTH';
 const target=base.privileged_host.target,actions=[];let paints=0,erases=0;
 function doAction(action,phase){const obs=v18.observe(h);if(!obs.legal_actions.includes(action))throw Error('illegal expert '+action);
  if(emit)emit({schema:'driv3r/corpus-factory-recovery-edits-decision-v19',split:base.split,episode_id:base.episode_id,
   time_index:actions.length,model_input:obs,target:action,phase,
   other_legal_not_selected:obs.legal_actions.filter(x=>x!==action),native_receipt:false});
  act(h,action);actions.push(action);if(action.startsWith('PAINT_'))paints++;if(action==='ERASE')erases++;}
 function nav(p,phase){const route=v18.shortest(h.w,h.position,p);if(!route)throw Error('recovery unreachable');
  for(const a of directions(route))doAction(a,phase);}
 for(const dx of [2,3,5,6,7]){
  const p=workcell(h.w,chosen,dx),k=key(p);nav(p,'recover_target');
  if(h.paint[k]!==target[k])doAction('PAINT_'+target[k],'correct_or_paint');
 }
 for(const dx of [2,3,5,6,7]){
  const p=workcell(h.w,other,dx);if(!h.paint[key(p)])continue;
  nav(p,'clean_inactive');doAction('ERASE','erase_contamination');
 }
 nav(h.w.exit,'return_exit');
 if(!h.done||h.invalid!==firstInvalid||key(h.position)!==key(h.w.exit))throw Error('expert recovery failed');
 if(Object.keys(h.paint).length!==Object.keys(target).length)throw Error('residual paint');
 for(const [k,v] of Object.entries(target))if(h.paint[k]!==v)throw Error('wrong motif');
 return {actions:actions.length,paint_actions:paints,erase_actions:erases,first_action:actions[0],initial_rejections:firstInvalid};
}
function recentKey(rec){return sha(JSON.stringify(v18.recent(rec.checkpoint_model_input,64)));}
function bagKey(rec){return sha(JSON.stringify(v18.eventBag(rec.checkpoint_model_input)));}
function recordStream(dest,seeds=4){if(!Number.isInteger(seeds)||seeds<1||seeds>12)throw Error('worldSeeds must be 1..12');
 if(fs.existsSync(dest))throw Error('refuse overwrite');fs.mkdirSync(dest,{recursive:true});
 const handles={},shards={},summary={},seen=new Set(),families=new Map();
 for(const split of SPLITS)for(const kind of ['episodes','decisions']){
  const name=`${split}_${kind}.jsonl`,fp=path.join(dest,name);handles[name]=fs.openSync(fp,'wx');shards[name]={rows:0,bytes:0,usage:split};}
 function append(split,kind,row){const name=`${split}_${kind}.jsonl`,buf=Buffer.from(JSON.stringify(row)+'\n');fs.writeSync(handles[name],buf);shards[name].rows++;shards[name].bytes+=buf.length;}
 try{
  for(const split of SPLITS){const worldCount=seeds+(split==='TRAIN'?2:0);let decisions=0,paints=0,erases=0,scriptPaint=0,rejected=0,episodes=0;
   for(let seed=0;seed<worldCount;seed++)for(const mode of MODES[split]){
    const family=[],labels=[];
    for(let i=0;i<v17.ORDERS_BY_SPLIT[split].length;i++){
     const rec=makeRecord(split,seed,i,mode),full=sha(JSON.stringify(rec.checkpoint_model_input));
     if(seen.has(full))throw Error('duplicate full observation '+rec.episode_id);seen.add(full);
     const h=createHost(rec),result=recovery(h,row=>{append(split,'decisions',{...row,episode_id:rec.episode_id,group_id:rec.group_id,family_id:rec.family_id,topology_group:rec.topology_group});decisions++;});
     family.push(rec);labels.push(result.first_action);paints+=result.paint_actions;erases+=result.erase_actions;
     scriptPaint+=rec.scripted_intervention_counts.accepted_paint;rejected+=rec.scripted_intervention_counts.rejected;
     append(split,'episodes',{...rec,oracle_metrics:result});episodes++;
    }
    if(family.length!==12||new Set(family.map(recentKey)).size!==1||new Set(family.map(bagKey)).size!==1)throw Error('projection leakage '+family[0].family_id);
    if(labels.filter(x=>x==='MOVE_NORTH').length!==6||labels.filter(x=>x==='MOVE_SOUTH').length!==6)throw Error('first action imbalance '+family[0].family_id);
    families.set(family[0].family_id,{split,mode,world:seed,recent64_classes:1,bag_classes:1,first_action_balance:{north:6,south:6}});
   }
   summary[split]={world_groups:worldCount,families:worldCount*MODES[split].length,episodes,decisions,expert_paint:paints,expert_erase:erases,scripted_paint:scriptPaint,scripted_refusals:rejected,modes:MODES[split]};
  }
 }finally{for(const fd of Object.values(handles))fs.closeSync(fd);}
 for(const [name,meta] of Object.entries(shards)){const bytes=fs.readFileSync(path.join(dest,name));if(bytes.length!==meta.bytes)throw Error('size mismatch');meta.sha256=sha(bytes);}
 const manifest={schema:'driv3r/corpus-factory-recovery-edits-v19',version:'recovery-edits-v19',synthetic_only:true,
  native_cybersand_abi_verified:false,dependencies:{v18:'motif_branch_v18.js',v17:'aliased_order_v17.js',v13:'relay_memory_v13.js'},
  generation_world_seeds:seeds,split_summary:summary,families:families.size,split_grouping:'world seed + split; twelve sibling orders always grouped',
  projection:'each within-world/mode 12-way family has identical last64 and unordered-event bag, balanced 6/6 legal first moves',
  leakage_warnings:['Related v17/v18 source order families: do not pool as independent heldout tests','Environment intervention scripts are shared within each counterfactual family and MUST NOT be model inputs','Event grammar still finite-state and solvable by a symbolic ordered-relay interpreter'],
  usage:'TRAIN fit; DEV tune; HELDOUT once after fixed selection. For rollout pass only checkpoint_model_input/observe and host-legal actions.',
  shards};fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
 return manifest;
}
if(require.main===module){if(!process.argv[2])throw Error('Usage: node recovery_edits_v19.js NEW_OUTPUT_DIR [worldSeeds=4]');
 const result=recordStream(process.argv[2],process.argv[3]===undefined?4:Number(process.argv[3]));console.log(JSON.stringify(result.split_summary));}
module.exports={MODES,prepare,makeRecord,createHost,recovery,recentKey,bagKey,recordStream};
