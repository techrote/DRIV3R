'use strict';
// DRIV3R Corpus Factory v22: public-law pulses, goal retarget, correction and split-isolated replay.
// All receipts/recipes are synthetic. Oracle planning may inspect topology but never hidden law bits.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./recipe_grounded_v21.js');
const DIR=[['MOVE_NORTH',0,-1],['MOVE_SOUTH',0,1],['MOVE_WEST',-1,0],['MOVE_EAST',1,0]];
const SPLITS={TRAIN:{groups:24,mode:['law','goal','both','law']},DEV:{groups:8,mode:['goal','law','both','double']},HELDOUT:{groups:8,mode:['both','double','goal','double']}};
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const key=(x,y)=>x+','+y;
const json=x=>JSON.parse(JSON.stringify(x));
function makeRecord(split,group,variant){
 if(!SPLITS[split]||!Number.isInteger(group)||group<0||!Number.isInteger(variant)||variant<0||variant>7)throw Error('invalid split/group/variant');
 const r=base.makeRecord(split,group,variant),mode=SPLITS[split].mode[group%SPLITS[split].mode.length];
 r.episode_id=r.episode_id.replace('v21-','v22-');r.mode='clean';
 r.pulse_plan=[{after_paints:split==='TRAIN'?2:split==='DEV'?3:4,kind:mode==='double'?'law':mode}];
 if(mode==='double')r.pulse_plan.push({after_paints:7,kind:'goal'});
 r.difficulty={TRAIN:'D4',DEV:'D5',HELDOUT:'D6'}[split];
 return r;
}
function createHost(rec){
 const h=base.createHost({...rec,world:json(rec.world),mode:'clean'});
 h.pulse_plan=json(rec.pulse_plan);h.pulses=[];h.paint_actions=0;h.original_target=h.target.active;
 return h;
}
function observe(h){
 const obs=base.observe(h);obs.objective='Follow accepted remote receipts; respond to public mission/law revisions; repair contaminated cells; finish one workshop, leave the other clean, return and STOP';
 obs.mission_revision=h.pulses.length;
 return obs;
}
function apply(h,action){
 const r=base.apply(h,action);
 if(!r.accepted||!action.startsWith('PAINT_'))return r;
 h.paint_actions++;
 if(h.pulse_plan.length&&h.paint_actions===h.pulse_plan[0].after_paints){
  const pulse=h.pulse_plan.shift();const kind=pulse.kind;
  if(kind==='law'||kind==='both'){
   h.world.palette=[h.world.palette[1],h.world.palette[0]];
   // Changing palette alone is a genuine material-law shift, not a hidden target update.
  }
  if(kind==='goal'||kind==='both')h.target.active=h.target.active==='N'?'S':'N';
  const receipt={kind:'mission_revision',revision:h.pulses.length+1,public_law_update:(kind==='law'||kind==='both')?'SWAP_BASE_MATERIAL_ORDER':'UNCHANGED',public_goal_update:(kind==='goal'||kind==='both')?'FLIP_WORKSHOP':'UNCHANGED',accepted:true};
  h.pulses.push(receipt);h.events.push(receipt);
 }
 return r;
}
function inferPublicMission(h){
 const x=observe(h),remote=x.history.events.filter(e=>e.kind==='remote_receipt').slice(0,3);
 if(remote.length!==3||remote.map(e=>e.channel).join(',')!=='OCHRE,VIOLET,JADE')throw Error('missing causal receipts');
 const bits=remote.map(e=>e.payload);if(bits.some(b=>b!==0&&b!==1))throw Error('invalid causal bit');
 let branch=(bits[0]^bits[1]^bits[2])?'S':'N';
 for(const e of x.history.events.filter(e=>e.kind==='mission_revision'))if(e.public_goal_update==='FLIP_WORKSHOP')branch=branch==='N'?'S':'N';
 if(branch!==h.target.active)throw Error('hidden/observed target disagree');
 if(JSON.stringify(x.public_recipe.base_materials)!==JSON.stringify(h.world.palette))throw Error('recipe not exposed');
 return {branch,phase:bits[0]^bits[1],palette:x.public_recipe.base_materials,motif:x.public_recipe.motif_pattern};
}
function route(h,goal){
 const q=[h.pos],seen=new Set(q),prior=new Map();
 for(let i=0;i<q.length;i++){
  const p=q[i];if(p===goal){const result=[];let at=goal;while(at!==h.pos){const [from,action]=prior.get(at);result.push(action);at=from;}return result.reverse();}
  const [x,y]=p.split(',').map(Number);
  for(const [action,dx,dy] of DIR){const n=key(x+dx,y+dy),t=h.world.tiles[n];if(!t||t.type==='protected'||(t.type==='gate'&&h.shutters_closed)||seen.has(n))continue;seen.add(n);prior.set(n,[p,action]);q.push(n);}
 }
 throw Error('unreachable '+h.pos+' -> '+goal);
}
function expected(m,index){return m.palette[(m.motif[index]+m.phase)%2];}
function oracleAction(h){
 // The target, goal flips, palette and motif are reconstructed from PUBLIC receipts/recipe.
 const m=inferPublicMission(h),todo=[];
 for(const branch of ['N','S'])for(let i=0;i<h.world.patches[branch].length;i++){
  const k=h.world.patches[branch][i],want=branch===m.branch?expected(m,i):null;
  if((h.paint[k]||null)!==want)todo.push({k,want});
 }
 if(todo.length){
  const at=todo.find(t=>t.k===h.pos);
  if(at)return h.paint[h.pos]?'ERASE':'PAINT_'+at.want;
  let best=null;for(const t of todo){const r=route(h,t.k);if(best===null||r.length<best.length)best=r;}
  return best[0];
 }
 return h.pos==='0,0'?'STOP':route(h,'0,0')[0];
}
function replay(rec,onDecision){
 const h=createHost(rec);let actions=0,paints=0,erases=0,moves=0;
 while(!h.done){
  if(actions++>600)throw Error('oracle timeout '+rec.episode_id);
  const input=observe(h),target=oracleAction(h);if(!input.legal_actions.includes(target))throw Error('illegal oracle '+target);
  const receipt=apply(h,target);if(!receipt.accepted)throw Error('refused oracle '+target);
  if(target.startsWith('PAINT_'))paints++;if(target==='ERASE')erases++;if(target.startsWith('MOVE_'))moves++;
  if(onDecision)onDecision({episode_id:rec.episode_id,world_id:rec.world_id,split:rec.split,step:actions-1,model_input:input,target,receipt,scenario_mode:rec.pulse_plan.map(x=>x.kind).join('+')});
 }
 if(!h.stop_accepted||h.pulses.length!==rec.pulse_plan.length||h.invalid!==0)throw Error('unsuccessful replay '+rec.episode_id);
 return {episode_id:rec.episode_id,world_id:rec.world_id,split:rec.split,scenario_mode:rec.pulse_plan.map(x=>x.kind).join('+'),geometry_sha256:rec.world.fingerprint,oracle_actions:actions,accepted_moves:moves,accepted_paints:paints,corrective_erases:erases,public_pulses:h.pulses.length,terminal_correct:true,provenance:'synthetic-v22 NOT native CyberSand'};
}
function verifyFamily(split,group){
 const rs=Array.from({length:8},(_,i)=>makeRecord(split,group,i));
 const signatures=rs.map(r=>{const x=observe(createHost(r));x.history.events=x.history.events.slice(-64);x.history.positions=x.history.positions.slice(-64);return sha(x);});
 if(new Set(signatures).size!==1)throw Error('last64 observation leaked target');
 const counts=rs.reduce((a,r)=>(a[r.private_target.active]++,a),{N:0,S:0});if(counts.N!==4||counts.S!==4)throw Error('unbalanced family');
 return rs;
}
function fileMeta(p,rows){const b=fs.readFileSync(p);return {sha256:sha(b),bytes:b.length,rows};}
function generate(dest,scale=1){
 if(!Number.isInteger(scale)||scale<1||scale>2)throw Error('scale 1..2');if(fs.existsSync(dest))throw Error('refusing overwrite');fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},worlds={};let total=0;
 for(const [split,cfg] of Object.entries(SPLITS)){
  const epPath=path.join(dest,split+'_episodes.jsonl'),decPath=path.join(dest,split+'_decisions.jsonl');
  const ep=fs.openSync(epPath,'w'),de=fs.openSync(decPath,'w');let episodes=0,decisions=0,duplicates=0,erases=0,paints=0,pulses=0;
  const seen=new Map();
  try{
   for(let group=0;group<cfg.groups*scale;group++){
    const rs=verifyFamily(split,group),world=rs[0].world;
    if(worlds[world.fingerprint])throw Error('geometry group leak');worlds[world.fingerprint]=world.id;
    for(const r of rs){
     const e=replay(r,d=>{
      const signature=sha(d.model_input),old=seen.get(signature);
      if(old!==undefined){if(old!==d.target)throw Error('contradictory fully observed inputs');duplicates++;}
      else seen.set(signature,d.target);
      d.training_weight=split==='TRAIN'&&old!==undefined?0:1;
      fs.writeSync(de,JSON.stringify(d)+'\n');decisions++;
     });
     fs.writeSync(ep,JSON.stringify(e)+'\n');episodes++;erases+=e.corrective_erases;paints+=e.accepted_paints;pulses+=e.public_pulses;
    }
   }
  }finally{fs.closeSync(ep);fs.closeSync(de);}
  const epFile=split+'_episodes.jsonl',decFile=split+'_decisions.jsonl';shards[epFile]=fileMeta(epPath,episodes);shards[decFile]=fileMeta(decPath,decisions);
  splits[split]={world_groups:cfg.groups*scale,episodes,decisions,corrective_erases:erases,accepted_paints:paints,public_pulses:pulses,exact_duplicate_inputs:duplicates,unique_inputs:seen.size,zero_weight_train_duplicates:split==='TRAIN'?duplicates:0};total+=decisions;
 }
 const manifest={schema:'driv3r/corpus_factory/v22',provenance:'synthetic host, not native CyberSand',generator:'curriculum_pulse_v22.js',parent:'recipe_grounded_v21.js',scale,splits,episodes:Object.values(splits).reduce((a,b)=>a+b.episodes,0),decisions:total,world_groups:Object.keys(worlds).length,worlds_sha256:sha(Object.entries(worlds).sort()),shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={SPLITS,makeRecord,createHost,observe,apply,inferPublicMission,oracleAction,replay,verifyFamily,generate};
if(require.main===module){if(!process.argv[2])throw Error('node curriculum_pulse_v22.js NEW_DIRECTORY [scale=1]');console.log(JSON.stringify(generate(path.resolve(process.argv[2]),Number(process.argv[3]||1)).splits));}
