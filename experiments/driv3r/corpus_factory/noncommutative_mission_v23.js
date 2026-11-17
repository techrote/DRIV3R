'use strict';
// DRIV3R Corpus Factory v23: 8-event noncommutative public instructions select
// BOTH workshop and material phase. Synthetic host/oracle, never native CyberSand.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const v22=require('./curriculum_pulse_v22.js');
const SPLITS={TRAIN:{groups:12,neutral:88},DEV:{groups:6,neutral:120},HELDOUT:{groups:6,neutral:160}};
const OPS={A:[1,0,2,3],B:[0,2,1,3],C:[0,1,3,2],D:[3,1,2,0],E:[2,1,0,3],F:[0,3,2,1],G:[1,2,3,0],H:[3,0,1,2]};
const DIR=[['MOVE_NORTH',0,-1],['MOVE_SOUTH',0,1],['MOVE_WEST',-1,0],['MOVE_EAST',1,0]];
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const copy=x=>JSON.parse(JSON.stringify(x));
const key=(x,y)=>x+','+y;
function evaluateProgram(program){return program.reduce((s,op)=>OPS[op][s],0);}
// Fixed, inspected programme banks. Each split has sixteen unique eight-operator
// sequences (four per final state) and common first TWO / last TWO operators.
// Thus first/last two instructions, inventory and last64 history cannot resolve
// the four-way target. All 48 complete orderings are split-disjoint.
const PROGRAMMES={
 TRAIN:['AGBCFHDE','AGBCHFDE','AGBFCHDE','AGBFHCDE','AGBHFCDE','AGCBFHDE','AGCBHFDE','AGCFBHDE','AGCHBFDE','AGFBCHDE','AGFCHBDE','AGFHBCDE','AGFHCBDE','AGHBCFDE','AGHCFBDE','AGHFBCDE'],
 DEV:['CHABFGDE','CHABGFDE','CHAFBGDE','CHAFGBDE','CHAGFBDE','CHBAGFDE','CHBFAGDE','CHBFGADE','CHBGAFDE','CHFAGBDE','CHFBGADE','CHFGABDE','CHFGBADE','CHGAFBDE','CHGBFADE','CHGFABDE'],
 HELDOUT:['FBCDHEAG','FBCEDHAG','FBCEHDAG','FBCHDEAG','FBCHEDAG','FBDCHEAG','FBDHCEAG','FBECDHAG','FBEDCHAG','FBEHDCAG','FBHCDEAG','FBHCEDAG','FBHDCEAG','FBHDECAG','FBHECDAG','FBHEDCAG']
};
function ablation(rows,cut){const bins=new Map();for(const {program,state} of rows){const k=cut(program),a=bins.get(k)||[0,0,0,0];a[state]++;bins.set(k,a);}return [...bins.values()].reduce((n,a)=>n+Math.max(...a),0)/rows.length;}
function banks(){
 const result={},ablationScores={},all=new Set();
 for(const [split,ps] of Object.entries(PROGRAMMES)){
  const rows=ps.map(s=>{const program=[...s];if(program.length!==8||new Set(program).size!==8||program.some(op=>!OPS[op])||all.has(s))throw Error('invalid/reused public programme '+s);all.add(s);return {program,state:evaluateProgram(program)};});
  const counts=[0,0,0,0];rows.forEach(r=>counts[r.state]++);if(counts.some(n=>n!==4))throw Error('unbalanced programme bank '+split);
  result[split]=rows;ablationScores[split]={first2:ablation(rows,p=>p.slice(0,2).join('')),last2:ablation(rows,p=>p.slice(-2).join('')),first3:ablation(rows,p=>p.slice(0,3).join('')),last3:ablation(rows,p=>p.slice(-3).join(''))};
 }
 return {result,ablationScores};
}
const BANK=banks();
function makeRecord(split,group,variant){
 const cfg=SPLITS[split];if(!cfg||!Number.isInteger(group)||group<0||group>=cfg.groups||!Number.isInteger(variant)||variant<0||variant>=16)throw Error('invalid split/group/variant');
 const r=v22.makeRecord(split,group,variant%8),scenario=BANK.result[split][variant];
 r.episode_id=`v23-${split.toLowerCase()}-${String(group).padStart(3,'0')}-p${variant}`;
 r.program=[...scenario.program];r.private_target={active:scenario.state&1?'S':'N',phase:(scenario.state>>1)&1,program_state:scenario.state};
 r.initial_history.events=[...r.program.map((op,i)=>({kind:'remote_program',ordinal:i,operator:op,accepted:true})),...Array.from({length:cfg.neutral},(_,i)=>({kind:'neutral_receipt',channel:'CLOCK',payload:i%4,accepted:true}))];
 r.initial_history.positions=Array.from({length:cfg.neutral},(_,i)=>i%4===0?'0,0':i%4===1?'0,-1':i%4===2?'0,0':'0,1');
 r.difficulty={TRAIN:'D5',DEV:'D6',HELDOUT:'D7'}[split];return r;
}
function createHost(r){return v22.createHost(r);}
function observe(h){const x=v22.observe(h);x.public_recipe.phase_rule='Apply remote_program operators IN ORDER to state 0 using public programme_operators; state bit 0 = workshop N/S, bit 1 = material phase';x.public_recipe.programme_operators=OPS;
 x.objective='Interpret ordered public receipt programme, choose correct workshop and motif phase, respond to public law/goal revisions, repair edits, return and STOP';return x;}
function publicMission(h){
 const x=observe(h),events=x.history.events.filter(e=>e.kind==='remote_program');
 if(events.length!==8||events.map(e=>e.ordinal).join(',')!=='0,1,2,3,4,5,6,7'||new Set(events.map(e=>e.operator)).size!==8)throw Error('bad public ordered programme');
 const state=evaluateProgram(events.map(e=>e.operator));let branch=state&1?'S':'N';
 for(const e of x.history.events.filter(e=>e.kind==='mission_revision'))if(e.public_goal_update==='FLIP_WORKSHOP')branch=branch==='N'?'S':'N';
 const phase=(state>>1)&1;
 if(branch!==h.target.active||phase!==h.target.phase)throw Error('private mission inconsistent with public receipts');
 if(JSON.stringify(x.public_recipe.base_materials)!==JSON.stringify(h.world.palette))throw Error('palette not public');
 return {branch,phase,palette:x.public_recipe.base_materials,motif:x.public_recipe.motif_pattern};
}
function route(h,goal){
 const queue=[h.pos],seen=new Set(queue),prior=new Map();
 for(let i=0;i<queue.length;i++){
  const p=queue[i];if(p===goal){const actions=[];for(let at=goal;at!==h.pos;){const [back,action]=prior.get(at);actions.push(action);at=back;}return actions.reverse();}
  const [x,y]=p.split(',').map(Number);
  for(const [action,dx,dy] of DIR){const next=key(x+dx,y+dy),t=h.world.tiles[next];if(!t||t.type==='protected'||(t.type==='gate'&&h.shutters_closed)||seen.has(next))continue;seen.add(next);prior.set(next,[p,action]);queue.push(next);}
 }throw Error('unreachable '+h.pos+' to '+goal);
}
function oracleAction(h){
 const m=publicMission(h),todo=[];
 for(const branch of ['N','S'])for(let i=0;i<h.world.patches[branch].length;i++){
  const k=h.world.patches[branch][i],want=branch===m.branch?m.palette[(m.motif[i]+m.phase)%2]:null;
  if((h.paint[k]||null)!==want)todo.push({k,want});
 }
 if(todo.length){const at=todo.find(t=>t.k===h.pos);if(at)return h.paint[h.pos]?'ERASE':'PAINT_'+at.want;
  let best=null;for(const t of todo){const r=route(h,t.k);if(!best||r.length<best.length)best=r;}return best[0];}
 return h.pos==='0,0'?'STOP':route(h,'0,0')[0];
}
function apply(h,action){return v22.apply(h,action);}
function replay(rec,onDecision){
 const h=createHost(rec);let actions=0,moves=0,paints=0,erases=0;
 while(!h.done){if(actions>=650)throw Error('oracle timeout '+rec.episode_id);
  const input=observe(h),target=oracleAction(h);if(!input.legal_actions.includes(target))throw Error('illegal oracle action '+target);
  const receipt=apply(h,target);if(!receipt.accepted)throw Error('refused oracle action '+target);
  if(target.startsWith('MOVE_'))moves++;if(target.startsWith('PAINT_'))paints++;if(target==='ERASE')erases++;
  if(onDecision)onDecision({episode_id:rec.episode_id,world_id:rec.world_id,split:rec.split,step:actions,model_input:input,target,receipt,training_weight:1});actions++;
 }
 if(!h.stop_accepted||h.invalid||h.pulses.length!==rec.pulse_plan.length)throw Error('terminal incorrect '+rec.episode_id);
 return {episode_id:rec.episode_id,world_id:rec.world_id,split:rec.split,programme:rec.program.join(''),programme_state:rec.private_target.program_state,geometry_sha256:rec.world.fingerprint,oracle_actions:actions,accepted_moves:moves,accepted_paints:paints,corrective_erases:erases,public_pulses:h.pulses.length,terminal_correct:true,provenance:'synthetic-v23; NOT native CyberSand'};
}
function verifyFamily(split,group){
 const rs=Array.from({length:16},(_,i)=>makeRecord(split,group,i));
 const last=rs.map(r=>{const x=observe(createHost(r));x.history.events=x.history.events.slice(-64);x.history.positions=x.history.positions.slice(-64);return sha(x);});
 if(new Set(last).size!==1)throw Error('last64 leakage');
 const states=[0,0,0,0];for(const r of rs)states[r.private_target.program_state]++;
 if(states.some(x=>x!==4))throw Error('unbalanced public programme target');
 if(new Set(rs.map(r=>[...r.program].sort().join(''))).size!==1)throw Error('unordered programme leak');return rs;
}
function fileMeta(file,rows){const h=crypto.createHash('sha256'),fd=fs.openSync(file,'r'),buf=Buffer.allocUnsafe(1<<20);let bytes=0,n;try{while((n=fs.readSync(fd,buf,0,buf.length,null))>0){h.update(buf.subarray(0,n));bytes+=n;}}finally{fs.closeSync(fd);}return {sha256:h.digest('hex'),bytes,rows};}
function generate(dest){
 if(fs.existsSync(dest))throw Error('refusing overwrite');fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},seenWorld=new Set(),allPrograms=new Set(),firstActions={};let total=0;
 for(const [split,cfg] of Object.entries(SPLITS)){
  const epFile=split+'_episodes.jsonl',decFile=split+'_decisions.jsonl',ep=fs.openSync(path.join(dest,epFile),'w'),de=fs.openSync(path.join(dest,decFile),'w');
  let episodes=0,decisions=0,erases=0,paints=0,pulses=0,duplicates=0;const seen=new Map(),programs=new Set();const first={N:0,S:0};
  try{for(let group=0;group<cfg.groups;group++){
   const rs=verifyFamily(split,group),world=rs[0].world;if(seenWorld.has(world.fingerprint))throw Error('cross-split geometry leakage');seenWorld.add(world.fingerprint);
   for(const r of rs){programs.add(r.program.join(''));
    const episode=replay(r,d=>{const signature=sha(d.model_input),old=seen.get(signature);if(old!==undefined){if(old!==d.target)throw Error('identical input conflicting target');duplicates++;d.training_weight=split==='TRAIN'?0:1;}else seen.set(signature,d.target);
     if(d.step===0)first[d.target==='MOVE_NORTH'?'N':d.target==='MOVE_SOUTH'?'S':'INVALID']=(first[d.target==='MOVE_NORTH'?'N':d.target==='MOVE_SOUTH'?'S':'INVALID']||0)+1;
     fs.writeSync(de,JSON.stringify(d)+'\n');decisions++;});
    fs.writeSync(ep,JSON.stringify(episode)+'\n');episodes++;erases+=episode.corrective_erases;paints+=episode.accepted_paints;pulses+=episode.public_pulses;
   }
  }}finally{fs.closeSync(ep);fs.closeSync(de);}
  for(const p of programs){if(allPrograms.has(p))throw Error('programme reused between splits');allPrograms.add(p);}
  if(first.N!==episodes/2||first.S!==episodes/2)throw Error('first action imbalance');firstActions[split]=first;
  shards[epFile]=fileMeta(path.join(dest,epFile),episodes);shards[decFile]=fileMeta(path.join(dest,decFile),decisions);
  splits[split]={world_groups:cfg.groups,episodes,decisions,programmes:programs.size,first_actions:first,corrective_erases:erases,accepted_paints:paints,public_pulses:pulses,exact_duplicate_inputs:duplicates,unique_model_inputs:seen.size,zero_weight_train_duplicates:split==='TRAIN'?duplicates:0};total+=decisions;
 }
 const manifest={schema:'driv3r/corpus_factory/v23',label:'synthetic host; NOT native CyberSand',generator:'noncommutative_mission_v23.js',parent:['curriculum_pulse_v22.js','recipe_grounded_v21.js'],public_operators:OPS,programme_bank_sha256:sha(BANK.result),programme_bank_method:'predeclared 8-permutation sequence subset, disjoint by split',programme_prefix_suffix_ablations:BANK.ablationScores,split_disjoint_programmes:true,split_disjoint_geometry:true,splits,episodes:Object.values(splits).reduce((a,b)=>a+b.episodes,0),decisions:total,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={OPS,SPLITS,BANK,makeRecord,createHost,observe,publicMission,oracleAction,apply,replay,verifyFamily,generate,evaluateProgram};
if(require.main===module){if(!process.argv[2])throw Error('Usage: node noncommutative_mission_v23.js NEW_EMPTY_DIRECTORY');console.log(JSON.stringify(generate(path.resolve(process.argv[2])).splits));}
