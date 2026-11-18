'use strict';
// DRIV3R corpus_factory v20: fresh synthetic 2-D topology + multi-receipt relay task.
// No policy is trained here. No claim of native CyberSand ABI equivalence.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const DIRECTIONS=[['MOVE_NORTH',0,-1],['MOVE_SOUTH',0,1],['MOVE_WEST',-1,0],['MOVE_EAST',1,0]];
const MATERIALS=['SAND','METAL','GLASS','ASH'];
const SPLITS={TRAIN:{groups:20,min:14,max:19,detours:2},DEV:{groups:8,min:20,max:25,detours:3},HELDOUT:{groups:8,min:26,max:33,detours:4}};
const key=(x,y)=>`${x},${y}`;
const hash=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
function fileMeta(file,rows){const h=crypto.createHash('sha256'),fd=fs.openSync(file,'r'),buf=Buffer.allocUnsafe(1024*1024);let bytes=0,n=0;try{while((n=fs.readSync(fd,buf,0,buf.length,null))>0){h.update(buf.subarray(0,n));bytes+=n;}}finally{fs.closeSync(fd);}return {sha256:h.digest('hex'),bytes,rows};}
function seeded(s){let x=parseInt(hash(s).slice(0,8),16)>>>0;return ()=>{x=(x+0x6D2B79F5)>>>0;let t=x;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function makeWorld(split,group){
 if(!SPLITS[split]||!Number.isInteger(group)||group<0)throw Error('invalid split/group');
 const cfg=SPLITS[split],rng=seeded(`v20:${split}:${group}`),rint=(lo,hi)=>lo+Math.floor(rng()*(hi-lo+1));
 const tiles={},patches={N:[],S:[]},gates=[];const paintPair=[MATERIALS[(group*3+(split==='TRAIN'?0:split==='DEV'?1:2))%4],MATERIALS[(group*3+2+(split==='TRAIN'?0:split==='DEV'?1:2))%4]];
 function tile(x,y,type='floor',extra={}){tiles[key(x,y)]={x,y,type,...extra};}
 tile(0,0,'hub');tile(1,0,'protected');tile(-1,0,'protected');
 const lengths={N:rint(cfg.min,cfg.max),S:rint(cfg.min,cfg.max)};
 for(const [branch,sign] of [['N',-1],['S',1]]){
  const len=lengths[branch];
  for(let i=1;i<=len;i++)tile(0,sign*i,i>len-5?'canvas':'floor',i>len-5?{branch,index:i-(len-4)}:{branch});
  // Gates have an always-walkable side bypass; dynamic closures are never soft-locks.
  const detours=[];const candidates=Array.from({length:len-11},(_,i)=>i+5);
  while(detours.length<cfg.detours && candidates.length){
   const at=candidates.splice(rint(0,candidates.length-1),1)[0];
   if(detours.every(p=>Math.abs(p.at-at)>3))detours.push({at,offset:rng()<.5?-1:1});
  }
  detours.sort((a,b)=>a.at-b.at);
  for(let j=0;j<detours.length;j++){
   const {at,offset}=detours[j];
   tile(offset,sign*(at-1),'floor',{branch});tile(offset,sign*at,'floor',{branch});tile(offset,sign*(at+1),'floor',{branch});
   tile(0,sign*at,'gate',{branch,gate_id:`${branch}-${j}`});gates.push(key(0,sign*at));
   // Protected cladding around the bypass tests safe navigation without entering it.
   tile(offset*2,sign*at,'protected');
  }
  // Deterministically generated blind alleys: their appearance/length changes across groups.
  for(let j=0;j<5;j++){
   const at=3+j*3+rint(0,1);if(at>=len-6||detours.some(d=>Math.abs(d.at-at)<=2))continue;
   const offset=rng()<.5?-1:1,spur=rint(1,2);
   for(let dx=1;dx<=spur;dx++)if(!tiles[key(offset*dx,sign*at)])tile(offset*dx,sign*at,'floor',{branch,dead_end:true});
  }
  for(let i=len-4;i<=len;i++)patches[branch].push(key(0,sign*i));
 }
 const id=`v20-${split.toLowerCase()}-${String(group).padStart(3,'0')}`;
 const fingerprint=hash(Object.keys(tiles).sort().map(k=>[k,tiles[k].type]));
 return {id,split,group,tiles,patches,gates,palette:paintPair,lengths,fingerprint,group_difficulty:split==='TRAIN'?'D3':split==='DEV'?'D4':'D5'};
}
function makeRecord(split,group,variant){
 if(!Number.isInteger(variant)||variant<0||variant>7)throw Error('variant 0..7 required');
 const world=makeWorld(split,group),bits=[variant&1,(variant>>1)&1,(variant>>2)&1],active=(bits[0]^bits[1]^bits[2])?'S':'N';
 const events=bits.map((v,i)=>({kind:'remote_receipt',channel:['OCHRE','VIOLET','JADE'][i],payload:v,accepted:true}));
 for(let i=0;i<80;i++)events.push({kind:'neutral_receipt',channel:'CLOCK',payload:i%4,accepted:true});
 const positions=[];for(let i=0;i<80;i++)positions.push(i%4===0?'0,0':i%4===1?'0,-1':i%4===2?'0,0':'0,1');
 return {episode_id:`${world.id}-c${variant}`,split,world_id:world.id,variant,world,private_target:{active,bits,phase:bits[0]^bits[1]},initial_history:{events,positions},mode:group%3===0?'protected_refusal':'clean'};
}
function createHost(rec){
 const h={world:rec.world,target:rec.private_target,pos:'0,0',paint:{},events:rec.initial_history.events.map(x=>({...x})),positions:[...rec.initial_history.positions],accepted_moves:0,invalid:0,shutters_closed:false,done:false,stop_accepted:false};
 if(rec.mode==='protected_refusal')apply(h,'MOVE_EAST');
 return h;
}
function tileAt(h,k){const t=h.world.tiles[k];if(!t)return null;if(t.type==='gate'&&h.shutters_closed)return {...t,type:'closed_gate'};return t;}
function legalActions(h){
 if(h.done)return [];
 const [x,y]=h.pos.split(',').map(Number),moves=DIRECTIONS.filter(([,dx,dy])=>{const t=tileAt(h,key(x+dx,y+dy));return t&&t.type!=='closed_gate'&&t.type!=='protected';}).map(x=>x[0]);
 if(h.world.tiles[h.pos]?.type==='canvas')moves.push(...MATERIALS.map(x=>`PAINT_${x}`),'ERASE');
 if(h.pos==='0,0')moves.push('STOP');
 return moves;
}
function observe(h){
 const [x,y]=h.pos.split(',').map(Number),visible=[];
 for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
  const k=key(x+dx,y+dy),t=tileAt(h,k);visible.push({dx,dy,terrain:t?.type||'unknown',material:t?.type==='canvas'?(h.paint[k]||'EMPTY'):null});
 }
 return {position:h.pos,view:visible,legal_actions:legalActions(h),objective:'Authorize one remote workshop, paint its five-cell alternating motif, keep other workshop clean, return to hub and STOP',history:{events:h.events.map(x=>({...x})),positions:h.positions.slice()},completed:h.done};
}
function apply(h,action){
 if(h.done)return {accepted:false,reason:'finished'};
 let accepted=false,reason='illegal',k=h.pos;
 const move=DIRECTIONS.find(x=>x[0]===action);
 if(move){const [x,y]=h.pos.split(',').map(Number);k=key(x+move[1],y+move[2]);const t=tileAt(h,k);if(t&&t.type!=='protected'&&t.type!=='closed_gate'){h.pos=k;accepted=true;h.accepted_moves++;}}
 else if(action==='ERASE'&&h.world.tiles[h.pos]?.type==='canvas'){delete h.paint[h.pos];accepted=true;}
 else if(action.startsWith('PAINT_')&&MATERIALS.includes(action.slice(6))&&h.world.tiles[h.pos]?.type==='canvas'){h.paint[h.pos]=action.slice(6);accepted=true;}
 else if(action==='STOP'&&h.pos==='0,0'){
  accepted=true;h.stop_accepted=correctTerminal(h);h.done=true;reason=h.stop_accepted?'completed':'premature_stop';
 }
 if(!accepted){h.invalid++;if(move)reason=tileAt(h,k)?.type==='protected'?'protected':tileAt(h,k)?.type==='closed_gate'?'shutter_closed':'blocked';}
 h.positions.push(h.pos);const receipt={kind:'action_receipt',action,accepted,reason:accepted?'ok':reason};h.events.push(receipt);
 if(h.accepted_moves===8&&!h.shutters_closed){h.shutters_closed=true;h.events.push({kind:'environment_receipt',channel:'DUAL_SHUTTERS',payload:'CLOSED',accepted:true});}
 return receipt;
}
function expected(h,k){const lane=h.target.active,a=h.world.patches[lane],idx=a.indexOf(k);if(idx<0)return null;return h.world.palette[(idx+h.target.phase)%2];}
function correctTerminal(h){
 if(h.pos!=='0,0')return false;
 for(const b of ['N','S'])for(const k of h.world.patches[b]){
  const value=b===h.target.active?expected(h,k):null;
  if((h.paint[k]||null)!==value)return false;
 }return true;
}
function route(h,goal){
 const q=[h.pos],seen=new Set(q),prev=new Map();
 for(let i=0;i<q.length;i++){
  const p=q[i];if(p===goal){const actions=[];let at=goal;while(at!==h.pos){const [back,action]=prev.get(at);actions.push(action);at=back;}return actions.reverse();}
  const [x,y]=p.split(',').map(Number);
  for(const [action,dx,dy] of DIRECTIONS){const next=key(x+dx,y+dy),t=tileAt(h,next);if(!t||t.type==='protected'||t.type==='closed_gate'||seen.has(next))continue;seen.add(next);prev.set(next,[p,action]);q.push(next);}
 }throw Error(`unreachable ${h.pos} => ${goal}`);
}
function oracleAction(h){
 const patches=h.world.patches[h.target.active];
 const todo=patches.filter(k=>h.paint[k]!==expected(h,k));
 if(todo.length){
  if(todo.includes(h.pos))return h.paint[h.pos]?'ERASE':`PAINT_${expected(h,h.pos)}`;
  let best=null;for(const k of todo){const r=route(h,k);if(best===null||r.length<best.length)best=r;}
  return best[0];
 }
 return h.pos==='0,0'?'STOP':route(h,'0,0')[0];
}
function replay(rec,onDecision){
 const h=createHost(rec),initial=observe(h);let steps=0,paints=0,moves=0,invalidStart=h.invalid;
 while(!h.done){
  if(++steps>400)throw Error(`oracle did not finish ${rec.episode_id}`);
  const input=observe(h),action=oracleAction(h);if(!input.legal_actions.includes(action))throw Error('oracle action illegal '+action);
  const receipt=apply(h,action);if(!receipt.accepted)throw Error('oracle refused '+action);
  if(action.startsWith('PAINT_'))paints++;if(action.startsWith('MOVE_'))moves++;
  if(onDecision)onDecision({episode_id:rec.episode_id,step:steps-1,model_input:input,target:action,receipt});
 }
 if(!h.stop_accepted||h.invalid!==invalidStart)throw Error('non-completing oracle');
 return {episode_id:rec.episode_id,world_id:rec.world_id,split:rec.split,geometry_sha256:rec.world.fingerprint,mode:rec.mode,oracle_actions:steps,accepted_moves:moves,accepted_paints:paints,initial_rejections:invalidStart,shutter_closed:h.shutters_closed,terminal_correct:h.stop_accepted,initial_model_input:initial,provenance:'synthetic-host-v20; not native CyberSand'};
}
function projection(rec,kind='last64'){
 const x=observe(createHost(rec));if(kind==='last64'){x.history.events=x.history.events.slice(-64);x.history.positions=x.history.positions.slice(-64);}
 if(kind.startsWith('omit')){const n=Number(kind.slice(4));x.history.events=x.history.events.map((r,i)=>i===n?{...r,payload:'MASKED'}:r);}
 return hash(x);
}
function verifyFamily(split,group){
 const records=Array.from({length:8},(_,i)=>makeRecord(split,group,i));
 if(new Set(records.map(x=>projection(x))).size!==1)throw Error('last64 leak');
 for(const mask of [1,2,4,3,5,6]){const a=new Map();for(const r of records){const pattern=r.private_target.bits.filter((_,i)=>mask&(1<<i)).join('');const row=a.get(pattern)||[0,0];row[r.private_target.active==='N'?0:1]++;a.set(pattern,row);}if([...a.values()].some(([n,s])=>n!==s))throw Error('one/two-receipt leak');}
 return records;
}
function recordStream(dest,scale=1){
 if(!Number.isInteger(scale)||scale<1||scale>4)throw Error('scale must be integer 1..4');
 if(fs.existsSync(dest))throw Error('refuse overwrite '+dest);fs.mkdirSync(dest,{recursive:true});
 const shards={},summary={},worldFingerprints={},worldIds={};let totalDecisions=0;
 for(const [split,cfg] of Object.entries(SPLITS)){
  const epFile=`${split}_episodes.jsonl`,decFile=`${split}_decisions.jsonl`,ep=fs.openSync(path.join(dest,epFile),'w'),de=fs.openSync(path.join(dest,decFile),'w');
  let decisions=0,actions=0,initialRejected=0,shutters=0;
  try{
   for(let group=0;group<cfg.groups*scale;group++){
    const records=verifyFamily(split,group),w=records[0].world;
    if(worldFingerprints[w.fingerprint])throw Error('cross-group geometry collision');worldFingerprints[w.fingerprint]=w.id;worldIds[w.id]=split;
    for(const rec of records){const row=replay(rec,d=>{fs.writeSync(de,JSON.stringify(d)+'\n');decisions++;});
     fs.writeSync(ep,JSON.stringify(row)+'\n');actions+=row.oracle_actions;initialRejected+=row.initial_rejections;shutters+=Number(row.shutter_closed);
    }
   }
  }finally{fs.closeSync(ep);fs.closeSync(de);}
  shards[epFile]=fileMeta(path.join(dest,epFile),cfg.groups*scale*8);shards[decFile]=fileMeta(path.join(dest,decFile),decisions);
  summary[split]={world_groups:cfg.groups*scale,episodes:cfg.groups*scale*8,decisions,oracle_actions:actions,intentionally_refused_initial_actions:initialRejected,shutter_events:shutters};totalDecisions+=decisions;
 }
 const manifest={schema:'driv3r/corpus_factory/v20-manifest',label:'synthetic host, not native CyberSand',generator:'topology_relay_v20.js',scale,split_config:SPLITS,summaries:summary,world_fingerprint_sha256:hash(Object.entries(worldFingerprints).sort()),world_group_count:Object.keys(worldFingerprints).length,episodes:Object.values(summary).reduce((a,v)=>a+v.episodes,0),decisions:totalDecisions,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={SPLITS,MATERIALS,makeWorld,makeRecord,createHost,observe,apply,legalActions,oracleAction,replay,projection,verifyFamily,recordStream,correctTerminal};
if(require.main===module){const out=process.argv[2],scale=Number(process.argv[3]||1);if(!out)throw Error('Usage: node topology_relay_v20.js NEW_DIRECTORY [scale 1..4]');process.stdout.write(JSON.stringify(recordStream(path.resolve(out),scale).summaries)+'\n');}
