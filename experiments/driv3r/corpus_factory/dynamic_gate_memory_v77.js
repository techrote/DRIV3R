#!/usr/bin/env node
'use strict';
// DATA ONLY. This is an independently simulated graph host, NOT the CyberSand ABI.
// All trainable rows are legal physical host actions; future facts are evaluator-only.
const fs=require('node:fs'), path=require('node:path'), crypto=require('node:crypto');
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'?x:JSON.stringify(x)).digest('hex');
function assert(p,m){if(!p)throw Error('v77 '+m);}
const SPLITS={TRAIN:12,DEV:6,HELDOUT:6};
const DIRS=[['MOVE_N',0,-1],['MOVE_E',1,0],['MOVE_S',0,1],['MOVE_W',-1,0]];
function world(split,index){
 const w=10+({TRAIN:0,DEV:12,HELDOUT:18}[split])+index;
 const topGate=3+(index*3)%Math.max(1,w-5), bottomGate=3+(index*7+1)%Math.max(1,w-5);
 const cells=new Set(['0,1','-1,1']);
 for(let x=0;x<=w;x++)for(const y of [0,2])cells.add(x+','+y);
 cells.add(w+',1');
 const geometry={width:w,top_gate:[topGate,0],bottom_gate:[bottomGate,2],cells:[...cells].sort()};
 return {width:w,topGate,bottomGate,cells,geometry_sha256:sha(geometry),split,index};
}
const key=(x,y)=>x+','+y;
function adjacent(h){return DIRS.filter(([a,dx,dy])=>{
 const x=h.pos[0]+dx,y=h.pos[1]+dy,k=key(x,y);
 if(!h.w.cells.has(k))return false;
 if(y===0&&x===h.w.topGate&&!h.gates.TOP)return false;
 if(y===2&&x===h.w.bottomGate&&!h.gates.BOTTOM)return false;
 return true;
 }).map(z=>z[0]);}
function see(h){
 return {schema:'driv3r/synthetic-gate-host/1',tick:h.tick,position:h.pos.slice(),
  goal:[h.w.width,1],map_width:h.w.width,legal:[...adjacent(h),...(key(...h.pos)===key(h.w.width,1)&&!h.painted?['PAINT_A']:[]),...(key(...h.pos)===key(h.w.width,1)&&h.painted?['STOP']:[])],
  local_gate_visible:{TOP:(h.pos[1]===0&&Math.abs(h.pos[0]-h.w.topGate)<=1)?h.gates.TOP:null,
   BOTTOM:(h.pos[1]===2&&Math.abs(h.pos[0]-h.w.bottomGate)<=1)?h.gates.BOTTOM:null},
  reports:h.inbox.map(x=>({...x}))};
}
function execute(h,action){const legal=see(h).legal;assert(legal.includes(action),'illegal '+action+' at '+key(...h.pos));
 if(action.startsWith('MOVE_')){const p=DIRS.find(x=>x[0]===action);h.pos[0]+=p[1];h.pos[1]+=p[2];}
 else if(action==='PAINT_A'){assert(!h.painted,'double paint');h.painted=true;}
 else if(action==='STOP'){assert(h.painted&&key(...h.pos)===key(h.w.width,1),'premature STOP');h.done=true;}
 h.tick++;h.inbox=[];
}
function initial(w){return {w,pos:[0,1],gates:{TOP:true,BOTTOM:true},tick:0,inbox:[],painted:false,done:false};}
function snapshot(h){return {measure_tick:h.tick,TOP:h.gates.TOP?'OPEN':'CLOSED',BOTTOM:h.gates.BOTTOM?'OPEN':'CLOSED'};}
function shortest(h){ // independent of privileged evaluation labels: BFS through observed, previously reported gates
 const q=[[h.pos[0],h.pos[1],[]]],visited=new Set([key(...h.pos)]);
 for(let i=0;i<q.length;i++){
  const [x,y,steps]=q[i];if(x===h.w.width&&y===1)return steps;
  for(const [cmd,dx,dy] of DIRS){const nx=x+dx,ny=y+dy,k=key(nx,ny);
   if(!h.w.cells.has(k)||visited.has(k))continue;
   if(ny===0&&nx===h.w.topGate&&!h.gates.TOP)continue;
   if(ny===2&&nx===h.w.bottomGate&&!h.gates.BOTTOM)continue;
   visited.add(k);q.push([nx,ny,steps.concat(cmd)]);
  }
 }throw Error('unreachable');
}
function episode(w,openLane,order,offPolicy){
 assert(['TOP','BOTTOM'].includes(openLane),'lane');
 const h=initial(w),steps=[],audit=[];
 const action=(cmd,weight=0)=>{const obs=see(h);assert(obs.legal.includes(cmd),'bad '+cmd);steps.push({input:obs,action:cmd,weight,focus_loss_weight:weight});execute(h,cmd);};
 const original=snapshot(h); // original snapshot is measured at tick 0, final change at tick 10
 let fresh=null,decision=-1,recovery=-1;
 for(let i=0;i<160;i++){
  if(i===10){h.gates.TOP=openLane==='TOP';h.gates.BOTTOM=openLane==='BOTTOM';fresh=snapshot(h);audit.push({event:'PHYSICAL_REMOTE_GATE_CHANGE',tick:h.tick,gates:{...h.gates}});}
  if(i===12||i===22){const chosen=(order==='FRESH_FIRST'?(i===12?fresh:original):(i===12?original:fresh));
   assert(chosen,'missing sample');for(const lane of ['TOP','BOTTOM'])h.inbox.push({type:'REMOTE_GATE_REPORT',lane,measure_tick:chosen.measure_tick,status:chosen[lane]});
  }
  action(i%2===0?'MOVE_W':'MOVE_E');
 }
 assert(key(...h.pos)==='0,1'&&h.tick===160,'critical clock');
 const critical=see(h);assert(critical.local_gate_visible.TOP===null&&critical.local_gate_visible.BOTTOM===null,'current leak');
 decision=steps.length;
 const optimal=shortest(h)[0];assert(optimal===(openLane==='TOP'?'MOVE_N':'MOVE_S'),'not top or bottom');
 if(offPolicy){
  // Force a legal but wrong route: host physically stops progression at the closed gate.
  const wrong=openLane==='TOP'?'MOVE_S':'MOVE_N';action(wrong,0);
  const gpos=openLane==='TOP'?w.bottomGate:w.topGate;
  for(let x=0;x<gpos-1;x++)action('MOVE_E',0);
  const blocked=see(h);assert(!blocked.legal.includes('MOVE_E'),'wrong lane should be blocked');
  recovery=steps.length;
 }
 for(const move of shortest(h))action(move,1);
 action('PAINT_A',1);action('STOP',1);
 assert(h.done&&h.painted,'nonterminal');
 return {episode:{steps},certificate:{schema:'driv3r/corpus_factory/dynamic-gate-v77-cert/1',world_index:w.index,
  split:w.split,geometry_sha256:w.geometry_sha256,open_lane:openLane,receipt_order:order,off_policy:offPolicy,
  decision_step:decision,recovery_step:recovery,first_optimal_action:optimal,
  current_input_sha256:sha(critical),history_report_sha256:sha(steps.flatMap(s=>s.input.reports)),
  environmental_events:audit,total_actions:steps.length,terminal_success:true,
  episode_sha256:sha(steps),synthetic_not_native_CyberSand:true}};
}
function write(out,quick=false){
 assert(!fs.existsSync(out),'output exists');fs.mkdirSync(out,{recursive:true});
 const manifest={schema:'driv3r/corpus_factory/dynamic-gate-memory-v77/1',date:'2026-11-19',
  synthetic_not_native_CyberSand:true,source_sha256:sha(fs.readFileSync(__filename,'utf8')),
  training_split_contract:'TRAIN streams only; DEV/HELDOUT and all certificates evaluator-only; off-policy wrong actions loss-masked',
  report_semantics:'Source-time stamps, out-of-order delivery, physical remote gate changes, accepted host movement and painting',
  files:{},splits:{}};
 const names=['TRAIN','DEV','HELDOUT'], seenGeometry=new Set();
 for(const split of names){
  const episodes=[],certs=[],n=quick?1:SPLITS[split],s={worlds:[],episodes:0,terminal_successes:0,legal_actions:0,paired_current_observation_groups:0,paired_conflicting_correct_actions:0,offpolicy_recoveries:0};
  for(let idx=0;idx<n;idx++){
   const w=world(split,idx);assert(!seenGeometry.has(w.geometry_sha256),'geometry collision');seenGeometry.add(w.geometry_sha256);
   s.worlds.push({world_index:idx,width:w.width,geometry_sha256:w.geometry_sha256,top_gate:w.topGate,bottom_gate:w.bottomGate});
   const actions=new Set(),currentHashes=new Set();
   for(const lane of ['TOP','BOTTOM'])for(const order of ['FRESH_FIRST','STALE_FIRST'])for(const off of [false,true]){
    const {episode:e,certificate:c}=episode(w,lane,order,off);
    c.episode=s.episodes;episodes.push(e);certs.push(c);
    s.episodes++;s.terminal_successes++;s.legal_actions+=c.total_actions;
    if(off)s.offpolicy_recoveries++;
    if(!off){actions.add(c.first_optimal_action);currentHashes.add(c.current_input_sha256);}
   }
   assert(actions.size===2&&currentHashes.size===1,'matched ambiguous current state');
   s.paired_current_observation_groups++;s.paired_conflicting_correct_actions++;
  }
  const rows=[['$S_gate_memory_stream.jsonl',episodes],['$S_gate_memory_certificates_EVALUATOR_ONLY.jsonl',certs]];
  for(const [template,items] of rows){const name=template.replace('$S',split==='TRAIN'?'TRAIN':split+'_EVALUATOR_ONLY');
   const bytes=items.map(x=>JSON.stringify(x)+'\n').join('');fs.writeFileSync(path.join(out,name),bytes,{flag:'wx'});
   manifest.files[name]={rows:items.length,bytes:Buffer.byteLength(bytes),sha256:sha(bytes)};
  }
  manifest.splits[split]=s;
 }
 fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(require.main===module){assert(process.argv[2],'usage node dynamic_gate_memory_v77.js NEW_DIR [--quick]');
 const m=write(path.resolve(process.argv[2]),process.argv.includes('--quick'));
 console.log(JSON.stringify(Object.fromEntries(Object.entries(m.splits).map(([k,v])=>[k,{worlds:v.worlds.length,episodes:v.episodes,actions:v.legal_actions}]))));}
module.exports={sha,world,initial,see,execute,episode,write,SPLITS,DIRS};
