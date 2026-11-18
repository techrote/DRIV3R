'use strict';
// Corpus Factory v38: host-accepted sequential material-law flips, not native CyberSand.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const sha=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
const assert=(b,m)=>{if(!b)throw Error(m);};
const clone=x=>JSON.parse(JSON.stringify(x));
const SPLITS={TRAIN:4,DEV:2,HELDOUT:2}, MODES=['clean','bad-nav-and-paint'];
const swap=a=>a==='PAINT_A'?'PAINT_B':a==='PAINT_B'?'PAINT_A':a;
const physical=(command,inverted)=>inverted?swap(command):command;
function lawInput(obs,inverted,changes){const input=view.compress(obs);input.law={kind:'public_live_tool_mapping',paint_A_yields:inverted?'B':'A',paint_B_yields:inverted?'A':'B',change_count:changes};return input;}
function writer(dest,name,items){const buf=Buffer.from(items.map(x=>JSON.stringify(x)).join('\n')+'\n');fs.writeFileSync(path.join(dest,name),buf,{flag:'wx'});return {rows:items.length,bytes:buf.length,sha256:sha(buf)};}
function simulate(w,planner,split,g,v,arm,lawInitiallyInverted,mode,pair){
 const h=base.prepare(w,split,g,v,arm).h;h.remaining=1200;
 let inverted=lawInitiallyInverted,changes=0,erases=0,paintsWhileFlipped=0,steps=0,offpolicy=null;
 let firstErase=false,restored=false;const records=[],states=[],start=planner.rank([h.pos,h.prev,h.status,h.latched])[0];
 assert(start.action===pair.first[arm].action,'pair label drift');
 let wrongNav=null;
 if(mode==='bad-nav-and-paint'){
   const rank=planner.rank([h.pos,h.prev,h.status,h.latched]);
   const others=rank.filter(x=>['MOVE_W','MOVE_E','MOVE_S','BACKTRACK'].includes(x.action)&&Number.isFinite(x.cost)&&x.cost-start.cost>=2).sort((a,b)=>a.action.localeCompare(b.action));
   assert(others.length>=2,'insufficient bad nav choices');const selected=others[(g+v+arm)%others.length];
   const ack=host.apply(h,selected.action);assert(ack.accepted,'bad nav not accepted');
   wrongNav={action:selected.action,regret:selected.cost-start.cost,accepted:true};
 }
 while(!h.done){
   assert(steps<200,'bounded recovery exceeded');
   const rank=planner.rank([h.pos,h.prev,h.status,h.latched]),r=rank[0],before=host.observe(h);
   assert(r&&Number.isFinite(r.cost),'unreachable suffix');
   const legal=before.legal_actions,command=physical(r.action,inverted);
   assert(legal.includes(command),'law-mapped command not legal');
   const input=lawInput(before,inverted,changes),stateKey=sha([h.pos,h.prev,h.status,h.latched]);
   const step={input,action:command,weight:1};
   let appliedPhysical=r.action,appliedCommand=command;
   if(mode==='bad-nav-and-paint'&&!offpolicy&&r.action.startsWith('PAINT_')){
     appliedPhysical=swap(r.action);appliedCommand=physical(appliedPhysical,inverted);
     const bad=rank.find(x=>x.action===appliedPhysical);
     assert(bad&&Number.isFinite(bad.cost)&&bad.cost-r.cost>=2,'paint intervention not detrimental');
     offpolicy={index:steps,expected:command,accepted_command:appliedCommand,strict_regret:bad.cost-r.cost};
     step.weight=0;
   }
   const receipt=host.apply(h,appliedPhysical);assert(receipt.accepted,'mapped action refused');
   // The underlying symbolic Host represents the *physical effect*. The public
   // tool receipt must instead report the command actually issued under the law.
   if(appliedPhysical.startsWith('PAINT_')){
      receipt.action=appliedCommand;
      receipt.material_result=appliedPhysical.slice(6);
      receipt.tool_mapping=inverted?'inverted':'normal';
      if(inverted)paintsWhileFlipped++;
   }
   // A real host-triggered change: first accepted corrective ERASE flips the
   // tool law; first subsequent accepted PAINT restores the initial law.
   if(appliedPhysical==='ERASE'){
      erases++;
      if(!firstErase){firstErase=true;inverted=!inverted;changes++;
        h.history.events.push({kind:'TOOL_LAW_CHANGED',accepted:true,at:w.nodes[h.pos],reason:'FIRST_CORRECTIVE_ERASE',paint_A_yields:inverted?'B':'A',paint_B_yields:inverted?'A':'B'});
      }
   }else if(appliedPhysical.startsWith('PAINT_')&&firstErase&&!restored){
      restored=true;inverted=!inverted;changes++;
      h.history.events.push({kind:'TOOL_LAW_CHANGED',accepted:true,at:w.nodes[h.pos],reason:'FIRST_POST_ERASE_PAINT',paint_A_yields:inverted?'B':'A',paint_B_yields:inverted?'A':'B'});
   }
   const after= h.done?0:planner.rank([h.pos,h.prev,h.status,h.latched])[0].cost;
   const claimed=rank.find(x=>x.action===appliedPhysical)?.cost;
   assert(claimed!==undefined&&claimed-1===after,'oracle state successor mismatch');
   records.push(step);states.push([stateKey,appliedPhysical,appliedCommand,sha([h.pos,h.prev,h.status,h.latched])]);steps++;
 }
 assert(firstErase&&restored&&changes===2&&paintsWhileFlipped>0,'no effective sequential law change');
 assert(host.correct(w,h.status).wrong===0&&host.correct(w,h.status).count>=2&&erases>0,'terminal contamination');
 assert((mode==='bad-nav-and-paint')===Boolean(offpolicy),'missing bad paint');
 return {records,states,summary:{split,world:w.geometry_sha256,arm,mode,law_initial:lawInitiallyInverted?'inverted':'normal',steps,
   accepted_bad_nav:wrongNav,accepted_bad_paint:offpolicy,erases,paints_while_inverted:paintsWhileFlipped,law_changes:changes,terminal_success:h.done}};
}
function generate(dest,{quick=false}={}){
 assert(!fs.existsSync(dest),'refuse overwriting output');fs.mkdirSync(dest,{recursive:true});
 const shards={},splits={},geometryOwners=new Map(),inputOwners=new Map();
 for(const [split,total] of Object.entries(SPLITS)){
   const episodes=[],rows=[],seen=new Map(),qualified=[],rejected=[],memoryBuckets=new Map();
   let positive=0,contrasts=0,duplicates=0,zero=0,lawFlips=0,invertedPaint=0;
   for(let g=0;g<(quick?1:total);g++)for(let v=0;v<8;v++){
     const w=base.world(split,g,v),planner=base.oracle(w),pair=base.qualify(w,planner,split,g,v);
     if(!pair){rejected.push([g,v]);continue;}
     const owner=geometryOwners.get(w.geometry_sha256);assert(!owner||owner===split,'geometry leakage');geometryOwners.set(w.geometry_sha256,split);
     const paired=[];
     for(let arm=0;arm<4;arm++)for(const mode of MODES){
       const a=simulate(w,planner,split,g,v,arm,false,mode,pair),b=simulate(w,planner,split,g,v,arm,true,mode,pair);
       assert(sha(a.states.map(s=>[s[0],s[1],s[3]]))===sha(b.states.map(s=>[s[0],s[1],s[3]])),'paired physical trajectories diverged');
       assert(a.records.length===b.records.length,'paired lengths differ');
       let different=0;
       for(let i=0;i<a.records.length;i++)if(a.records[i].action!==b.records[i].action){
         assert(a.records[i].action.startsWith('PAINT_')&&b.records[i].action.startsWith('PAINT_'),'nonpaint action diverged');different++;
       }
       assert(different>0,'no effective command-law contrast');positive+=different;contrasts++;
       if(mode==='clean'){for(const item of [a,b]){const key=[g,v,item.summary.law_initial].join(':');const group=memoryBuckets.get(key)||[];group.push({input:clone(item.records[0].input),action:item.records[0].action});memoryBuckets.set(key,group);}}
       for(const item of [a,b]){
         for(const row of item.records){const k=sha(row.input),prev=seen.get(k),other=inputOwners.get(k);
           assert(!other||other===split,'cross-split model-input leakage');inputOwners.set(k,split);
           assert(prev===undefined||prev===row.action,'conflicting label');
           if(prev!==undefined&&split==='TRAIN'){row.weight=0;duplicates++;}else seen.set(k,row.action);
           if(!row.weight)zero++;rows.push(row);
         }
         lawFlips+=item.summary.law_changes;invertedPaint+=item.summary.paints_while_inverted;
         episodes.push({...item.summary,g,v});
       }
     }
     qualified.push({geometry:w.geometry_sha256,g,v,paired_contexts:8,paired_law_modes:2});
   }
   assert(qualified.length>0,'no qualifying families');
   const memory=[];
   for(const [family,items] of memoryBuckets){
     assert(items.length===4,'incomplete four-way memory '+family);
     assert(new Set(items.map(x=>x.action)).size===4,'four-way labels collapsed');
     assert(new Set(items.map(x=>sha(x.input))).size===4,'durable histories not distinctive');
     const noMemory=items.map(x=>({...x.input,durable:[]}));
     assert(new Set(noMemory.map(sha)).size===1,'recent-only four-way challenge leaked');
     for(let i=0;i<4;i++)memory.push({input:items[i].input,historyless_input:noMemory[i],action:items[i].action});
   }
   shards[split+'_policy.jsonl']=writer(dest,split+'_policy.jsonl',rows);
   if(split!=='TRAIN'){shards[split+'_challenge.jsonl']=writer(dest,split+'_challenge.jsonl',episodes);
     shards[split+'_memory.jsonl']=writer(dest,split+'_memory.jsonl',memory);}
   splits[split]={families:qualified.length,accepted_episodes:episodes.length,decision_rows:rows.length,paired_trajectories:contrasts,
      paired_paint_command_changes:positive,four_way_memory_families:memoryBuckets.size,four_way_first_action_ceiling:0.25,law_transitions:lawFlips,paint_actions_while_inverted:invertedPaint,zero_weight:zero,
      exact_train_duplicates:duplicates,rejections:rejected,geometries:qualified.map(x=>x.geometry)};
 }
 const source=fs.readFileSync(__filename),deps=['causal_shortcut_v32.js','causal_gate_v26.js','outcome_forks_v25.js','bounded_view_v30.js'];
 const manifest={schema:'driv3r/corpus_factory/host-live-law-v38/manifest',synthetic_not_native_CyberSand:true,
   generator:'closed_loop_law_v38.js',generator_sha256:sha(source),dependencies:deps.map(file=>({file,sha256:sha(fs.readFileSync(path.join(__dirname,file)))})),
   host_invariant:'physical effect submitted via existing synthetic host.apply, tool receipt records commanded token and actual material; public law switches on accepted ERASE and then on the first later PAINT',
   trainer_contract:'Only row.input is model-visible. row.action is teacher-only. weight=0 excluded. DEV/HELDOUT challenge metadata is evaluation only.',
   caveats:['Synthetic host, no native CyberSand material law','Intervention schedule is deterministic, not autonomous model control','No actual SLM weights or neural latency/RSS','No dynamic obstruction or new graph grammar'],quick,splits,shards};
 fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,simulate,lawInput,swap,physical,sha};
if(require.main===module){const [output,flag]=process.argv.slice(2);assert(output,'usage: node closed_loop_law_v38.js OUT [--quick]');
 console.log(JSON.stringify(generate(path.resolve(output),{quick:flag==='--quick'}).splits,null,2));}
