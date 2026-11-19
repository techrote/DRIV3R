"use strict";
// Opt-in CPU synthetic research. No import-time optimization and no CyberSand host ABI claim.
const study=function study(mode) {
  "use strict";
  const started=Date.now();
  const MODES={
    perfect:{s:1,fp:0,loss:0,delay:0},
    high:{s:.98,fp:.01,loss:.01,delay:1},
    middle:{s:.90,fp:.04,loss:.04,delay:2},
    poor:{s:.70,fp:.10,loss:.10,delay:4}
  };
  if(!Object.hasOwn(MODES,mode))throw Error("invalid observation mode");
  const cfg=MODES[mode], DIR={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]};
  const ACTIONS=["N","E","S","W","PAINT","ERASE","STOP"], CAPS=[[36,36],[72,64],[128,64],[128,128]];
  const TARGETS=[[2,1],[9,1],[9,6],[2,6],[9,3]];
  const POLICIES=["never","always","warning","trained","oracle"];
  const key=p=>p.join(","),eq=(a,b)=>a[0]===b[0]&&a[1]===b[1];
  function noise(s,tag){let v=(Math.imul(s,1664525)+1013904223)>>>0;for(let c of tag)v=Math.imul(v^c.charCodeAt(0),16777619)>>>0;return v/4294967296}
  function digest(s){let v=2166136261;for(let c of s)v=Math.imul(v^c.charCodeAt(0),16777619);return(v>>>0).toString(16)}
  function blocked(seed,split){let b=new Set();const gaps=[1,3,5];let a=gaps[Math.floor(noise(seed,"gapA")*3)],c=gaps[Math.floor(noise(seed,"gapB")*3)];for(let y=0;y<8;y++){if(y!==a)b.add("4,"+y);if(y!==c)b.add("7,"+y)}if(split==="dev"&&seed%5===0)b.add("6,0");return b}
  function route(start,goal,b){
    let q=[start],seen=new Set([key(start)]),prev=new Map();
    for(let i=0;i<q.length;i++){
      if(eq(q[i],goal))break;
      for(let [a,d] of Object.entries(DIR)){let p=[q[i][0]+d[0],q[i][1]+d[1]],k=key(p);if(p[0]<0||p[0]>=11||p[1]<0||p[1]>=8||b.has(k)||seen.has(k))continue;seen.add(k);prev.set(k,[key(q[i]),a]);q.push(p)}
    }
    let k=key(goal),res=[];if(!seen.has(k))return res;
    while(k!==key(start)){let p=prev.get(k);res.push(p[1]);k=p[0]}return res.reverse();
  }
  class Host {
    constructor(seed,horizon,split){
      Object.assign(this,{seed,horizon,split,pos:[1,1],walls:blocked(seed,split),ink:{},tick:0,log:[],msg:[],pending:[],calls:0,back:0,first:-1,erased:false,event:false,gate:null,gateUntil:0,fault:1+Math.floor(noise(seed,"fault")*2),faulted:false,targets:{36:2,72:3,128:5}[horizon]});
    }
    barriers(){let b=new Set(this.walls);if(this.gate&&this.tick<this.gateUntil)b.add(key(this.gate));return b}
    visible(){return this.ink[key(this.pos)]??null}
    menu(){let b=this.barriers(),m=[];for(let [a,d] of Object.entries(DIR)){let p=[this.pos[0]+d[0],this.pos[1]+d[1]];if(p[0]>=0&&p[0]<11&&p[1]>=0&&p[1]<8&&!b.has(key(p)))m.push(a)}if(!b.has(key(this.pos)))m.push(this.visible()===null?"PAINT":"ERASE");m.push("STOP");return m}
    step(a){
      let menu=this.menu();if(!menu.includes(a))throw Error("illegal "+a);
      let before=[...this.pos],old=this.visible(),result="ok";
      if(a in DIR){let d=DIR[a];this.pos=[this.pos[0]+d[0],this.pos[1]+d[1]]}
      else if(a==="PAINT"){let ix=TARGETS.findIndex(t=>eq(t,this.pos)),bad=ix===this.fault&&!this.faulted;if(bad)this.faulted=true;this.ink[key(this.pos)]=bad?2:1;result=bad?"wrong_material":"painted";if(ix===0&&this.first<0&&!bad)this.first=this.tick}
      else if(a==="ERASE"){delete this.ink[key(this.pos)];result="erased"}else result="stopped";
      let r={tick:this.tick,action:a,menu,before,after:[...this.pos],before_material:old,after_material:this.visible(),result};
      this.log.push(r);this.tick++;
      if(!this.event&&this.first>=0&&this.tick>=Math.max(7,Math.floor(this.horizon/4))){
        this.event=true;this.erased=noise(this.seed,"hidden_damage")<(this.split==="dev"?.30:.16);
        if(this.erased)delete this.ink[key(TARGETS[0])];
        let alert=noise(this.seed,"channel_alert")<(this.erased?cfg.s:cfg.fp);
        let lost=noise(this.seed,"channel_loss")<cfg.loss;
        if(!lost){let lag=cfg.delay===0?0:Math.floor(noise(this.seed,"channel_delay")*(cfg.delay+1));this.pending.push({at:this.tick+lag,kind:alert?"warning":"epoch",origin:this.tick})}
      }
      if(this.tick===Math.floor(this.horizon/3)){
        let cell=[5,[1,3,5][Math.floor(noise(this.seed,"gate")*3)]];
        if(!this.walls.has(key(cell))&&!eq(this.pos,cell)){this.gate=cell;this.gateUntil=this.tick+3+Math.floor(noise(this.seed,"gate_duration")*3)}
      }
      for(let m of this.pending.filter(m=>m.at<=this.tick))this.msg.push({kind:m.kind,seen:this.tick,origin:m.origin});
      this.pending=this.pending.filter(m=>m.at>this.tick);return r;
    }
    clone(){
      let h=new Host(this.seed,this.horizon,this.split);Object.assign(h,this,{pos:[...this.pos],walls:new Set(this.walls),ink:{...this.ink},log:this.log.map(r=>({...r,menu:[...r.menu],before:[...r.before],after:[...r.after]})),msg:this.msg.map(x=>({...x})),pending:this.pending.map(x=>({...x})),gate:this.gate?[...this.gate]:null});return h
    }
    result(cap){let ok=TARGETS.slice(0,this.targets).every(t=>this.ink[key(t)]===1),counts=Object.fromEntries(ACTIONS.map(a=>[a,0]));for(let r of this.log)counts[r.action]++;return{goal:+ok,utility:(ok?1000:0)-2*this.tick-3*this.calls-this.back,ticks:this.tick,calls:this.calls,counts,cap,positions:this.log.map(r=>r.after),hash:digest(JSON.stringify(this.log))}}
  }
  function go(h,dest,cap){let path=[],target="";while(!eq(h.pos,dest)&&h.tick<cap){if(target!==key(dest)||!path.length||!h.menu().includes(path[0])){path=route(h.pos,dest,h.barriers());h.calls++;target=key(dest)}if(!path.length)return false;let a=path.shift();if(a==="N"||a==="W")h.back++;h.step(a)}return eq(h.pos,dest)}
  function paint(h,cap){if(h.visible()===2&&h.tick<cap)h.step("ERASE");if(h.visible()===null&&h.tick<cap)h.step("PAINT");if(h.visible()===2&&h.tick<cap){h.step("ERASE");if(h.tick<cap)h.step("PAINT")}return h.visible()===1}
  function prefix(seed,hor,split,cap){let h=new Host(seed,hor,split);for(let p of TARGETS.slice(0,h.targets)){if(h.tick>=cap||!go(h,p,cap)||!paint(h,cap))break}return h}
  function finish(root,cap,rev){let h=root.clone();if(rev&&h.tick<cap&&go(h,TARGETS[0],cap))paint(h,cap);if(h.tick<cap)h.step("STOP");return{host:h,out:h.result(cap)}}
  function features(h,cap){let warn=h.msg.filter(m=>m.kind==="warning"),epoch=h.msg.filter(m=>m.kind==="epoch");return[1,+!!warn.length,+!!epoch.length,Math.min(1,(h.tick-(warn.at(-1)?.seen??h.tick))/30),Math.min(1,(h.tick-Math.max(0,h.first))/128),(cap-h.tick)/128,Math.min(1,route(h.pos,TARGETS[0],h.barriers()).length/30),+(h.horizon===36),+(h.horizon===128)]}
  function predict(w,x){let z=w.reduce((s,v,i)=>s+v*x[i],0);return 1/(1+Math.exp(-Math.max(-25,Math.min(25,z))))}
  function fit(rows){let w=Array(9).fill(0);for(let t=0;t<850;t++){let grad=Array(9).fill(0);for(let r of rows){let error=predict(w,r.x)-r.erased;for(let i=0;i<9;i++)grad[i]+=error*r.x[i]}for(let i=0;i<9;i++)w[i]-=.7/(1+t/1200)*(grad[i]/rows.length+.0008*w[i])}return w}
  function assertReceipt(log){for(let i=0;i<log.length;i++){let r=log[i];if(r.tick!==i||!r.menu.includes(r.action))throw Error("receipt tick/menu");if(i&&!eq(log[i-1].after,r.before))throw Error("receipt continuity");if(r.action in DIR){let d=DIR[r.action];if(!eq(r.after,[r.before[0]+d[0],r.before[1]+d[1]]))throw Error("receipt move")}else if(!eq(r.after,r.before))throw Error("receipt nonmove");if(r.action==="ERASE"&&r.after_material!==null)throw Error("receipt erase");if(r.action==="PAINT"&&![1,2].includes(r.after_material))throw Error("receipt paint")}return log.length}
  function replay(h){let n=assertReceipt(h.log),r=new Host(h.seed,h.horizon,h.split);for(let receipt of h.log){let fresh=r.step(receipt.action);if(JSON.stringify(fresh)!==JSON.stringify(receipt))throw Error("same-host replay")}if(JSON.stringify(r.ink)!==JSON.stringify(h.ink)||JSON.stringify(r.msg)!==JSON.stringify(h.msg)||!eq(r.pos,h.pos))throw Error("state mismatch");return n}
  let training=[];for(let seed=12100;seed<12170;seed++)for(let [hor,cap] of CAPS){let h=prefix(seed,hor,"train",cap),b=finish(h,cap,false).out,a=finish(h,cap,true).out;training.push({x:features(h,cap),erased:+h.erased,yes:a.utility,no:b.utility})}
  let weights=fit(training),cuts=[0,.005,.01,.02,.05,.1,.2,.35,.5,.7,1.01],frontier=cuts.map(c=>({cutoff:c,utility:training.reduce((s,r)=>s+(predict(weights,r.x)>=c?r.yes:r.no),0)/training.length})).sort((a,b)=>b.utility-a.utility||a.cutoff-b.cutoff),chosen=frontier[0].cutoff;
  const rows=[],paths=[];let replayed=0;for(let seed=15100;seed<15160;seed++)for(let [hor,cap] of CAPS){
    let root=prefix(seed,hor,"dev",cap),obs=features(root,cap),pr=predict(weights,obs),no=finish(root,cap,false).out,yes=finish(root,cap,true).out,warning=root.msg.some(m=>m.kind==="warning");
    for(let policy of POLICIES){let revisit=policy==="always"?true:policy==="never"?false:policy==="warning"?warning:policy==="trained"?pr>=chosen:yes.utility>no.utility;let r=finish(root,cap,revisit);replayed+=replay(r.host);if(r.out.hash!==(revisit?yes:no).hash)throw Error("paired fork failed");rows.push({seed,hor,cap,policy,revisit,prob:+pr.toFixed(6),warning,actual_erasure_for_audit:root.erased,...r.out,positions:undefined});if((seed===15104||seed===15121||seed===15139)&&["always","trained","never"].includes(policy))paths.push({seed,hor,cap,policy,goal:r.out.goal,positions:r.out.positions})}
  }
  let corrupted=new Host(12100,36,"train");corrupted.step("E");let invalidAction=[{...corrupted.log[0],action:"TELEPORT"}],invalidCoord=[{...corrupted.log[0],after:[1000,1000]}],rejects=0;for(let xs of [invalidAction,invalidCoord]){try{assertReceipt(xs)}catch(e){rejects++}}if(rejects!==2||rows.length!==1200||training.length!==280||replayed<10000)throw Error("study qualification failed");
  let summary={},cohorts={},paired={};for(let p of POLICIES){let a=rows.filter(r=>r.policy===p),actions=Object.fromEntries(ACTIONS.map(x=>[x,a.reduce((s,r)=>s+r.counts[x],0)]));summary[p]={goals:a.reduce((s,r)=>s+r.goal,0),episodes:a.length,mean_utility:+(a.reduce((s,r)=>s+r.utility,0)/a.length).toFixed(4),planners:a.reduce((s,r)=>s+r.calls,0),revisits:a.filter(r=>r.revisit).length,actions};let losses=[],wins=[];for(let r of a){let b=rows.find(z=>z.policy==="always"&&z.seed===r.seed&&z.hor===r.hor&&z.cap===r.cap),delta=r.utility-b.utility;if(delta<0)losses.push({seed:r.seed,hor:r.hor,cap:r.cap,delta,goal:r.goal,baseline_goal:b.goal,hash:r.hash,erased:r.actual_erasure_for_audit,warning:r.warning});else if(delta>0)wins.push({seed:r.seed,hor:r.hor,cap:r.cap,delta,hash:r.hash})}paired[p]={losses:losses.length,wins:wins.length,ties:a.length-losses.length-wins.length,negative_cases:losses,positive_examples:wins.slice(0,12)}}
  for(let [hor,cap] of CAPS)cohorts[hor+"/"+cap]=Object.fromEntries(POLICIES.map(p=>[p,rows.filter(r=>r.hor===hor&&r.cap===cap&&r.policy===p).reduce((s,r)=>s+r.goal,0)]));
  let e={study:"SIGNAL-FRONTIER-012",mode,channel:cfg,date:"2026-11-19",game_host:"independent synthetic material/movement/painting host; NOT CyberSand native ABI",worlds:{train_seeds:[12100,12169],development_seeds:[15100,15159],train_damage_prior:.16,development_damage_prior:.30},caps:CAPS,heldout_opened:false,training:{updates:850,rows:training.length,weights:weights.map(v=>+v.toFixed(12)),brier:+(training.reduce((s,r)=>s+(predict(weights,r.x)-r.erased)**2,0)/training.length).toFixed(7),selected_cutoff:chosen,frontier},development:{paired_tasks:240,rollouts:rows.length,receipt_steps_replayed:replayed,summary,cohorts,paired,per_episode:rows.map(r=>({seed:r.seed,hor:r.hor,cap:r.cap,policy:r.policy,revisit:r.revisit,prob:r.prob,warning:r.warning,erased_for_audit:r.actual_erasure_for_audit,goal:r.goal,utility:r.utility,ticks:r.ticks,calls:r.calls,hash:r.hash})),paths},corrupted_action_and_coordinate_rejected:rejects===2,elapsed_v8_wall_ms:Date.now()-started,limitations:["Observation channels are hypothetical and not official CyberSand ABI.","Same-host deterministic replay with separate receipt algebra; not native or independently implemented scientific replay.","128/64 is a synthetic native-cap proxy; 128/128 is separate shadow.","The V8 campaign reports whole-study wall time, not process RSS or per-decision latency.","Text weights are not deployable checkpoint binaries.","FNV32 receipt hash is equality aid, not cryptographic provenance."]};
  return e;
};
const gallery=function gallery(results){
  const cases=[];
  for(let e of results){let all=e.development.paths;for(let x of all.filter(p=>p.hor===72)){cases.push({mode:e.mode,...x})}}
  const height=cases.length*98+30;
  let o=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="'+height+'" viewBox="0 0 960 '+height+'">','<rect width="100%" height="100%" fill="#101e2a"/>','<text x="20" y="20" fill="white" font-size="14">SIGNAL-FRONTIER-012 — matched synthetic 72-decision trajectories (not native)</text>'];
  for(let i=0;i<cases.length;i++){let z=cases[i],y=34+i*98,colors={always:"#6bd4ae",never:"#eea56a",trained:"#b1a4ff"};o.push('<polyline fill="none" stroke="'+colors[z.policy]+'" stroke-width="1.8" points="'+z.positions.map(p=>(25+22*p[0])+","+(y+8*p[1])).join(" ")+'"/>','<text x="335" y="'+(y+20)+'" fill="#e7edf4" font-size="12" font-family="monospace">'+z.mode+' / '+z.policy+' seed='+z.seed+' '+z.hor+'/'+z.cap+' goal='+z.goal+'</text>')}
  o.push("</svg>");return o.join("\n")+"\n";
};
if(typeof module!=="undefined")module.exports={study,gallery};
if(typeof require!=="undefined"&&require.main===module){const fs=require("fs"),path=require("path"),crypto=require("crypto");if(process.argv.length!==3||!path.isAbsolute(process.argv[2]))throw Error("Usage: node signal_frontier_012.js /absolute/output/directory (explicit CPU run)");let dir=process.argv[2],modes=["perfect","high","middle","poor"],res=modes.map(study);for(let r of res){r.node_rss_at_end_bytes=process.memoryUsage().rss;r.source_sha256=crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex")}fs.mkdirSync(dir,{recursive:true});for(let r of res)fs.writeFileSync(path.join(dir,"SIGNAL_FRONTIER_012_"+r.mode+".json"),JSON.stringify(r,null,2)+"\n");fs.writeFileSync(path.join(dir,"SIGNAL_FRONTIER_012_PATHS.svg"),gallery(res));console.log(JSON.stringify(res.map(r=>({mode:r.mode,summary:r.development.summary,elapsed_ms:r.elapsed_v8_wall_ms,node_rss_at_end_bytes:r.node_rss_at_end_bytes})),null,2));}
