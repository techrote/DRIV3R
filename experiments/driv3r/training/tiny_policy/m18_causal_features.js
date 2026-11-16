/* M18 Tiny Game Policy Trainer. Node 20+, dependency-free. */
const runM18 = function M18Experiment(){
"use strict";
// M18, CPU-only deterministic supervised policies. No native/production writes.
const SZ=20, LIM=80, DIR=[[0,-1],[0,1],[-1,0],[1,0]], FAMS=["Greebles","Pipework","Architecture"],
 SPLIT={train:[214000,214018],dev:[215000,215004],eval:[216000,216009]};
const id=(x,y)=>y*SZ+x, inB=(x,y)=>x>=0&&y>=0&&x<SZ&&y<SZ;
function random(seed){let x=seed>>>0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}
function hash(s){let h=2166136261;for(const c of s)h=Math.imul(h^c.charCodeAt(0),16777619);return (h>>>0).toString(16).padStart(8,"0");}
class World{
 constructor(seed,fam,split){
  if(!FAMS.includes(fam)||seed<SPLIT[split][0]||seed>=SPLIT[split][1])throw Error("invalid world");
  this.seed=seed;this.fam=fam;this.split=split;this.wall=new Uint8Array(400);this.prot=new Uint8Array(400);
  this.paint=new Uint8Array(400);this.x=10;this.y=10;this.g=0;this.t=0;this.heading=3;this.path=[[10,10]];this.retraces=0;
  let rnd=random((seed*37+13)>>>0), v=seed%3;
  for(let y=0;y<20;y++)for(let x=0;x<20;x++)if(x<2||x>17||y<2||y>17)this.wall[id(x,y)]=1;
  if(split==="eval"){
   if(v===0)for(const [lo,hi] of [[3,16],[6,13]]){for(let i=lo;i<=hi;i++){this.wall[id(i,lo)]=1;this.wall[id(i,hi)]=1;this.wall[id(lo,i)]=1;this.wall[id(hi,i)]=1;}
    for(let j=lo+1;j<lo+4;j++)this.wall[id(j,lo)]=0;for(let j=hi-4;j<hi-1;j++)this.wall[id(hi,j)]=0;}
   if(v===1)for(let k=0;k<4;k++){let y=[4,7,13,16][k];for(let x=3;x<=16;x++)this.wall[id(x,y)]=1;let gap=3+(seed+3*k)%11;for(let x=gap;x<gap+3;x++)this.wall[id(x,y)]=0;}
   if(v===2)for(let k=0;k<4;k++){let x=[4,7,13,16][k];for(let y=(k%2?7:3);y<=(k%2?16:13);y++)this.wall[id(x,y)]=1;this.wall[id(x,k%2?9:11)]=0;}
  }else{
   if(v===0)for(let x of [5,9,14]){for(let y=3;y<17;y++)this.wall[id(x,y)]=1;let gap=3+(seed+x)%11;for(let y=gap;y<gap+3;y++)this.wall[id(x,y)]=0;}
   if(v===1)for(let y of [5,8,13,16]){for(let x=3;x<17;x++)this.wall[id(x,y)]=1;let gap=3+(seed+2*y)%11;for(let x=gap;x<gap+3;x++)this.wall[id(x,y)]=0;}
   if(v===2)for(let k=0;k<4;k++){let x=3+4*k;for(let y=3;y<17;y++)if((y+2*k)%7!==0)this.wall[id(x,y)]=1;}
   for(let k=0;k<6;k++){let x=3+(rnd()*14|0),y=3+(rnd()*14|0);if(Math.abs(x-10)+Math.abs(y-10)>5)this.wall[id(x,y)]=1;}
  }
  for(let k=0;k<6;k++){let x=3+(rnd()*14|0),y=3+(rnd()*14|0);if(Math.abs(x-10)+Math.abs(y-10)>5)this.prot[id(x,y)]=1;}
  for(let y=8;y<=12;y++)for(let x=8;x<=12;x++){this.wall[id(x,y)]=0;this.prot[id(x,y)]=0;}
  this.paint[id(10,10)]=1;
  let desired={
   Greebles:[[15,10],[15,4],[4,10],[4,15],[15,15],[10,10]],
   Pipework:[[16,10],[16,3],[3,3],[3,16],[16,16],[10,10]],
   Architecture:[[10,3],[3,3],[3,16],[16,16],[16,3],[10,10]]}[fam];
  let reach=[[10,10]],seen=new Set([id(10,10)]);
  for(let k=0;k<reach.length;k++){const [x,y]=reach[k];for(const [dx,dy] of DIR){let xx=x+dx,yy=y+dy,p=id(xx,yy);
   if(this.safe(xx,yy)&&!seen.has(p)){seen.add(p);reach.push([xx,yy]);}}}
  this.goals=desired.map(([gx,gy])=>reach.reduce((a,b)=>Math.abs(b[0]-gx)+Math.abs(b[1]-gy)<Math.abs(a[0]-gx)+Math.abs(a[1]-gy)?b:a));
  this.progress();
 }
 safe(x,y){return inB(x,y)&&!this.wall[id(x,y)]&&!this.prot[id(x,y)];}
 progress(){while(this.g<6&&this.x===this.goals[this.g][0]&&this.y===this.goals[this.g][1])this.g++;}
 target(){return this.goals[this.g]??[this.x,this.y];}
 legal(){let out=[];if(this.t===LIM)return out;for(let a=0;a<4;a++)if(this.safe(this.x+DIR[a][0],this.y+DIR[a][1]))out.push(a);out.push(4);return out;}
 sense(){let out=[];for(let plane of [this.wall,this.prot,this.paint])for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){let x=this.x+dx,y=this.y+dy;
   out.push(inB(x,y)?plane[id(x,y)]:+(plane===this.wall));}return out;}
 base(){const g=this.target(),leg=this.legal();return this.sense().concat(
   [0,1,2,3].map(a=>+(a===this.heading)),FAMS.map(f=>+(f===this.fam)),
   [(g[0]-this.x)/20,(g[1]-this.y)/20,this.g/6,this.t/LIM],
   [0,1,2,3,4].map(a=>+leg.includes(a)));}
 oracle(){if(this.g===6)return 4;let q=[[this.x,this.y,4]],seen=new Uint8Array(400);seen[id(this.x,this.y)]=1;
  const g=this.target();for(let i=0;i<q.length;i++){let [x,y,first]=q[i];if(x===g[0]&&y===g[1])return first;
   for(let a=0;a<4;a++){let xx=x+DIR[a][0],yy=y+DIR[a][1],j=id(xx,yy);if(this.safe(xx,yy)&&!seen[j]){seen[j]=1;q.push([xx,yy,i===0?a:first]);}}}
  return 4;
 }
 greedy(){if(this.g===6)return 4;const g=this.target(),opts=this.legal().filter(a=>a<4);
  opts.sort((a,b)=>{let v=c=>Math.abs(this.x+DIR[c][0]-g[0])+Math.abs(this.y+DIR[c][1]-g[1]);
    return v(a)-v(b)||+(a!==this.heading)-+(b!==this.heading)||a-b;});return opts[0]??4;}
 step(a){if(!this.legal().includes(a))throw Error("illegal applied action");if(a<4){this.x+=DIR[a][0];this.y+=DIR[a][1];
   this.heading=a;let j=id(this.x,this.y);if(this.paint[j])this.retraces++;this.paint[j]=1;this.path.push([this.x,this.y]);}
   this.t++;this.progress();
 }
 metrics(){let edges=new Map();for(let k=1;k<this.path.length;k++){let a=this.path[k-1].join(","),b=this.path[k].join(",");
   if(!edges.has(a))edges.set(a,new Set());if(!edges.has(b))edges.set(b,new Set());edges.get(a).add(b);edges.get(b).add(a);}
  return {cells:this.paint.reduce((a,b)=>a+b,0),waypoints:this.g,retraces:this.retraces,branches:[...edges.values()].filter(s=>s.size>=3).length,turns:this.t};}
 ascii(){let visited=new Set(this.path.map(p=>id(...p))),out=[];for(let y=2;y<18;y++){let s="";for(let x=2;x<18;x++){let i=id(x,y);
    s+=x===this.x&&y===this.y?"@":x===10&&y===10?"S":this.prot[i]?"!":this.wall[i]?"#":visited.has(i)?"*":".";}out.push(s);}return out.join("\n");}
}
class MapMemory{
 constructor(){this.map=new Int8Array(400).fill(-1);this.trail=new Uint8Array(400);}
 update(w){const obs=w.sense();for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){let x=w.x+dx,y=w.y+dy,i=(dy+2)*5+dx+2;
   if(inB(x,y))this.map[id(x,y)]=(obs[i]||obs[25+i])?1:0;}this.trail[id(w.x,w.y)]=1;}
 features(w){let v=w.base();for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++){let x=w.x+dx,y=w.y+dy;
    v.push(inB(x,y)?this.map[id(x,y)]:-1,inB(x,y)?this.trail[id(x,y)]:0);}return v;}
 planner(w){if(w.g===6)return 4;let dist=new Float64Array(400).fill(1e9),first=new Int8Array(400).fill(-1),done=new Uint8Array(400),start=id(w.x,w.y),dest=id(...w.target());
  dist[start]=0;first[start]=4;
  for(let k=0;k<400;k++){let p=-1,best=1e9;for(let j=0;j<400;j++)if(!done[j]&&dist[j]<best){p=j;best=dist[j];}
   if(p<0)break;if(p===dest)return w.legal().includes(first[p])?first[p]:w.greedy();
   done[p]=1;let x=p%20,y=p/20|0;for(let a=0;a<4;a++){let xx=x+DIR[a][0],yy=y+DIR[a][1];if(xx<2||xx>17||yy<2||yy>17)continue;let j=id(xx,yy);if(this.map[j]===1)continue;
    let v=best+(this.map[j]<0?1.5:1);if(v<dist[j]){dist[j]=v;first[j]=p===start?a:first[p];}}}
  return w.greedy();
 }
}
function features(w,m){m.update(w);let local=w.base(),memory=m.features(w);return {local,memory,legal:w.legal(),label:w.oracle()};}
function collect(split){let arr=[];for(let seed=SPLIT[split][0];seed<SPLIT[split][1];seed++)for(let f of FAMS)for(let e=0;e<2;e++){
  let w=new World(seed,f,split),m=new MapMemory(),r=random(seed*11+FAMS.indexOf(f)*17+e*101);
  for(let t=0;t<LIM;t++){let ex=features(w,m);if(!ex.legal.includes(ex.label))throw Error("oracle illegal");
   arr.push(ex);let a=ex.label;if(e===1&&r()<.3){let opts=ex.legal.filter(a=>a<4);a=opts[r()*opts.length|0]??4;}
   w.step(a);if(a===4)break;
  }
 }return arr;}
const train=collect("train"),dev=collect("dev"),dimLocal=train[0].local.length,dimMemory=train[0].memory.length;
function fit(name,seed,d,hidden,key){
  let r=random(seed),K=5,count=hidden?hidden*d+hidden+hidden*K+K:K*d+K;
  let w=new Float64Array(count),m=new Float64Array(count),v=new Float64Array(count);
  for(let j=0;j<count;j++)w[j]=(r()*2-1)*.03;
  function forward(x){let h=new Float64Array(hidden),z=new Float64Array(K);
    if(hidden){for(let a=0;a<hidden;a++){let s=w[d*hidden+a];for(let i=0;i<d;i++)s+=w[a*d+i]*x[i];h[a]=Math.tanh(s);}
      for(let a=0;a<K;a++){let s=w[d*hidden+hidden+K*hidden+a];for(let j=0;j<hidden;j++)s+=w[d*hidden+hidden+a*hidden+j]*h[j];z[a]=s;}}
    else for(let a=0;a<K;a++){let s=w[K*d+a];for(let i=0;i<d;i++)s+=w[a*d+i]*x[i];z[a]=s;}return {h,z};
  }
  const decide=(ex,ablate=false)=>{let x=ex[key];if(ablate){x=x.slice();x.fill(0,dimLocal);}
    let s=forward(x).z;for(let a=0;a<K;a++)if(!ex.legal.includes(a))s[a]=-1e8;return s.indexOf(Math.max(...s));};
  let loss=0;
  for(let step=1;step<=240;step++){
   let grad=new Float64Array(count),sum=0;
   for(let b=0;b<24;b++){
    let ex=train[r()*train.length|0],x=ex[key],f=forward(x),s=f.z.map((q,a)=>ex.legal.includes(a)?q:-1e8),mx=Math.max(...s),p=s.map(q=>Math.exp(q-mx)),normal=p.reduce((a,b)=>a+b,0);
    for(let a=0;a<K;a++)p[a]/=normal;
    sum-=Math.log(Math.max(1e-12,p[ex.label]));
    if(hidden){let dh=new Float64Array(hidden);for(let a=0;a<K;a++){let dl=(p[a]-+(a===ex.label))/24;grad[d*hidden+hidden+K*hidden+a]+=dl;
      for(let j=0;j<hidden;j++){grad[d*hidden+hidden+a*hidden+j]+=dl*f.h[j];dh[j]+=dl*w[d*hidden+hidden+a*hidden+j];}}
      for(let j=0;j<hidden;j++){let delta=dh[j]*(1-f.h[j]**2);grad[d*hidden+j]+=delta;for(let i=0;i<d;i++)grad[j*d+i]+=delta*x[i];}}
    else for(let a=0;a<K;a++){let dl=(p[a]-+(a===ex.label))/24;grad[K*d+a]+=dl;for(let i=0;i<d;i++)grad[a*d+i]+=dl*x[i];}
   }
   loss=sum/24;
   for(let i=0;i<count;i++){let g=Math.max(-1,Math.min(1,grad[i]));m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g;
    w[i]-=.006*(m[i]/(1-.9**step))/(Math.sqrt(v[i]/(1-.999**step))+1e-8)+.000006*w[i];}
  }
  let right=0,incorrectStop=0;for(const ex of dev){let a=decide(ex);right+=+(a===ex.label);incorrectStop+=+(a===4&&ex.label!==4);}
  return {name,parameters:count,steps:240,devActionAgreement:+(right/dev.length).toFixed(4),
    devIncorrectStop:incorrectStop,lastMiniBatchLoss:+loss.toFixed(4),weightsFNV32:hash(JSON.stringify([...w].map(x=>+x.toFixed(9)))),decide};
}
const heads={
 "local-mlp":fit("local-mlp",1801,dimLocal,16,"local"),
 "memory-linear":fit("memory-linear",1802,dimMemory,0,"memory"),
 "memory-mlp":fit("memory-mlp",1803,dimMemory,16,"memory")
};
const P=["oracle","map-planner","greedy","novelty","random-move","random-legal","local-mlp","memory-linear","memory-mlp","memory-mlp-ablated"];
function rollout(seed,f,pol,verify=false){
  let w=new World(seed,f,"eval"),m=new MapMemory(),r=random(seed*113+FAMS.indexOf(f)*19),actions=[],receipts=[];
  for(let t=0;t<LIM;t++){let ex=features(w,m),a,leg=ex.legal;
    if(pol==="oracle")a=ex.label;
    else if(pol==="map-planner")a=m.planner(w);
    else if(pol==="greedy")a=w.greedy();
    else if(pol==="novelty"){let opts=leg.filter(a=>a<4);opts.sort((a,b)=>w.paint[id(w.x+DIR[a][0],w.y+DIR[a][1])]-w.paint[id(w.x+DIR[b][0],w.y+DIR[b][1])]||a-b);a=opts[0]??4;}
    else if(pol==="random-move"){let opts=leg.filter(a=>a<4);a=opts[r()*opts.length|0]??4;}
    else if(pol==="random-legal")a=leg[r()*leg.length|0];
    else a=heads[pol==="memory-mlp-ablated"?"memory-mlp":pol].decide(ex,pol==="memory-mlp-ablated");
    if(!leg.includes(a))throw Error("illegal game action");receipts.push([w.x,w.y,w.g,leg.join(""),a].join(":"));actions.push(a);w.step(a);if(a===4)break;
  }
  let metrics=w.metrics();
  if(verify){let second=new World(seed,f,"eval");for(const a of actions){if(!second.legal().includes(a))throw Error("independent legal audit");second.step(a);}
    if(JSON.stringify(second.metrics())!==JSON.stringify(metrics)||second.ascii()!==w.ascii())throw Error("independent replay failed");}
  return {seed,f,policy:pol,...metrics,actionSequence:actions.join(""),receiptFNV32:hash(JSON.stringify(receipts)),art:verify?w.ascii():undefined,
    visited:[...w.paint].map((v,i)=>v?i:-1).filter(v=>v>=0)};
}
const rows=[],gallery=[];
for(let seed=SPLIT.eval[0];seed<SPLIT.eval[1];seed++)for(const f of FAMS)for(const policy of P){
  let sample=rollout(seed,f,policy,seed===SPLIT.eval[0]),again=rollout(seed,f,policy,false);
  if(sample.actionSequence!==again.actionSequence||sample.receiptFNV32!==again.receiptFNV32)throw Error("unstable deterministic run");
  if(sample.art)gallery.push({seed,f,policy,ascii:sample.art,actions:sample.actionSequence});delete sample.art;rows.push(sample);
}
const means={};
for(const pol of P){let s=rows.filter(r=>r.policy===pol),o={};
 for(const key of ["cells","waypoints","retraces","branches","turns"])o[key]=+(s.reduce((x,r)=>x+r[key],0)/s.length).toFixed(3);
 let ious=[];for(let seed=SPLIT.eval[0];seed<SPLIT.eval[1];seed++){
  let tri=s.filter(r=>r.seed===seed);for(let a=0;a<3;a++)for(let b=a+1;b<3;b++){
   let x=new Set(tri[a].visited),y=new Set(tri[b].visited),i=[...x].filter(t=>y.has(t)).length;ious.push(i/(x.size+y.size-i));}}
 o.familyFootprintIoU=+(ious.reduce((a,b)=>a+b,0)/ious.length).toFixed(3);means[pol]=o;
}
const compare={};
for(const [a,b] of [["memory-mlp","local-mlp"],["memory-mlp","map-planner"],["memory-mlp-ablated","memory-mlp"],["memory-linear","local-mlp"]]){
 const diffs=Array.from({length:9},(_,k)=>{let seed=SPLIT.eval[0]+k;const mean=(p)=>rows.filter(r=>r.seed===seed&&r.policy===p).reduce((s,r)=>s+r.waypoints,0)/3;
   return mean(a)-mean(b);}),r=random(18018),boots=[];
 for(let k=0;k<2000;k++){let s=0;for(let i=0;i<9;i++)s+=diffs[r()*9|0];boots.push(s/9);}
 boots.sort((x,y)=>x-y);compare[a+" minus "+b]={mean:+(diffs.reduce((s,x)=>s+x,0)/9).toFixed(3),seedCluster95:[+boots[49].toFixed(3),+boots[1949].toFixed(3)]};
}
let probe=new World(SPLIT.eval[0],"Greebles","eval"),mem=new MapMemory();mem.update(probe);
if(mem.map[id(2,2)]!==-1||dimMemory!==dimLocal+162)throw Error("causal memory feature leak");
return {
 protocol:"M18-CAUSAL-MAP-NO-PLANNER-HINT",date:"2026-11-16",syntheticOnly:true,
 caveat:"Exploratory game-like 20x20 synthetic worlds with projected reachable goals, NOT native CyberSand ABI or sealed evaluation; engineered sensor-map memory, not a learned world model.",
 splits:SPLIT,budget:{optimizerUpdates:720,trainExamples:train.length,devExamples:dev.length,evalTrajectories:rows.length,legalActions:rows.reduce((s,r)=>s+r.turns,0),evalActionCap:LIM},
 inputDimensions:{local:dimLocal,memory:dimMemory},
 heads:Object.fromEntries(Object.entries(heads).map(([k,v])=>[k,{parameters:v.parameters,steps:v.steps,devActionAgreement:v.devActionAgreement,devIncorrectStop:v.devIncorrectStop,lastMiniBatchLoss:v.lastMiniBatchLoss,noncryptographicWeightFNV32:v.weightsFNV32}])),
 means,pairedWaypointDiffs:compare,
 assertions:{noDirectPlannerHint:true,unseenCellsRemainUnknown:true,independentFirstSeedReplay:true,allActionsLegal:true,allTrajectoriesDeterministic:true,trainDevEvalSeedDisjoint:true,evalGeometriesNotInTrain:true},
 trajectories:rows.map(({visited,...x})=>x),asciiGallery:gallery,
 limitations:"No saved model checkpoints or immutable hosted artifact; no pretrained SmolLM2/OPT/Pythia tuning, LoRA, physical game test, generative aesthetics validation or native integration. Family effects conflate host-specified waypoints. Maps are deterministic causal memory, not a learned encoder. No model promoted."
};
};
if(typeof module!=='undefined'&&require.main===module)process.stdout.write(JSON.stringify(runM18(),null,2)+'\n');
module.exports=runM18;
