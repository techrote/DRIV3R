/* M17 bounded synthetic causal memory action-head experiment.
 * JavaScript/Node 20+, no dependencies; three 240-step CPU AdamW heads.
 * source is the function executed for this report, with no game-side writes.
 * Usage: node m17_causal_memory.js > m17_result.json
 */
const m17Experiment = function m17Experiment() {
  "use strict";
  const SIZE=20, BUDGET=80;
  const FAMS=["Greebles","Pipework","Architecture"];
  const MOVE=[[0,-1],[0,1],[-1,0],[1,0]];
  const SPLITS={train:[201000,201016],dev:[202000,202004],eval:[203000,203009]};
  function rand(seed){let s=seed>>>0;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};}
  function fnv(s){let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return (h>>>0).toString(16).padStart(8,"0");}
  function idx(x,y){return y*SIZE+x;}
  function valid(x,y){return x>=0&&y>=0&&x<SIZE&&y<SIZE;}
  class World {
    constructor(seed,fam,split) {
      if(!(seed>=SPLITS[split][0]&&seed<SPLITS[split][1]&&FAMS.includes(fam)))throw Error("split/family");
      this.seed=seed;this.fam=fam;this.split=split;
      this.block=new Uint8Array(SIZE*SIZE);this.protect=new Uint8Array(SIZE*SIZE);
      this.visited=new Uint8Array(SIZE*SIZE);this.x=this.y=10;this.heading=3;this.t=0;this.g=0;this.path=[[10,10]];this.retraces=0;
      const rng=rand((seed*37+7)>>>0);
      for(let y=0;y<SIZE;y++)for(let x=0;x<SIZE;x++)if(x<2||x>SIZE-3||y<2||y>SIZE-3)this.block[idx(x,y)]=1;
      if(split!=="eval") {
        for(const x of [5,9,14])for(let y=3;y<17;y++)this.block[idx(x,y)]=1;
        for(const x of [5,9,14]){const gap=4+(seed+x)%9;for(let y=gap;y<gap+3;y++)this.block[idx(x,y)]=0;}
        for(let k=0;k<5;k++){const x=3+Math.floor(rng()*14),y=3+Math.floor(rng()*14);if(Math.abs(x-10)+Math.abs(y-10)>5)this.block[idx(x,y)]=1;}
      } else {
        const variant=(seed-SPLITS.eval[0])%3;
        if(variant===0){
          for(const [lo,hi] of [[3,16],[6,13]]) {
            for(let i=lo;i<=hi;i++){this.block[idx(i,lo)]=1;this.block[idx(i,hi)]=1;this.block[idx(lo,i)]=1;this.block[idx(hi,i)]=1;}
            for(let i=hi-4;i<=hi-2;i++)this.block[idx(i,lo)]=0;
            for(let i=lo+2;i<=lo+4;i++)this.block[idx(hi,i)]=0;
          }
        } else if(variant===1) {
          for(let k=0;k<4;k++){const y=[4,7,13,16][k];for(let x=3;x<17;x++)this.block[idx(x,y)]=1;const hole=3+(seed+k*5)%11;for(let x=hole;x<hole+3;x++)this.block[idx(x,y)]=0;}
        } else {
          for(const [x,hole] of [[5,5],[8,13],[12,5],[15,13]]){for(let y=3;y<17;y++)this.block[idx(x,y)]=1;for(let y=hole;y<hole+2;y++)this.block[idx(x,y)]=0;}
        }
      }
      for(let k=0;k<5;k++){const x=3+Math.floor(rng()*14),y=3+Math.floor(rng()*14);if(Math.abs(x-10)+Math.abs(y-10)>5)this.protect[idx(x,y)]=1;}
      for(let y=8;y<=12;y++)for(let x=8;x<=12;x++){this.block[idx(x,y)]=0;this.protect[idx(x,y)]=0;}
      this.visited[idx(10,10)]=1;
      const points={
        Greebles:[[15,10],[15,4],[4,10],[4,15],[15,15],[10,10]],
        Pipework:[[16,10],[16,3],[3,3],[3,16],[16,16],[10,10]],
        Architecture:[[10,3],[3,3],[3,16],[16,16],[16,3],[10,10]]
      }[fam];
      const reach=new Map([["10,10",0]]),q=[[10,10]];
      for(let k=0;k<q.length;k++){let [x,y]=q[k];for(let [dx,dy] of MOVE){const a=x+dx,b=y+dy,key=a+","+b;if(this.safe(a,b)&&!reach.has(key)){reach.set(key,reach.get(x+","+y)+1);q.push([a,b]);}}}
      this.goals=points.map(([gx,gy])=>q.slice().sort((a,b)=>(Math.abs(a[0]-gx)+Math.abs(a[1]-gy))-(Math.abs(b[0]-gx)+Math.abs(b[1]-gy))||reach.get(a.join(","))-reach.get(b.join(",")))[0]);
      this.progress();
    }
    safe(x,y){return valid(x,y)&&!this.block[idx(x,y)]&&!this.protect[idx(x,y)];}
    progress(){while(this.g<this.goals.length&&this.x===this.goals[this.g][0]&&this.y===this.goals[this.g][1])this.g++;}
    target(){return this.goals[this.g]??[this.x,this.y];}
    legal(){if(this.t>=BUDGET)return [];return [0,1,2,3].filter(i=>this.safe(this.x+MOVE[i][0],this.y+MOVE[i][1])).concat([4]);}
    sense(){
      const a=[];
      for(const property of ["block","protect","visited"])
        for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
          const x=this.x+dx,y=this.y+dy;
          a.push(valid(x,y)?this[property][idx(x,y)]:(property==="block"?1:0));
        }
      return a;
    }
    features(){
      const g=this.target(),leg=this.legal();
      return this.sense().concat([0,1,2,3].map(a=>+(a===this.heading)),
        FAMS.map(f=>+(f===this.fam)),[(g[0]-this.x)/SIZE,(g[1]-this.y)/SIZE,this.t/BUDGET,this.g/6],
        [0,1,2,3,4].map(a=>+leg.includes(a)));
    }
    step(a){
      if(!Number.isInteger(a)||!this.legal().includes(a))throw Error("illegal");
      if(a!==4){this.x+=MOVE[a][0];this.y+=MOVE[a][1];this.heading=a;let p=idx(this.x,this.y);if(this.visited[p])this.retraces++;this.visited[p]=1;this.path.push([this.x,this.y]);}
      this.t++;this.progress();
    }
    oracle(){
      if(this.g===6)return 4;
      let target=this.target(),q=[[this.x,this.y]],seen=new Set([this.x+","+this.y]),first=[4];
      for(let k=0;k<q.length;k++){
        let [x,y]=q[k];if(x===target[0]&&y===target[1])return first[k];
        for(let a=0;a<4;a++){let nx=x+MOVE[a][0],ny=y+MOVE[a][1],key=nx+","+ny;
          if(this.safe(nx,ny)&&!seen.has(key)){seen.add(key);q.push([nx,ny]);first.push(k===0?a:first[k]);}}
      }return 4;
    }
    greedy(){
      if(this.g===6)return 4;const g=this.target(),leg=this.legal().filter(x=>x<4);
      return leg.sort((a,b)=>{function val(t){return Math.abs(g[0]-this.x-MOVE[t][0])+Math.abs(g[1]-this.y-MOVE[t][1]);}return val.call(this,a)-val.call(this,b)||+(a!==this.heading)-+(b!==this.heading)||a-b;})[0]??4;
    }
    state(){return fnv(JSON.stringify([this.x,this.y,this.g,this.t,this.heading,this.path,this.visited.toString()]));}
    metrics(){
      const edges=new Map();for(let k=1;k<this.path.length;k++){const a=this.path[k-1].join(","),b=this.path[k].join(",");if(!edges.has(a))edges.set(a,new Set());if(!edges.has(b))edges.set(b,new Set());edges.get(a).add(b);edges.get(b).add(a);}
      return {cells:this.visited.reduce((a,b)=>a+b,0),waypoints:this.g,retraces:this.retraces,branches:[...edges.values()].filter(s=>s.size>=3).length};
    }
  }
  class MapMemory{
    constructor(){this.grid=new Int8Array(SIZE*SIZE).fill(-1);}
    update(w){
      const p=w.sense();let n=0;
      for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++,n++){
        let x=w.x+dx,y=w.y+dy;if(valid(x,y))this.grid[idx(x,y)]=(p[n]||p[25+n])?1:0;
      }
    }
    proposal(w){
      this.update(w);if(w.g===6)return 4;
      const dist=new Float64Array(SIZE*SIZE).fill(1e15),first=new Int8Array(SIZE*SIZE).fill(-1);
      const start=idx(w.x,w.y),target=idx(...w.target());dist[start]=0;first[start]=4;
      const used=new Uint8Array(SIZE*SIZE);
      for(let turn=0;turn<SIZE*SIZE;turn++){
        let p=-1,best=1e15;for(let j=0;j<dist.length;j++)if(!used[j]&&dist[j]<best){p=j;best=dist[j];}
        if(p<0)break;if(p===target)return first[p];used[p]=1;
        let x=p%SIZE,y=Math.floor(p/SIZE);
        for(let a=0;a<4;a++){
          let xx=x+MOVE[a][0],yy=y+MOVE[a][1];if(xx<1||xx>SIZE-2||yy<1||yy>SIZE-2)continue;
          let j=idx(xx,yy);if(this.grid[j]===1)continue;
          let d=best+(this.grid[j]<0?1.35:1.0);
          if(d+1e-9<dist[j]){dist[j]=d;first[j]=p===start?a:first[p];}
        }
      }
      return w.greedy();
    }
    hint(w){const a=this.proposal(w);return [0,1,2,3,4].map(i=>+(i===a)).concat([this.grid.reduce((n,t)=>n+(t<0),0)/(SIZE*SIZE)]);}
  }
  const seeds=(split)=>Array.from({length:SPLITS[split][1]-SPLITS[split][0]},(_,i)=>SPLITS[split][0]+i);
  const examples=(split)=>{
    let out=[];
    for(const seed of seeds(split))for(const fam of FAMS)for(let ep=0;ep<2;ep++){
      let w=new World(seed,fam,split),map=new MapMemory(),rng=rand((seed*29+FAMS.indexOf(fam)*7+ep)>>>0);
      for(let t=0;t<BUDGET;t++){
        let hint=map.hint(w),teacher=hint.indexOf(1),legal=w.legal();
        if(!legal.includes(teacher))throw Error("teacher illegal");
        out.push({x:w.features(),z:hint,y:teacher,legal});
        let a=teacher;
        if(ep===1&&rng()<.2){const opts=legal.filter(a=>a<4);a=opts[Math.floor(rng()*opts.length)]??4;}
        w.step(a);if(a===4)break;
      }
    }
    return out;
  };
  const train=examples("train"),dev=examples("dev");
  function transform(ex,kind,rff){
    if(kind==="graph-linear")return ex.x.concat(ex.z,1);
    if(kind==="local-rff"){let x=ex.x,v=[];for(const row of rff){let sum=0;for(let i=0;i<x.length;i++)sum+=row[i]*x[i];v.push(Math.tanh(sum));}return v.concat(1);}
    return ex.x.concat(1);
  }
  function fit(kind,seed,steps=240){
    const rng=rand(seed),rff=[];
    if(kind==="local-rff")for(let j=0;j<40;j++)rff.push(Array.from({length:train[0].x.length},()=>.25*(rng()*2-1)));
    const dim=transform(train[0],kind,rff).length;
    const weights=Array.from({length:5},()=>Array.from({length:dim},()=>.025*(rng()*2-1)));
    const mom=weights.map(row=>row.map(()=>0)),vel=weights.map(row=>row.map(()=>0));
    function scores(ex){let x=transform(ex,kind,rff);return weights.map(row=>row.reduce((v,w,i)=>v+w*x[i],0));}
    let latest=0;
    for(let step=1;step<=steps;step++){
      let grad=weights.map(row=>row.map(()=>0)),loss=0;
      for(let k=0;k<24;k++){
        let ex=train[Math.floor(rng()*train.length)],x=transform(ex,kind,rff),s=scores(ex).map((v,i)=>ex.legal.includes(i)?v:-1e6);
        const max=Math.max(...s),prob=s.map(v=>Math.exp(v-max)),z=prob.reduce((a,b)=>a+b,0);
        for(let a=0;a<5;a++)prob[a]/=z;
        loss-=Math.log(Math.max(1e-12,prob[ex.y]));
        for(let a=0;a<5;a++)for(let i=0;i<dim;i++)grad[a][i]+=(prob[a]-+(a===ex.y))*x[i]/24;
      }
      latest=loss/24;
      for(let a=0;a<5;a++)for(let i=0;i<dim;i++){
        let g=Math.max(-1,Math.min(1,grad[a][i])),m=mom[a][i]=.9*mom[a][i]+.1*g,v=vel[a][i]=.999*vel[a][i]+.001*g*g;
        weights[a][i]-=.012*(m/(1-.9**step))/(Math.sqrt(v/(1-.999**step))+1e-8)+.00001*weights[a][i];
      }
    }
    let right=0;
    for(const ex of dev){let s=scores(ex).map((x,i)=>ex.legal.includes(i)?x:-1e8);let a=s.indexOf(Math.max(...s));right+=+(a===ex.y);}
    return {kind,weights,rff,scorer:scores,steps,trainExamples:train.length,devAccuracy:+(right/dev.length).toFixed(4),lastMiniBatchLoss:+latest.toFixed(4),parameters:5*dim};
  }
  const heads={
    "local-linear":fit("local-linear",1701),
    "local-rff":fit("local-rff",1702),
    "graph-linear":fit("graph-linear",1703)
  };
  const policies=["oracle","greedy","novelty","causal-map","local-linear","local-rff","graph-linear","graph-linear-ablated"];
  function roll(seed,fam,pol){
    let w=new World(seed,fam,"eval"),map=new MapMemory(),events=[];
    for(let t=0;t<BUDGET;t++){
      const legal=w.legal(),before=w.state();
      let a;
      if(pol==="oracle")a=w.oracle();
      else if(pol==="greedy")a=w.greedy();
      else if(pol==="novelty"){const opts=legal.filter(i=>i<4);opts.sort((a,b)=>w.visited[idx(w.x+MOVE[a][0],w.y+MOVE[a][1])]-w.visited[idx(w.x+MOVE[b][0],w.y+MOVE[b][1])]||a-b);a=opts[0]??4;}
      else if(pol==="causal-map")a=map.proposal(w);
      else {
        let ex={x:w.features(),z:pol.startsWith("graph")?map.hint(w):[],legal};
        if(pol==="graph-linear-ablated")ex.z=new Array(6).fill(0);
        let head=heads[pol==="graph-linear-ablated"?"graph-linear":pol];
        let s=head.scorer(ex).map((n,i)=>legal.includes(i)?n:-1e8);
        a=s.indexOf(Math.max(...s));
      }
      if(!legal.includes(a))throw Error("host legal violation");
      w.step(a);events.push({b:before,l:legal.join(""),a,z:w.state()});
      if(a===4)break;
    }
    return {seed,fam,policy:pol,metrics:w.metrics(),events};
  }
  let rows=[],sample=[];
  for(let seed of seeds("eval"))for(let fam of FAMS)for(let policy of policies){
    const r=roll(seed,fam,policy);
    const again=roll(seed,fam,policy);
    if(JSON.stringify(r)!==JSON.stringify(again))throw Error("nondeterministic replay");
    rows.push({seed,fam,policy,...r.metrics,turns:r.events.length,receiptFNV32:fnv(JSON.stringify(r.events))});
    if(seed===seeds("eval")[0])sample.push({fam,policy,actions:r.events.map(e=>e.a).join(""),metrics:r.metrics});
  }
  const summary={};
  for(const p of policies){let matches=rows.filter(r=>r.policy===p);summary[p]={};
    for(const key of ["cells","waypoints","retraces","branches","turns"])summary[p][key]=+(matches.reduce((a,b)=>a+b[key],0)/matches.length).toFixed(3);
  }
  const diffs={};
  for(const [a,b] of [["causal-map","greedy"],["graph-linear","local-linear"],["graph-linear-ablated","graph-linear"],["graph-linear","causal-map"]]){
    const vals=seeds("eval").map(seed=>{
      let ma=rows.filter(r=>r.seed===seed&&r.policy===a).reduce((s,r)=>s+r.waypoints,0)/3;
      let mb=rows.filter(r=>r.seed===seed&&r.policy===b).reduce((s,r)=>s+r.waypoints,0)/3;
      return ma-mb;
    });
    const rng=rand(1717),boot=[];
    for(let k=0;k<2000;k++){let mean=0;for(let i=0;i<vals.length;i++)mean+=vals[Math.floor(rng()*vals.length)];boot.push(mean/vals.length);}
    boot.sort((a,b)=>a-b);
    diffs[a+" vs "+b]={delta:+(vals.reduce((x,y)=>x+y,0)/vals.length).toFixed(3),cluster95:[+boot[49].toFixed(3),+boot[1949].toFixed(3)]};
  }
  const weightsDigest={};
  for(const [name,h] of Object.entries(heads))weightsDigest[name]=fnv(JSON.stringify(h.weights));
  return {
    protocol:"M17-JS-CAUSAL-MAP-2026-11-16",syntheticOnly:true,
    caveat:"Exploratory 20x20 toy world, two-dimensional navigation and host-projected reachable targets; NOT CyberSand ABI, pretrained LLM adaptation, native gameplay or sealed FINAL",
    headDetails:Object.fromEntries(Object.entries(heads).map(([name,h])=>[name,{adamwSteps:h.steps,trainExamples:h.trainExamples,devAccuracy:h.devAccuracy,loss:h.lastMiniBatchLoss,parameters:h.parameters,noncryptographicWeightFingerprint:weightsDigest[name]}])),
    splits:SPLITS,trainSamples:train.length,devSamples:dev.length,evaluatedTrajectories:rows.length,
    legalActionsReplayed:rows.reduce((s,r)=>s+r.turns,0),summary,pairedWaypointDiffs:diffs,
    rows,exampleActionSequences:sample,
    scientificBoundary:"Direct action hint is generated by persistent causal graph; graph-linear is a student of a strong deterministic planner, not independent planning. No checkpoints committed."
  };
};
if (typeof module !== "undefined" && require.main === module) {
  process.stdout.write(JSON.stringify(m17Experiment(), null, 2) + "\n");
}
