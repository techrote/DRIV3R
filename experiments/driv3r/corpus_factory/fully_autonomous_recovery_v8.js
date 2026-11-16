/* DRIV3R Corpus Factory v8: autonomous, synthetic, host-legal recovery.
Research fixture only: no native ABI equivalence, no game-side changes.
Run: node fully_autonomous_recovery_v8.js /path/to/NEW/output/directory
*/
"use strict";
const V="driv3r/synthetic-autonomous-recovery-v8",BUDGET=176,EPOCHS=24;
const DIR=[["N",0,-1],["E",1,0],["S",0,1],["W",-1,0]],ORDER={N:4,E:3,S:2,W:1};
const K=(x,y)=>x+","+y;
function rand(seed){let s=(seed^0x9e3779b9)>>>0;return()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296);}
function sha256(str){
 const b=[];for(let i=0;i<str.length;i++){let c=str.charCodeAt(i);if(c<128)b.push(c);else if(c<2048)b.push(192|(c>>6),128|(c&63));else b.push(224|(c>>12),128|((c>>6)&63),128|(c&63));}
 const n=b.length;b.push(128);while(b.length%64!==56)b.push(0);for(let i=7;i>=0;i--)b.push(Math.floor(n*8/256**i)&255);
 const p=[],ks=[],H=[];for(let q=2;p.length<64;q++)if(p.every(x=>q%x)){p.push(q);let a=Math.sqrt(q),z=Math.cbrt(q);if(H.length<8)H.push(((a-Math.floor(a))*2**32)>>>0);ks.push(((z-Math.floor(z))*2**32)>>>0);}
 const rot=(x,k)=>(x>>>k)|(x<<(32-k));
 for(let off=0;off<b.length;off+=64){
   const w=[];for(let t=0;t<16;t++){let i=off+4*t;w[t]=((b[i]<<24)|(b[i+1]<<16)|(b[i+2]<<8)|b[i+3])>>>0;}
   for(let t=16;t<64;t++){let x=w[t-15],y=w[t-2];w[t]=(w[t-16]+(rot(x,7)^rot(x,18)^(x>>>3))+w[t-7]+(rot(y,17)^rot(y,19)^(y>>>10)))>>>0;}
   let [a,c,d,e,f,g,h,j]=H;for(let t=0;t<64;t++){let s=rot(f,6)^rot(f,11)^rot(f,25),ch=(f&g)^(~f&h),t1=(j+s+ch+ks[t]+w[t])>>>0,t2=((rot(a,2)^rot(a,13)^rot(a,22))+((a&c)^(a&d)^(c&d)))>>>0;
     j=h;h=g;g=f;f=(e+t1)>>>0;e=d;d=c;c=a;a=(t1+t2)>>>0;}
   const z=[a,c,d,e,f,g,h,j];for(let t=0;t<8;t++)H[t]=(H[t]+z[t])>>>0;
 }
 return H.map(x=>x.toString(16).padStart(8,"0")).join("");
}
function makeWorld(seed,good,grammar){
 if(!["N","S"].includes(good)||!["straight","switchback"].includes(grammar))throw Error("invalid scenario");
 const cells=new Set(),add=(x,y)=>cells.add(K(x,y)),Y={N:4,S:22};
 for(let x=1;x<=4;x++)add(x,13);
 for(const arm of ["N","S"]){
   const y=Y[arm],step=arm==="N"?-1:1;
   for(let z=13;z!==y+step;z+=step)add(4,z);
   if(arm!==good){for(let x=4;x<=14+seed%8;x++)add(x,y);continue;}
   if(grammar==="straight"){for(let x=4;x<=30;x++)add(x,y);}
   else{
     const yy=y+step*2;
     for(let x=4;x<=12;x++)add(x,y);
     for(let v=y;v!==yy+step;v+=step)add(12,v);
     for(let x=12;x<=23;x++)add(x,yy);
     for(let v=yy;v!==y-step;v-=step)add(23,v);
     for(let x=23;x<=30;x++)add(x,y);
   }
   for(let v=y;v!==13-step;v-=step)add(30,v);add(31,13);
 }
 const r=rand(seed*17+5),protectedCells=new Set();
 while(protectedCells.size<24){const x=8+Math.floor(r()*26),y=1+Math.floor(r()*25),p=K(x,y);
   if(!cells.has(p)&&!(x>=2&&x<=6&&y>=11&&y<=15))protectedCells.add(p);}
 const fingerprint=sha256(JSON.stringify({seed,grammar,cells:[...cells].sort(),protected:[...protectedCells].sort()}));
 return {seed,good,grammar,cells,protectedCells,fingerprint,start:[1,13],goal:[31,13]};
}
function legal(w,p){const a=[];for(const [d,dx,dy] of DIR){let x=p[0]+dx,y=p[1]+dy,k=K(x,y);
 if(w.cells.has(k)&&!w.protectedCells.has(k))a.push({d,p:[x,y],k});}return a;}
function shortest(w){let q=[[...w.start,0]],seen=new Set([K(...w.start)]);
 for(let i=0;i<q.length;i++){const [x,y,n]=q[i];if(x===31&&y===13)return n;
  for(const c of legal(w,[x,y]))if(!seen.has(c.k)){seen.add(c.k);q.push([...c.p,n+1]);}}
 return null;}
function motif(seed,x,y){let n=(Math.imul(seed+1,123457)+Math.imul(x+1,8167)+Math.imul(y+1,10007))>>>0;return ["#","s","m"][n%3];}
function local(w,p,paint){
 const grid=[];for(let dy=-2;dy<=2;dy++){let r="";for(let dx=-2;dx<=2;dx++){const x=p[0]+dx,y=p[1]+dy,k=K(x,y);
 r+=w.protectedCells.has(k)?"P":paint.has(k)?paint.get(k)[0].toLowerCase():w.cells.has(k)?".":motif(w.seed,x,y);}grid.push(r);}return grid;}
const ND=14;
function feature(w,st,c,mask){
 const dist=p=>Math.abs(31-p[0])+Math.abs(13-p[1]),vis=st.visits.get(c.k)||0;
 const prev=st.trace.length>1?K(...st.trace[st.trace.length-2]):"",age=st.last.has(c.k)?st.t-st.last.get(c.k):9999;
 const a=[...DIR.map(x=>+(x[0]===c.d)),(dist(st.p)-dist(c.p))/8,(c.p[0]-st.p[0])/8,
   (c.p[1]-st.p[1])/8,legal(w,c.p).length/4,+(!vis),Math.min(vis,6)/6,
   +(prev===c.k),+(age<=6),Math.min(age,30)/30,+(st.p[0]===4&&st.p[1]===13)];
 if(mask)for(let j=8;j<=12;j++)a[j]=0;
 if(a.length!==ND)throw Error("feature dimension mismatch");return a;
}
function teacher(w,st,choices){
 const fresh=choices.filter(c=>!(st.visits.get(c.k)||0));
 const d=c=>Math.abs(31-c.p[0])+Math.abs(13-c.p[1]);
 if(fresh.length)return fresh.sort((a,b)=>d(a)-d(b)||ORDER[b.d]-ORDER[a.d])[0];
 const prev=st.trace.length>1?K(...st.trace[st.trace.length-2]):"";
 return choices.slice().sort((a,b)=>+(a.k===prev)-+(b.k===prev)||
 (st.visits.get(a.k)||0)-(st.visits.get(b.k)||0)||ORDER[b.d]-ORDER[a.d])[0];
}
function choose(w,st,choices,weights,masked){
 let best=choices[0],v=-Infinity;
 for(const c of choices){const x=feature(w,st,c,masked);let s=0;for(let i=0;i<ND;i++)s+=weights[i]*x[i];
 if(s>v+1e-12){best=c;v=s;}}return best;
}
function rollout(w,kind,weights,mask,exportRows=false){
 const start=K(...w.start),st={p:[...w.start],t:0,visits:new Map([[start,1]]),last:new Map([[start,0]]),trace:[[...w.start]],paint:new Map()};
 const rows=[];let returns=0,branches=0,retraces=0,painted=0;
 while(st.t<BUDGET&&K(...st.p)!==K(...w.goal)){
   const cs=legal(w,st.p);if(!cs.length)break;
   const c=kind==="teacher"?teacher(w,st,cs):choose(w,st,cs,weights,mask);
   if(!cs.some(v=>v.k===c.k)||w.protectedCells.has(c.k))throw Error("illegal host action");
   if(exportRows)rows.push({tick:st.t,pos:[...st.p],local_5x5:local(w,st.p,st.paint),
     goal_delta:[31-st.p[0],13-st.p[1]],legal_actions:cs.map(v=>v.d),
     candidate_features:cs.map(v=>feature(w,st,v,false)),host_visits:cs.map(v=>st.visits.get(v.k)||0),
     teacher_action:c.d,hard_negative_actions:cs.filter(v=>v.d!==c.d).map(v=>v.d)});
   if(st.trace.length>=2&&c.k===K(...st.trace[st.trace.length-2]))returns++;
   if(st.p[0]===4&&st.p[1]===13)branches++;
   if(st.visits.has(c.k))retraces++;
   st.p=[...c.p];st.trace.push([...c.p]);st.t++;
   st.visits.set(c.k,(st.visits.get(c.k)||0)+1);st.last.set(c.k,st.t);
   if(!st.paint.has(c.k)){st.paint.set(c.k,["Sand","Stone","Metal"][(w.seed+c.p[0]*7+c.p[1]*11)%3]);painted++;}
 }
 return {solved:K(...st.p)===K(...w.goal),steps:st.t,unique:st.visits.size,
   two_step_returns:returns,branches,retraces,painted,trace:st.trace,rows};
}
function specs(lo,n,grammar){const a=[];for(let s=lo;s<lo+n;s++)for(const good of ["N","S"])a.push({seed:s,good,grammar});return a;}
function trainCorpus(spec){
 return spec.map(q=>{const w=makeWorld(q.seed,q.good,q.grammar),e=rollout(w,"teacher",null,false,true);
   if(!e.solved)throw Error("teacher failed "+q.seed+":"+q.good);
   return {split:"TRAIN",scenario_id:q.grammar+":"+q.seed+":"+q.good,paired_group_id:q.grammar+":"+q.seed,
    grammar:q.grammar,world_sha256:w.fingerprint,source:"independent-synthetic-v8",native_receipt:false,
    difficulty:"B2",host_applied_steps:e.steps,decisions:e.rows};});
}
function fit(rows,mask){
 const comparisons=[];
 for(const r of rows)for(const z of r.decisions){const idx=z.legal_actions.indexOf(z.teacher_action);
   for(let i=0;i<z.legal_actions.length;i++)if(i!==idx){const a=z.candidate_features[idx].slice(),b=z.candidate_features[i].slice();
     if(mask)for(let j=8;j<=12;j++){a[j]=0;b[j]=0;}
     comparisons.push(a.map((v,j)=>v-b[j]));}}
 const w=Array(ND).fill(0);
 for(let epoch=0;epoch<EPOCHS;epoch++){
   const r=rand(epoch+123456),ix=comparisons.map((_,i)=>i);
   for(let i=ix.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[ix[i],ix[j]]=[ix[j],ix[i]];}
   const eta=.095/Math.sqrt(1+epoch);
   for(const i of ix){const x=comparisons[i];let z=0;for(let j=0;j<ND;j++)z+=w[j]*x[j];
     const g=1/(1+Math.exp(Math.max(-50,Math.min(50,z))));
     for(let j=0;j<ND;j++)w[j]+=eta*(g*x[j]-.00005*w[j]);
   }
 }
 return {weights:w,comparisons:comparisons.length,sha256:sha256(JSON.stringify(w))};
}
function score(spec,models){
 const result={};
 for(const [name,m] of Object.entries(models)){
   const z={episodes:0,paired_groups:0,solved:0,both_solved_pairs:0,mean_unique:0,
     mean_two_step_returns:0,mean_retraces:0,mean_painted:0,mean_steps:0,illegal_moves:0,protected_entries:0};
   for(let i=0;i<spec.length;i+=2){let solved=0;
     for(const q of spec.slice(i,i+2)){const w=makeWorld(q.seed,q.good,q.grammar),e=rollout(w,"model",m.weights,m.mask);
       z.episodes++;z.solved+=+e.solved;solved+=+e.solved;z.mean_unique+=e.unique;
       z.mean_two_step_returns+=e.two_step_returns;z.mean_retraces+=e.retraces;z.mean_painted+=e.painted;z.mean_steps+=e.steps;}
     z.paired_groups++;z.both_solved_pairs+=+(solved===2);
   }
   for(const k of ["mean_unique","mean_two_step_returns","mean_retraces","mean_painted","mean_steps"])z[k]=Math.round(z[k]/z.episodes*1000)/1000;
   result[name]=z;
 }
 return result;
}
function gallery(spec,models){
 let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="900"><rect width="100%" height="100%" fill="#f9f9f9"/><style>text{font:13px monospace}</style><text x="20" y="25">DRIV3R v8 synthetic model-applied trajectories; red=protected, green=goal</text>',i=0;
 for(const q of spec.slice(0,4))for(const name of ["memory","sham"]){
   const w=makeWorld(q.seed,q.good,q.grammar),e=rollout(w,"model",models[name].weights,models[name].mask);
   const x=15+(i%2)*535,y=48+Math.floor(i/2)*210,z=5.4;i++;
   svg+='<rect x="'+x+'" y="'+(y-19)+'" width="525" height="195" fill="white" stroke="#999"/>';
   svg+='<text x="'+x+'" y="'+(y-3)+'">'+name+' '+q.seed+' '+q.good+' '+(e.solved?"SOLVED":"FAIL")+' steps='+e.steps+'</text>';
   for(const k of w.cells){const p=k.split(",").map(Number);svg+='<rect x="'+(x+p[0]*z)+'" y="'+(y+p[1]*z)+'" width="'+z+'" height="'+z+'" fill="#ddd"/>';}
   for(const k of w.protectedCells){const p=k.split(",").map(Number);svg+='<rect x="'+(x+p[0]*z)+'" y="'+(y+p[1]*z)+'" width="'+z+'" height="'+z+'" fill="#a33"/>';}
   svg+='<polyline fill="none" stroke="#1679c4" stroke-width="1.7" points="'+e.trace.map(p=>(x+(p[0]+.5)*z)+","+(y+(p[1]+.5)*z)).join(" ")+'"/>';
   svg+='<circle cx="'+(x+31.5*z)+'" cy="'+(y+13.5*z)+'" r="4" fill="#00833e"/>';
 }
 return svg+'</svg>';
}
function main(){
 if(sha256("abc")!=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")throw Error("SHA256 selftest");
 const sets={TRAIN:specs(10000,48,"straight"),DEV:specs(20000,12,"straight"),
   HELDOUT_same:specs(30000,32,"straight"),HELDOUT_unseen:specs(40000,32,"switchback")};
 const fingerprints=new Set(),seedSplit=new Map();
 for(const [split,scenarios] of Object.entries(sets)){
   for(let i=0;i<scenarios.length;i+=2){
     const [a,b]=scenarios.slice(i,i+2);if(a.seed!==b.seed||a.good===b.good)throw Error("pair mismatch");
     if(seedSplit.has(a.seed)&&seedSplit.get(a.seed)!==split)throw Error("split leakage");seedSplit.set(a.seed,split);
     const wa=makeWorld(a.seed,a.good,a.grammar),wb=makeWorld(b.seed,b.good,b.grammar);
     if(local(wa,[4,13],new Map()).join()!==local(wb,[4,13],new Map()).join())throw Error("fork observation leaks");
     if(legal(wa,[4,13]).map(c=>c.d).join()!==legal(wb,[4,13]).map(c=>c.d).join())throw Error("fork menu leaks");
     for(const w of [wa,wb]){
       if(fingerprints.has(w.fingerprint))throw Error("geometry duplicates across splits");
       fingerprints.add(w.fingerprint);const n=shortest(w);if(n===null||n>BUDGET)throw Error("unreachable task");
       if(w.protectedCells.size!==24||[...w.cells].some(c=>w.protectedCells.has(c)))throw Error("protected overlap");
     }
   }
 }
 const corpus=trainCorpus(sets.TRAIN);
 const trainJSONL=corpus.map(x=>JSON.stringify(x)).join("\n")+"\n";
 const actual=fit(corpus,false),sham=fit(corpus,true);
 const models={memory:{weights:actual.weights,mask:false},sham:{weights:sham.weights,mask:true},
   memory_masked:{weights:actual.weights,mask:true}};
 const scores={DEV:score(sets.DEV,models),HELDOUT_same:score(sets.HELDOUT_same,models),
   HELDOUT_unseen:score(sets.HELDOUT_unseen,models)};
 for(const group of Object.values(scores))for(const result of Object.values(group)){
   if(result.episodes!==result.paired_groups*2||result.illegal_moves||result.protected_entries)throw Error("unsafe scoring");
 }
 if(corpus.some(x=>x.split!=="TRAIN")||!trainJSONL.includes('"hard_negative_actions"'))throw Error("dataset leakage");
 const report={schema:V,execution:"synthetic-host, model-applied every action",source:"original external synthetic",
   partition:{train_groups:48,dev_groups:12,heldout_same_groups:32,heldout_unseen_grammar_groups:32,
     members_per_pair:2,unique_geometry_hashes:fingerprints.size,group_disjoint:true,training_only_export:true},
   training:{algorithm:"deterministic pairwise logistic SGD",epochs:EPOCHS,feature_dimension:ND,
     train_worlds:corpus.length,train_decisions:corpus.reduce((n,x)=>n+x.decisions.length,0),
     compared_hard_negative_actions:actual.comparisons,train_jsonl_sha256:sha256(trainJSONL),
     memory_weights_sha256:actual.sha256,sham_weights_sha256:sham.sha256},
   evaluation:scores,budget:BUDGET,
   safety:{all_moves_host_legal:true,paint_on_unprotected_passable_cells_only:true,native_receipt:false,game_modifications:0},
   limitations:["Synthetic generator/host, not CyberSand native evidence",
     "Autonomous direction choices only: paint material determined by host, not model",
     "Counterfactual good-arm pairs are correlated and share nearby observations",
     "HELDOUT_unseen is a changed synthetic grammar, not an external engine",
     "The sham policy is denied all route-history features but shares input dimension",
     "No official game DEV evaluation or production model checkpoint"]};
 return {report,trainJSONL,svg:gallery(sets.HELDOUT_unseen,models)};
}
if(typeof require!=="undefined"&&require.main===module){
 const fs=require("fs"),out=process.argv[2];
 if(!out||fs.existsSync(out))throw Error("provide a NEW output directory");
 const r=main();fs.mkdirSync(out,{recursive:false});
 fs.writeFileSync(out+"/report.json",JSON.stringify(r.report,null,2)+"\n");
 fs.writeFileSync(out+"/TRAIN_only.jsonl",r.trainJSONL);
 fs.writeFileSync(out+"/gallery.svg",r.svg);
 console.log(JSON.stringify(r.report,null,2));
}
