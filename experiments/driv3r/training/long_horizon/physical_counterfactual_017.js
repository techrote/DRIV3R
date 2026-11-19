"use strict";
// PHYSICAL-COUNTERFACTUAL-017. Explicit opt-in, bounded CPU study. Never executes on import/push.
// Original source files and negative ledgers preserved. No actual CyberSand ABI claims.
const fs=require("fs"),crypto=require("crypto"),vm=require("vm"),path=require("path");
const PINNED_HOST_BLOB="45a426f893203f93fa9a3972d63bd4a110ceaa3d";

function vector(c,h,cap){
 const ob=h.observation(), tail=h.log.slice(-8),dist=c.bfs(ob.pos,c.TARGETS[0],h.walls).length;
 return[1,ob.tick/128,(cap-ob.tick)/128,dist/32,+(h.horizon===36),+(h.horizon===72),+(h.horizon===128),+(cap===64),+(cap===128),h.walls.size/30,tail.filter(r=>r.action in c.DIR).length/8,tail.filter(r=>r.action==="ERASE").length/8,+(ob.material_here===1),+(ob.menu.includes("W"))];
}

function twins(c,seed,hor,cap,split){
 const prefix=c.prefix(seed,hor,split,cap),A=c.key(c.TARGETS[0]);
 if(!prefix.paintedA||!prefix.hazard||c.eq(prefix.pos,c.TARGETS[0]))throw Error("ineligible prefix");
 const pair=[0,1].map(damage=>{
  let root=prefix.clone();root.ink[A]=1;root.damage=!!damage;if(damage)delete root.ink[A];
  return{root,damage,pre:prefix.log.length,x:vector(c,root,cap),ob:JSON.stringify(root.observation()),hash:c.hash(JSON.stringify(root.log))};
 });
 if(JSON.stringify(pair[0].x)!==JSON.stringify(pair[1].x)||pair[0].ob!==pair[1].ob||pair[0].hash!==pair[1].hash)throw Error("counterfactual leaked into observed features");
 return pair;
}

function predict(w,x){let z=0;for(let i=0;i<w.length;i++)z+=w[i]*x[i];return 1/(1+Math.exp(-Math.max(-25,Math.min(25,z))))}

function fit(rows){
 let w=Array(rows[0].x.length).fill(0);
 for(let it=0;it<1200;it++){let grad=Array(w.length).fill(0);for(let r of rows){let e=predict(w,r.x)-r.y;for(let j=0;j<w.length;j++)grad[j]+=e*r.x[j]}for(let j=0;j<w.length;j++)w[j]-=.6/(1+it/800)*(grad[j]/rows.length+.0005*w[j])}
 return w;
}

function replayCF(c,executed,seed,hor,split,pre,damage){
 const original=executed.log,other=new c.Host(seed,hor,split),A=c.key(c.TARGETS[0]);
 for(let i=0;i<original.length;i++){
  if(i===pre){if(!other.paintedA||!other.hazard)throw Error("bad intervention boundary");other.ink[A]=1;other.damage=!!damage;if(damage)delete other.ink[A]}
  if(JSON.stringify(other.step(original[i].action))!==JSON.stringify(original[i]))throw Error("source receipt diverged "+[seed,hor,split,damage,i]);
 }
 if(other.tick!==executed.tick||JSON.stringify(other.ink)!==JSON.stringify(executed.ink)||!c.eq(other.pos,executed.pos))throw Error("final state diverged");
 return c.assertReceipts(original);
}

function summarize(rows,p,c){
 let a=rows.filter(r=>r.policy===p),n=a.length;
 return{records:n,goals:a.reduce((s,r)=>s+r.goal,0),meanUtility:+(a.reduce((s,r)=>s+r.utility,0)/n).toFixed(3),calls:a.reduce((s,r)=>s+r.calls,0),checks:a.reduce((s,r)=>s+r.checks,0),actions:Object.fromEntries(c.ACT.map(k=>[k,a.reduce((s,r)=>s+r.counts[k],0)]))};
}

function study(c){
 const time=Date.now(),training=[],caps=c.CAPS,policies=["never","late","guard","learner","prefix_oracle"];
 for(let seed=40100;seed<40180;seed++)for(let [hor,cap] of caps)for(let t of twins(c,seed,hor,cap,"train"))training.push({x:t.x,y:t.damage});
 const w=fit(training),scores=training.map(t=>predict(w,t.x)),calibration=[];
 for(let cutoff of [.25,.5,.75]){
  let goal=0,utility=0,calls=0;
  for(let seed=40100;seed<40180;seed++)for(let [hor,cap] of caps)for(let t of twins(c,seed,hor,cap,"train")){let o=c.finish(t.root,cap,predict(w,t.x)>=cutoff?"feasible":"never").out;goal+=o.goal;utility+=o.utility;calls+=o.calls}
  calibration.push({cutoff,goals:goal,meanUtility:+(utility/training.length).toFixed(3),calls});
 }
 calibration.sort((a,b)=>b.goals-a.goals||b.meanUtility-a.meanUtility||a.cutoff-b.cutoff);
 const cutoff=calibration[0].cutoff,rows=[],paths=[],negatives=[];let receipts=0,pairCount=0;
 for(let seed=50100;seed<50160;seed++)for(let [hor,cap] of caps){
  const pair=twins(c,seed,hor,cap,"dev");pairCount++;
  for(let t of pair){
   let p=predict(w,t.x),run={};
   for(let policy of policies){
    let mode=policy==="guard"?"feasible":policy==="learner"?(p>=cutoff?"feasible":"never"):policy==="prefix_oracle"?(t.damage?"feasible":"never"):policy;
    let executed=c.finish(t.root,cap,mode),o=executed.out;
    receipts+=replayCF(c,executed.host,seed,hor,"dev",t.pre,t.damage);
    const rec={seed,hor,cap,damage:t.damage,policy,mode,goal:o.goal,utility:o.utility,calls:o.calls,checks:o.checks,ticks:o.ticks,counts:o.counts,receiptHash:o.hash};
    rows.push(rec);run[policy]=rec;
    if([50101,50114,50129,50145,50158].includes(seed)&&["never","guard","prefix_oracle"].includes(policy))paths.push({seed,hor,cap,damage:t.damage,policy,goal:o.goal,positions:o.positions});
   }
   if(run.learner.goal<run.guard.goal||run.learner.goal<run.prefix_oracle.goal||run.learner.utility<run.guard.utility)negatives.push({seed,hor,cap,damage:t.damage,learner:run.learner.goal,guard:run.guard.goal,oracle:run.prefix_oracle.goal});
  }
 }
 const bad=new c.Host(77,36,"train");bad.step("E");let rejected=0;
 for(let forged of [[{...bad.log[0],action:"TELEPORT"}],[{...bad.log[0],after:[99,99]}]]){try{c.assertReceipts(forged)}catch(e){rejected++}}
 if(training.length!==640||rows.length!==2400||pairCount!==240||receipts<5000||rejected!==2)throw Error("campaign size/integrity");
 const byHorizon={};for(let [hor,cap] of caps)byHorizon[`${hor}/${cap}`]=Object.fromEntries(policies.map(p=>[p,rows.filter(r=>r.hor===hor&&r.cap===cap&&r.policy===p).reduce((s,r)=>s+r.goal,0)]));
 let t0=Date.now(),checksum=0;for(let i=0;i<100000;i++)checksum+=predict(w,training[i%training.length].x);
 return{experiment:"PHYSICAL_COUNTERFACTUAL_017",qualification:"Fresh same-source synthetic replay with explicit hidden material intervention; not native acceptance",setup:{trainWorlds:80,devWorlds:60,trainRecords:training.length,optimizerSteps:1200,devTwinPairs:pairCount,devPolicyRollouts:rows.length,replayedReceipts:receipts,corruptionRejections:rejected,horizonCaps:caps},training:{weights:w,cutoff,calibration,labelPrevalence:training.reduce((s,r)=>s+r.y,0)/training.length,probabilityRange:[Math.min(...scores),Math.max(...scores)]},development:{summary:Object.fromEntries(policies.map(p=>[p,summarize(rows,p,c)])),byHorizon,negativeCases:negatives,rows},paths,performance:{v8CampaignMs:Date.now()-time,predictBatch100kMs:Date.now()-t0,predictChecksum:checksum}};
}

function gallery(paths){
 let s=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${Math.ceil(Math.min(paths.length,30)/3)*152}"><rect width="100%" height="100%" fill="#121a24"/>`];
 for(let i=0;i<Math.min(paths.length,30);i++){let r=paths[i],x=i%3*400,y=Math.floor(i/3)*152,pt=r.positions.map(a=>`${x+12+a[0]*25},${y+18+a[1]*12}`).join(" ");s.push(`<g><rect x="${x+4}" y="${y+4}" width="392" height="144" fill="#263443" stroke="#496678"/><polyline points="${pt}" fill="none" stroke="${r.goal?"#83dbbc":"#ff9890"}" stroke-width="2"/><text x="${x+12}" y="${y+138}" fill="white" font-size="11" font-family="monospace">${r.policy} seed ${r.seed} ${r.hor}/${r.cap} erase=${r.damage} goal=${r.goal}</text></g>`)}
 return s.concat("</svg>").join("\n")+"\n";
}

function loadHost(){const src=fs.readFileSync(path.join(__dirname,"physical_inspection_016.js"));const blob=crypto.createHash("sha1").update("blob "+src.length+"\0").update(src).digest("hex");if(blob!==PINNED_HOST_BLOB)throw Error("host source drift, requalification needed");return vm.runInNewContext(src.toString("utf8")+"\n;({Host,prefix,finish,bfs,TARGETS,CAPS,DIR,ACT,key,eq,assertReceipts,hash})",{})}
if(require.main===module){const out=process.argv[2];if(!out||!path.isAbsolute(out))throw Error("Run explicitly: node physical_counterfactual_017.js /absolute/output/dir");const report=study(loadHost());report.performance.nodeRssBytes=process.memoryUsage().rss;report.performance.nodePeakRssKiB=process.resourceUsage().maxRSS;fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,"PHYSICAL_COUNTERFACTUAL_017_EVIDENCE.json"),JSON.stringify(report,null,2)+"\n");fs.writeFileSync(path.join(out,"PHYSICAL_COUNTERFACTUAL_017_PATHS.svg"),gallery(report.paths));console.log(JSON.stringify({setup:report.setup,training:report.training,development:report.development.summary,cohorts:report.development.byHorizon,performance:report.performance},null,2))}
module.exports={study,gallery,loadHost};
