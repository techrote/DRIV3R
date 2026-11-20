"use strict";
/* FORECAST-BREAK-EVEN-003 — deterministic decision-theory sensitivity.
 * Source: the receipt-replayed FORECAST-UTILITY-002 DEVELOPMENT aggregates.
 * Hypothetical reliability values are analytical expectations, NOT fresh game
 * rollouts, actual CyberSand observations or trained new model checkpoints.
 */
function forecastBreakEven(e) {
  const check=(ok,msg)=>{if(!ok)throw Error(msg)};
  const round=(n,p=6)=>+n.toFixed(p);
  const cohorts=[["native64_proxy",36],["native64_proxy",72],
    ["native64_proxy",128],["shadow128",128]];
  const rewards=[0,200,1000],reliabilities=[.5,.75,.875,.9375,.96875,.984375,1];
  check(e.study==="FORECAST-UTILITY-002"&&e.version==="driv3r-forecast-utility/1","Evidence version");
  check(e.action_receipts_replayed===82493&&e.independent_replay_episodes===1344&&
        e.tests_passed===16&&e.train_examples===432&&e.updates===720,
        "Evidence replay/optimizer identity");
  const index=new Map();
  for(const x of e.aggregates) {
    const k=[x.cohort,x.horizon,x.mode,x.arm].join(":");
    check(!index.has(k),"Duplicate evidence aggregate");
    index.set(k,x);
  }
  function get(cohort,h,mode,arm){
    const row=index.get([cohort,h,mode,arm].join(":"));
    check(row&&row.n===16&&Number.isFinite(row.mean_utility),"Missing balanced DEVELOPMENT aggregate");
    return row;
  }
  const utility=(r,reward)=>r.mean_utility+(reward-1000)*r.goals/r.n;
  const frontier=[];
  for(const [cohort,horizon] of cohorts)for(const reward of rewards) {
    const E0=get(cohort,horizon,"early","lazy"),E1=get(cohort,horizon,"early","overlap"),
          L0=get(cohort,horizon,"late","lazy"),L1=get(cohort,horizon,"late","overlap"),
          I0=get(cohort,horizon,"irrelevant","lazy"),I1=get(cohort,horizon,"irrelevant","overlap");
    const ue0=utility(E0,reward),ue1=utility(E1,reward),
          ul0=utility(L0,reward),ul1=utility(L1,reward),ui=utility(I0,reward);
    check(Math.abs(ui-utility(I1,reward))<1e-8,"Irrelevant edit altered utility");
    const cost=ue0-ue1,benefit=ul1-ul0;
    check(cost>0&&benefit>0,"Tradeoff signs changed");
    const threshold=benefit/(cost+benefit);
    const overlap=(ue1+ul1+ui)/3,oracle=(ue0+ul1+ui)/3;
    const sweep=reliabilities.map(p=>{
      // Relevant modes are balanced, conditional reliability is symmetric.
      const plus=p*benefit-(1-p)*cost>0;
      const minus=(1-p)*benefit-p*cost>0;
      const earlyReplanChance=(1-p)*Number(plus)+p*Number(minus);
      const lateReplanChance=p*Number(plus)+(1-p)*Number(minus);
      const u=(ue0-earlyReplanChance*cost+ul0+lateReplanChance*benefit+ui)/3;
      check(u+1e-8>=overlap&&u<=oracle+1e-8,"Non-optimal decision rule or oracle breach");
      return {reliability:p,positive_replan:plus,negative_replan:minus,
        expected_utility:round(u,4),gain_vs_overlap:round(u-overlap,4),
        regret_to_oracle:round(oracle-u,4)};
    });
    frontier.push({cohort,horizon,reward,original_n_per_mode:16,
      original_goal_counts:{early_lazy:E0.goals,early_overlap:E1.goals,
        late_lazy:L0.goals,late_overlap:L1.goals},
      avoidable_early_cost:round(cost,4),late_intervention_benefit:round(benefit,4),
      negative_signal_break_even_reliability:round(threshold,8),
      overlap_utility:round(overlap,4),oracle_utility:round(oracle,4),sweep});
  }
  const original=frontier.filter(x=>x.reward===1000);
  const mean=(a,k)=>a.reduce((s,x)=>s+x[k],0)/a.length;
  check(Math.abs(mean(original,"overlap_utility")-e.summary.overlap.mean_utility)<.002,
        "Failed to reconstruct receipt-replayed overlap policy");
  check(Math.abs(mean(original,"oracle_utility")-e.summary.privileged_oracle.mean_utility)<.002,
        "Failed to reconstruct privileged oracle");
  const combined=reliabilities.map(p=>{
    const rows=original.map(x=>x.sweep.find(z=>z.reliability===p));
    return {reliability:p,expected_utility:round(mean(rows,"expected_utility"),4),
      gain_vs_overlap:round(mean(rows,"gain_vs_overlap"),4),
      oracle_regret:round(mean(rows,"regret_to_oracle"),4),
      cohorts_using_negative_signal:rows.filter(x=>!x.negative_replan).length};
  });
  check(Math.abs(combined[1].expected_utility-e.summary.overlap.mean_utility)<.002,
        "75% result differs from source campaign");
  check(combined[6].oracle_regret<.0002,"Perfect-signal limit");
  const h128=original.find(x=>x.cohort==="native64_proxy"&&x.horizon===128);
  check(h128.original_goal_counts.late_lazy===0&&h128.original_goal_counts.late_overlap===0,
        "Native64 128-turn censor incorrectly treated as goal completion");
  return {study:"FORECAST-BREAK-EVEN-003",version:"1",date:"2026-11-19",
    input_git_blob_sha1:"2be8ebbbce9da1b67b2bbfc36c801b5e5400a1b3",
    upstream_program_git_blob_sha1:"4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873",
    prior_source_reexecuted_receipts:82493,prior_source_reexecuted_episodes:1344,
    prior_optimizer_updates:720,new_optimizer_updates:0,new_game_rollouts:0,
    qualification:"Analytical sensitivity of previously exact-source-replayed DEVELOPMENT outcomes; not 7 new executed game sweeps",
    assumptions:"Early/late balanced, independent symmetric forecast errors exchangeable across seeds; reward = goal reward - 2*steps - 3*planner calls - backtracks",
    native_64_paint_censor_intact:true,shadow128_separately_identified:true,
    external_future_signal_not_current_host_observation:true,
    native_receipts:0,structural_heldout_opened:false,
    frontier,combined};
}
if(typeof module!=="undefined")module.exports={forecastBreakEven};
if(typeof require!=="undefined"&&typeof module!=="undefined"&&require.main===module){
  const fs=require("node:fs"),crypto=require("node:crypto"),path=require("node:path");
  function gitBlob(file){
    const b=fs.readFileSync(file);
    return crypto.createHash("sha1").update(Buffer.from("blob "+b.length+"\0")).update(b).digest("hex");
  }
  const dir=__dirname,source=path.join(dir,"forecast_utility_lab.js"),
        input=path.join(dir,"FORECAST_UTILITY_EVIDENCE_2026-11-19.json");
  if(gitBlob(source)!=="4ed7e4d2c11f15a3d3160fecb027eaefb2e2d873" ||
     gitBlob(input)!=="2be8ebbbce9da1b67b2bbfc36c801b5e5400a1b3")
    throw Error("Upstream source/evidence Git blob mismatched");
  const result=forecastBreakEven(JSON.parse(fs.readFileSync(input,"utf8")));
  if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(result,null,2)+"\n",{flag:"wx"});
  console.log(JSON.stringify({study:result.study,combined:result.combined},null,2));
}
