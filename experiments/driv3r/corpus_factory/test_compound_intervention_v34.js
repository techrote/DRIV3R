'use strict';
// Independent verification of compact reconstruction, host legality, exact regret,
// terminal success, 4-way memory alias and topology-disjoint splits for v34.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const gen=require('./compound_intervention_v34.js'),base=require('./causal_shortcut_v32.js');
const host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const hash=x=>crypto.createHash('sha256').update(Buffer.isBuffer(x)||typeof x==='string'?x:JSON.stringify(x)).digest('hex');
function read(d,n){return fs.readFileSync(path.join(d,n),'utf8').trimEnd().split('\n').map(JSON.parse);}
function run(d){
  const m=JSON.parse(fs.readFileSync(path.join(d,'manifest.json')));
  assert.equal(m.schema,'driv3r/corpus_factory/offpolicy-recovery-v34/manifest');
  assert.equal(m.generator_sha256,hash(fs.readFileSync(path.join(__dirname,m.generator))));
  for(const dep of m.dependencies)assert.equal(dep.sha256,hash(fs.readFileSync(path.join(__dirname,dep.file))));
  assert.equal(m.first_action_recent64_uniform_ceiling,0.25);
  const owners=new Map(),geo=new Map(),seenTotal=new Set();let totalEpisodes=0,totalDecisions=0,totalWrong=0,totalErases=0,totalRegret=0;
  for(const [split,s] of Object.entries(m.splits)){
    const eps=read(d,split+'_episodes.jsonl'),rows=read(d,split+'_decisions.jsonl');
    assert.equal(eps.length,s.episodes);assert.equal(rows.length,s.decisions);
    assert.equal(s.rejected.length,0);assert.equal(s.qualified_four_way_families*12,eps.length);
    assert.equal(s.offpolicy_recovery_episodes,eps.length/3);
    const states=new Map(),families=new Map(),worldCache=new Map();let perSplitRegret=0,perSplitErases=0,perSplitWrong=0,perSplitPaint=0,perSplitPaintRegret=0;
    for(const ep of eps){
      assert.equal(ep.terminal_success,true);assert.equal(ep.erasures,ep.mode==='wrong-first-plus-paint'?3:2);
      assert.ok(['clean','wrong-first','wrong-first-plus-paint'].includes(ep.mode));
      const match=/^([A-Z]+)-v34-g(\d+)-v(\d+)-arm(\d+)-(clean|wrong-first|wrong-first-plus-paint)$/.exec(ep.episode_id);assert.ok(match);
      const [,sp,g,v,arm,mode]=match;assert.equal(sp,split);assert.equal(mode,ep.mode);
      const key=`${split}:${g}:${v}`;if(!worldCache.has(key)){const w=base.world(split,Number(g),Number(v));worldCache.set(key,{w,planner:base.oracle(w)});}
      const {w,planner}=worldCache.get(key),prepared=base.prepare(w,split,Number(g),Number(v),Number(arm));
      assert.equal(ep.world_sha256,w.world_sha256);assert.equal(ep.geometry_sha256,w.geometry_sha256);
      const prior=geo.get(w.geometry_sha256);assert.ok(!prior||prior===split);geo.set(w.geometry_sha256,split);
      const h=prepared.h,priorRank=planner.rank([h.pos,h.prev,h.status,h.latched]);
      assert.equal(ep.first_action_required,priorRank[0].action);
      assert.deepEqual(ep.first_legal_action_regrets,gen.optionsCertificate(priorRank));
      const wrong=gen.chooseWrong(priorRank,Number(g),Number(v),Number(arm));
      assert.equal(ep.accepted_bad_first_cost,wrong.cost);
      if(mode!=='clean'){
        assert.ok(wrong.cost-priorRank[0].cost>=2);
        const receipt=host.apply(h,wrong.action);assert.deepEqual(receipt,ep.off_policy_accepted_intervention.receipt);
        assert.equal(ep.off_policy_accepted_intervention.action,wrong.action);
        assert.equal(ep.realised_first_action_regret,wrong.cost-priorRank[0].cost);
        assert.equal(ep.steps,wrong.cost-1+(ep.off_policy_paint_intervention?.strict_regret||0));perSplitRegret+=ep.realised_first_action_regret;perSplitWrong++;
        if(mode==='wrong-first-plus-paint'){assert.ok(ep.off_policy_paint_intervention.strict_regret>=2);perSplitPaint++;perSplitPaintRegret+=ep.off_policy_paint_intervention.strict_regret;}
      }else{
        assert.equal(ep.off_policy_accepted_intervention,null);
        assert.equal(ep.realised_first_action_regret,0);assert.equal(ep.steps,priorRank[0].cost);
      }
      assert.deepEqual(h.history,ep.initial_history);
      assert.equal(hash(host.observe(h)),ep.initial_full_observation_sha256);
      states.set(ep.episode_id,{ep,h,w,planner,step:0,erases:0,wrongPaint:0,regretPaint:0,rebuilt:structuredClone(ep.initial_history),first:null});
      if(!families.has(ep.family_id))families.set(ep.family_id,[]);families.get(ep.family_id).push(ep);
    }
    for(const row of rows){
      const t=states.get(row.episode_id);assert.ok(t);
      assert.equal(row.step,t.step);
      const obs=host.observe(t.h),rank=t.planner.rank([t.h.pos,t.h.prev,t.h.status,t.h.latched]);
      assert.deepEqual(row.state,Object.fromEntries(Object.entries(obs).filter(([k])=>k!=='history')));
      t.rebuilt.events.push(...row.events_added);t.rebuilt.positions.push(...row.positions_added);
      assert.equal(hash({...row.state,history:t.rebuilt}),row.model_input_sha256);
      assert.equal(hash(obs),row.model_input_sha256);
      assert.equal(hash(row.bounded_model_input),row.bounded_model_input_sha256);
      assert.deepEqual(row.bounded_model_input,view.compress(obs));
      assert.ok(JSON.stringify(row.bounded_model_input).length<=8192);
      assert.deepEqual(row.oracle_action_costs,gen.optionsCertificate(rank));
      assert.equal(row.target,rank[0].action);
      for(const z of row.oracle_action_costs){
        assert.ok(obs.legal_actions.includes(z.action));
        assert.ok(z.regret===null||z.regret>=0);
      }
      const prior=owners.get(row.model_input_sha256);
      assert.ok(!prior||prior.split===split&&prior.target===row.target);
      owners.set(row.model_input_sha256,{split,target:row.target});seenTotal.add(row.model_input_sha256);
      if(t.step===0)t.first={obs:structuredClone(obs),target:row.target};
      const actual=row.applied_action;assert.ok(obs.legal_actions.includes(actual));
      if(row.intervention){assert.equal(row.intervention,'HOST_ACCEPTED_WRONG_MATERIAL');assert.equal(t.ep.mode,'wrong-first-plus-paint');assert.equal(row.training_weight,0);assert.equal(t.wrongPaint,0);
        const alt=rank.find(x=>x.action===actual);assert.ok(alt&&alt.cost-rank[0].cost>=2);assert.ok(row.target.startsWith('PAINT_'));assert.notEqual(actual,row.target);
        assert.equal(t.ep.off_policy_paint_intervention.step,t.step);assert.equal(t.ep.off_policy_paint_intervention.strict_regret,alt.cost-rank[0].cost);
        t.regretPaint+=alt.cost-rank[0].cost;t.wrongPaint++;
      }else assert.equal(actual,row.target);
      const r=host.apply(t.h,actual);assert.deepEqual(r,row.receipt);assert.equal(r.accepted,true);
      if(actual==='ERASE')t.erases++;
      t.step++;
    }
    for(const t of states.values()){
      assert.equal(t.h.done,true);assert.equal(t.step,t.ep.steps);
      assert.equal(t.erases,t.ep.mode==='wrong-first-plus-paint'?3:2);assert.equal(host.correct(t.w,t.h.status).wrong,0);perSplitErases+=t.erases;
      assert.equal(t.wrongPaint,t.ep.mode==='wrong-first-plus-paint'?1:0);assert.equal(t.ep.realised_combined_regret,t.ep.realised_first_action_regret+t.regretPaint);
    }
    for(const f of families.values()){
      assert.equal(f.length,12);
      const cs=f.filter(ep=>ep.mode==='clean'),ws=f.filter(ep=>ep.mode==='wrong-first'),ps=f.filter(ep=>ep.mode==='wrong-first-plus-paint');
      assert.equal(cs.length,4);assert.equal(ws.length,4);assert.equal(ps.length,4);
      const cleanFirst=cs.map(ep=>states.get(ep.episode_id).first);
      assert.deepEqual(new Set(cleanFirst.map(z=>z.target)),new Set(['BACKTRACK','MOVE_W','MOVE_S','MOVE_E']));
      for(const conf of [{memory:false},{lastPaintOnly:true}])
        assert.equal(new Set(cleanFirst.map(z=>hash(view.compress(z.obs,conf)))).size,1);
      assert.equal(new Set(cleanFirst.map(z=>hash(z.obs.history.events.slice(-64)))).size,1);
      assert.equal(new Set(cleanFirst.map(z=>hash(z.obs))).size,4);
      for(const ep of [...ws,...ps])assert.equal(ep.steps+1-cs.find(x=>x.hidden_interventions.older_wrong_target_index===ep.hidden_interventions.older_wrong_target_index).steps,ep.realised_combined_regret);
      assert.equal(cs.filter(x=>x.difficulty.strict_gate_causal_control).length,1);
    }
    assert.equal(perSplitWrong,s.accepted_detrimental_first_actions);
    assert.equal(perSplitErases,s.corrective_erasures);
    assert.equal(perSplitRegret,s.total_proven_first_action_regret);
    assert.equal(perSplitPaint,s.accepted_wrong_material_paint_interventions);
    assert.equal(perSplitPaintRegret,s.total_proven_paint_regret);
    for(const suffix of ['episodes','decisions']){
      const filename=split+'_'+suffix+'.jsonl',buf=fs.readFileSync(path.join(d,filename));
      assert.equal(hash(buf),m.shards[filename].sha256);assert.equal(buf.length,m.shards[filename].bytes);
    }
    totalEpisodes+=eps.length;totalDecisions+=rows.length;totalWrong+=perSplitWrong;totalRegret+=perSplitRegret;totalErases+=perSplitErases;
  }
  console.log('PASS v34 host-legal compound navigation+paint recovery / strict suffix costs / leak controls / all compact hashes',JSON.stringify({episodes:totalEpisodes,decisions:totalDecisions,wrongFirst:totalWrong,regret:totalRegret,erasures:totalErases,distinctInputs:seenTotal.size}));
  return {episodes:totalEpisodes,decisions:totalDecisions,wrongFirst:totalWrong,regret:totalRegret,erasures:totalErases};
}
if(require.main===module){const d=process.argv[2]||path.join(fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v34-test-')),'quick');
  if(!process.argv[2])gen.generate(d,{quick:true});run(d);}
module.exports={run};
