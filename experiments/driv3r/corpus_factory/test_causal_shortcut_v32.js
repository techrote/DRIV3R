'use strict';
// Corpus Factory v32 independent structural, leakage and host-replay acceptance.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto');
const gen=require('./causal_shortcut_v32.js'),host=require('./causal_gate_v26.js'),view=require('./bounded_view_v30.js');
const hash=v=>crypto.createHash('sha256').update(Buffer.isBuffer(v)||typeof v==='string'?v:JSON.stringify(v)).digest('hex');
const read=(dir,n)=>fs.readFileSync(path.join(dir,n),'utf8').trimEnd().split('\n').map(JSON.parse);
function verify(dir){
  const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
  assert.equal(manifest.schema,'driv3r/corpus_factory/causal-shortcut-v32/manifest');
  assert.equal(manifest.generator_sha256,hash(fs.readFileSync(path.join(__dirname,'causal_shortcut_v32.js'))));
  for(const dep of manifest.dependencies)assert.equal(dep.sha256,hash(fs.readFileSync(path.join(__dirname,dep.file))));
  assert.equal(manifest.restricted_first_action_theoretical_ceiling,0.25);
  const allInputs=new Map(),geometries=new Map();let episodeCount=0,decisionCount=0,eraseCount=0,pairCount=0;
  for(const [split,stats] of Object.entries(manifest.splits)){
    const eps=read(dir,split+'_episodes.jsonl'),rows=read(dir,split+'_decisions.jsonl');
    assert.equal(eps.length,stats.episodes);assert.equal(rows.length,stats.decisions);
    assert.equal(stats.strict_four_way_worlds*4,stats.episodes);
    assert.equal(stats.rejected.length,0);
    assert.equal(stats.exact_duplicates,0);
    const states=new Map(),pairs=new Map();
    for(const ep of eps){
      assert.equal(ep.terminal_success,true);assert.equal(ep.erases,2);
      assert.equal(ep.difficulty.post_paint_accepted_moves,160);
      assert.equal(ep.gate_causal_control.arm,Number(/arm(\d+)$/.exec(ep.episode_id)[1]));
      assert.equal(ep.initial_history.events.some(x=>x.kind==='GATE_TRANSITION'&&x.open===true),true);
      assert.equal(ep.initial_history.events.some(x=>x.kind==='GATE_TRANSITION'&&x.open===false),false);
      assert.ok(ep.first_action_regret>=2);
      assert.deepEqual(ep.hidden_interventions.latest_wrong_target_index,4);
      const p=ep.pair_id;if(!pairs.has(p))pairs.set(p,[]);pairs.get(p).push(ep);
      const prior=geometries.get(ep.geometry_sha256);assert.ok(prior===undefined||prior===split);geometries.set(ep.geometry_sha256,split);
      const m=/^([A-Z]+)-v32-g(\d+)-v(\d+)-arm(\d+)$/.exec(ep.episode_id);assert.ok(m);
      const [,part,g,v,arm]=m;assert.equal(part,split);
      const w=gen.world(split,Number(g),Number(v));assert.equal(w.world_sha256,ep.world_sha256);
      // Independent causal intervention audit: same graph and status, actual
      // accepted paint opened the root-east edge; freeze it closed and require
      // the published regret to match freshly computed suffix values.
      const root=w.root,east=w.nidx.get('1,0');
      assert.equal(w.rule.gate_edge[0],'0,0');assert.equal(w.rule.gate_edge[1],'1,0');
      assert.equal(host.edgeAllowed(w,root,east,0),false);
      assert.equal(host.edgeAllowed(w,root,east,1),true);
      assert.equal(host.edgeAllowed(w,root,east,1,false),false);
      const preparedForControl=gen.prepare(w,split,Number(g),Number(v),Number(arm));
      const ss=[preparedForControl.h.pos,preparedForControl.h.prev,preparedForControl.h.status,preparedForControl.h.latched];
      const dyn=gen.oracle(w).rank(ss),frozen=gen.oracle(w,{fixed:false}).rank(ss),top=frozen[0];
      const actualChoice=dyn.find(x=>x.action===top.action);
      assert.equal(ep.gate_causal_control.frozen_first,top.action);
      assert.equal(ep.gate_causal_control.dynamic_first,dyn[0].action);
      assert.equal(ep.gate_causal_control.strict_regret,actualChoice?actualChoice.cost-dyn[0].cost:Infinity);
      if(Number(arm)===1){assert.equal(dyn[0].action,'MOVE_E');assert.equal(top.action,'MOVE_S');
        assert.ok(ep.gate_causal_control.strict_regret>=2);} 
      else assert.equal(ep.gate_causal_control.strict_regret,0,'non-east arms are explicit negative controls');
      const p0=gen.prepare(w,split,Number(g),Number(v),Number(arm));
      assert.equal(p0.accepted,ep.difficulty.scripted_accepted_actions);
      assert.deepEqual(p0.h.history,ep.initial_history);
      assert.equal(hash(host.observe(p0.h)),ep.initial_full_observation_sha256);
      states.set(ep.episode_id,{h:p0.h,ep,count:0,erases:0,first:null,w,planner:gen.oracle(w)});
    }
    for(const row of rows){
      const s=states.get(row.episode_id);assert.ok(s,'known episode');
      assert.equal(row.step,s.count);
      const o=host.observe(s.h);
      assert.deepEqual(o,{...row.state,history:o.history},'state fields reflect host');
      // Reconstruct the model observation strictly from episode history plus
      // decision deltas, without consulting hidden interventions.
      if(!s.rebuilt)s.rebuilt=structuredClone(s.ep.initial_history);
      s.rebuilt.events.push(...row.events_added);s.rebuilt.positions.push(...row.positions_added);
      const reconstructed={...row.state,history:s.rebuilt};
      assert.equal(hash(reconstructed),row.model_input_sha256);
      assert.equal(hash(o),row.model_input_sha256,'replayed host matches serialized model input');
      assert.equal(hash(row.bounded_model_input),row.bounded_model_input_sha256);
      assert.deepEqual(view.compress(o),row.bounded_model_input);
      assert.ok(JSON.stringify(row.bounded_model_input).length<=8192);
      const prior=allInputs.get(row.model_input_sha256);
      assert.ok(prior===undefined||(prior.split===split&&prior.target===row.target),'no cross-split leak or conflicting label');
      allInputs.set(row.model_input_sha256,{split,target:row.target});
      const best=s.planner.rank([s.h.pos,s.h.prev,s.h.status,s.h.latched])[0];
      assert.equal(row.target,best.action,'exact independently recomputed suffix oracle');
      if(s.count===0)s.first={obs:structuredClone(o),label:row.target};
      const receipt=host.apply(s.h,row.target);
      assert.deepEqual(receipt,row.receipt,'host action receipt reproduced');
      assert.equal(receipt.accepted,true);
      if(row.target==='ERASE')s.erases++;
      s.count++;
    }
    for(const s of states.values()){
      assert.equal(s.h.done,true);assert.equal(s.count,s.ep.steps);assert.equal(s.erases,2);
      assert.equal(host.correct(s.w,s.h.status).wrong,0);
      assert.ok(host.correct(s.w,s.h.status).count>=2);
      eraseCount+=s.erases;
    }
    for(const arms of pairs.values()){
      assert.equal(arms.length,4,'one set of four matched counterfactuals');
      const cases=arms.map(e=>states.get(e.episode_id).first),labels=new Set(cases.map(c=>c.label));
      assert.deepEqual([...labels].sort(),['BACKTRACK','MOVE_E','MOVE_S','MOVE_W'].sort());
      for(const mode of [{memory:false},{lastPaintOnly:true}])
        assert.equal(new Set(cases.map(c=>hash(view.compress(c.obs,mode)))).size,1,'all four projections identical');
      assert.equal(new Set(cases.map(c=>hash(c.obs.history.events.slice(-64)))).size,1,'recent64 identical');
      assert.equal(new Set(cases.map(c=>hash(c.obs))).size,4,'full history identifies all four');
    }
    pairCount+=pairs.size;episodeCount+=eps.length;decisionCount+=rows.length;
    for(const suffix of ['episodes','decisions']){
      const name=split+'_'+suffix+'.jsonl',buf=fs.readFileSync(path.join(dir,name));
      assert.equal(hash(buf),manifest.shards[name].sha256);
      assert.equal(buf.length,manifest.shards[name].bytes);
    }
    assert.ok(stats.strict_frozen_gate_first_disagreements>0,'gate effect established');
    assert.deepEqual(stats.first_action,{MOVE_W:stats.strict_four_way_worlds,MOVE_E:stats.strict_four_way_worlds,
      MOVE_S:stats.strict_four_way_worlds,BACKTRACK:stats.strict_four_way_worlds});
  }
  console.log('PASS four-way held-out alias / exact suffix oracle / full host replay / SHA-256 / split exclusion',
    JSON.stringify({pairs:pairCount,episodes:episodeCount,decisions:decisionCount,erasures:eraseCount}));
  return manifest;
}
if(require.main===module){
  const dir=process.argv[2]||path.join(fs.mkdtempSync(path.join(os.tmpdir(),'driv3r-v32-test-')),'data');
  if(!process.argv[2])gen.generate(dir,{quick:true});
  const manifest=verify(dir);
  if(!manifest.quick){assert.equal(manifest.splits.TRAIN.episodes,256);
    assert.equal(manifest.splits.DEV.episodes,128);assert.equal(manifest.splits.HELDOUT.episodes,128);}
}
module.exports={verify};
