'use strict';
// DRIV3R Corpus Factory v37: counterfactual material-tool semantics.
// Synthetic off-host one-step forks, NOT native CyberSand trajectories.
// No policy learner. A live public law is installed immediately before each
// decision; all previously observed receipts remain under the old law.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const SPLITS=['TRAIN','DEV','HELDOUT'];
const LAWS=['normal','inverted'];
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
function assert(x,msg){if(!x)throw Error(msg);}
function fileHash(file){return new Promise((resolve,reject)=>{const h=crypto.createHash('sha256'),s=fs.createReadStream(file);s.on('data',b=>h.update(b));s.on('error',reject);s.on('end',()=>resolve(h.digest('hex')));});}
async function* jsonl(file){const input=fs.createReadStream(file),rl=readline.createInterface({input,crlfDelay:Infinity});try{for await(const line of rl)if(line)yield JSON.parse(line);}finally{rl.close();input.destroy();}}
function writer(file){const fd=fs.openSync(file,'wx'),hash=crypto.createHash('sha256');let rows=0,bytes=0;return {write(x){const b=JSON.stringify(x)+'\n';fs.writeSync(fd,b);hash.update(b);rows++;bytes+=Buffer.byteLength(b);},finish(){fs.closeSync(fd);return {rows,bytes,sha256:hash.digest('hex')};}};}
function swap(a){return a==='PAINT_A'?'PAINT_B':a==='PAINT_B'?'PAINT_A':a;}
function output(action,law){assert(LAWS.includes(law),'unknown law');return law==='normal'?action:swap(action);}
function desired(input){const t=input.targets.find(x=>x[0]===input.p);assert(t,'paint target absent');return 'PAINT_'+t[1];}
function selected(row){const a=row.action;if(a==='PAINT_A'||a==='PAINT_B'||a==='ERASE'||a==='STOP')return true;
  // Deterministic, stratified navigation negatives: not labels from an oracle re-run.
  return ['MOVE_N','MOVE_S','MOVE_E','MOVE_W','BACKTRACK'].includes(a)&&parseInt(sha(row.input).slice(0,8),16)%12===0;}
function fork(row,law){const input={...row.input,law:{kind:'public_tool_output_mapping',scope:'FROM_THIS_DECISION_ONWARD',paint_A_yields:law==='normal'?'A':'B',paint_B_yields:law==='normal'?'B':'A'}};
  const action=law==='normal'?row.action:swap(row.action);
  assert(input.legal.includes(action),'transformed action not legal');
  assert(output(action,law)===row.action,'physical action effect not preserved');
  if(row.action.startsWith('PAINT_')){
    assert(row.action===desired(row.input),'source paint disagrees with required material');
    assert(input.legal.includes('PAINT_A')&&input.legal.includes('PAINT_B'),'pair lacks both paint actions');
    assert(input.local[0][4]==='EMPTY','source target not empty');
  }
  return {input,action,weight:row.weight};
}
function geometry(row){return sha({cells:row.input.cells,edges:row.input.edges,protected:row.input.protected,gate:row.input.gate.slice(0,2)});}
async function generate(parent,dest){assert(!fs.existsSync(dest),'refuse overwrite');
  const original=JSON.parse(fs.readFileSync(path.join(parent,'manifest.json')));
  assert(original.schema==='driv3r/corpus_factory/trainer-safe-v35/manifest','source must be v35');
  // A partial, modified, or renamed parent must never silently become a new study.
  for(const [file,stat] of Object.entries(original.shards)){
    const f=path.join(parent,file);assert(fs.statSync(f).size===stat.bytes,'parent size mismatch '+file);
    assert(await fileHash(f)===stat.sha256,'parent hash mismatch '+file);
  }
  fs.mkdirSync(dest,{recursive:true});const splits={},shards={},owners=new Map(),worldOwners=new Map();
  for(const split of SPLITS){
    const policy=writer(path.join(dest,split+'_policy.jsonl'));
    const challenge=writer(path.join(dest,split+'_counterfactual.jsonl'));
    const seen=new Map(),worlds=new Set(),dist={PAINT_A:0,PAINT_B:0,ERASE:0,STOP:0,NAV:0};
    let n=0,pairCount=0,positivePairs=0,controlPairs=0,weightOne=0,weightZero=0,duplicates=0;
    for await(const row of jsonl(path.join(parent,split+'_policy.jsonl'))){n++;
      assert(row.input&&typeof row.action==='string'&&[0,1].includes(row.weight),'invalid source row');
      if(!selected(row))continue;
      const orig=sha(row.input),g=geometry(row),other=worldOwners.get(g);
      assert(!other||other===split,'cross-split geometry overlap');worldOwners.set(g,split);worlds.add(g);
      const a=fork(row,'normal'),b=fork(row,'inverted');
      const kind=row.action.startsWith('PAINT_')?'PAINT':row.action==='ERASE'?'ERASE':row.action==='STOP'?'STOP':'NAV';
      const changed=a.action!==b.action;
      assert(changed===(kind==='PAINT'),'false positive law impact');
      if(changed)positivePairs++;else controlPairs++;
      if(dist[row.action]!==undefined)dist[row.action]++;else dist.NAV++;
      for(const candidate of [a,b]){
        const k=sha(candidate.input),p=seen.get(k),owner=owners.get(k);
        assert(!owner||owner===split,'cross-split input overlap');owners.set(k,split);
        if(p!==undefined){assert(p===candidate.action,'conflicting same input');candidate.weight=0;duplicates++;}else seen.set(k,candidate.action);
        policy.write(candidate);if(candidate.weight)weightOne++;else weightZero++;
      }
      // Only paired evaluation examples, never include an ID or teacher cost in model input.
      if(split!=='TRAIN')challenge.write({normal:a,inverted:b,law_free_input_hash:orig,changes_optimal_action:changed});
      pairCount++;
    }
    assert(positivePairs>0&&controlPairs>0,'missing counterfactual or control');
    shards[split+'_policy.jsonl']=policy.finish();
    shards[split+'_counterfactual.jsonl']=challenge.finish();
    splits[split]={geometry_groups:worlds.size,source_rows_examined:n,paired_decisions:pairCount,policy_rows:pairCount*2,
      changed_action_pairs:positivePairs,unchanged_action_controls:controlPairs,weight_one:weightOne,weight_zero:weightZero,
      exact_duplicate_inputs:duplicates,source_actions:dist};
  }
  const manifest={schema:'driv3r/corpus_factory/material-law-forks-v37/manifest',synthetic_host_not_native_CyberSand:true,
    generated_utc:'2026-11-18',generator:'material_law_forks_v37.js',generator_sha256:sha(fs.readFileSync(__filename)),
    parent_schema:original.schema,parent_manifest_sha256:sha(fs.readFileSync(path.join(parent,'manifest.json'))),
    parent_shards:original.shards,law_intervention:'A public, instantaneous, future-only tool mapping installed at an already-reached v35 decision state. Prior action receipts refer to the pre-installation normal mapping.',
    proof_scope:'For a fixed observed physical state, swapping subsequent PAINT command tokens preserves every physical action effect in the synthetic symbolic host. Does NOT constitute fresh closed-loop replay after the law update.',
    model_input_contract:'Only row.input is model-visible; action and weight are external training/evaluation fields. Counterfactual shards are evaluation-only and carry paired answer labels.',
    exclusions:['No CyberSand native law is asserted','Not a new topology/physics implementation','Not a trained neural model','No autonomous law-change trajectory qualification'],splits,shards};
  fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');return manifest;
}
module.exports={generate,fork,output,swap,selected,sha};
if(require.main===module){const [parent,dest]=process.argv.slice(2);assert(parent&&dest,'usage: node material_law_forks_v37.js V35_DIR FRESH_OUT_DIR');
  generate(path.resolve(parent),path.resolve(dest)).then(m=>console.log(JSON.stringify(m.splits,null,2))).catch(e=>{console.error(e.stack);process.exitCode=1;});}
