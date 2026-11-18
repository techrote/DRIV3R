'use strict';
// DRIV3R Corpus Factory v35: honest, teacher-isolated training/held-out view of
// v34's actual host-accepted recovery trajectories. No policy training; synthetic only.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),readline=require('node:readline');
const SPLITS=['TRAIN','DEV','HELDOUT'];
const MODEL_KEYS=['v','p','legal','goal','ok','left','cells','edges','targets','protected','gate','local','recent','recent_positions','durable'];
const sha=x=>crypto.createHash('sha256').update(typeof x==='string'||Buffer.isBuffer(x)?x:JSON.stringify(x)).digest('hex');
const fileSha=file=>new Promise((resolve,reject)=>{const h=crypto.createHash('sha256');const s=fs.createReadStream(file);s.on('data',b=>h.update(b));s.on('error',reject);s.on('end',()=>resolve(h.digest('hex')));});
const lines=async function*(file){const stream=fs.createReadStream(file,{encoding:'utf8'}),r=readline.createInterface({input:stream,crlfDelay:Infinity});try{for await(const line of r)if(line.trim())yield JSON.parse(line);}finally{r.close();stream.destroy();}};
function assert(cond,msg){if(!cond)throw Error(msg);}
function input(r){
  const src=r.bounded_model_input;
  assert(src&&typeof src==='object'&&!Array.isArray(src),'missing bounded input');
  const keys=Object.keys(src).sort();assert(JSON.stringify(keys)===JSON.stringify([...MODEL_KEYS].sort()),'unexpected model input keys');
  assert(src.v==='synthetic-v30'&&Array.isArray(src.durable)&&src.durable.length>0,'invalid durable memory');
  assert(src.legal.includes(r.target),'non-legal teacher action');
  assert(sha(src)===r.bounded_model_input_sha256,'v34 bounded hash mismatch');
  return src;
}
function stripped(src){return {...src,durable:[]};}
function writer(file){const stream=fs.createWriteStream(file,{flags:'wx'});let rows=0,bytes=0;const h=crypto.createHash('sha256');
  return {write(obj){const b=JSON.stringify(obj)+'\n';stream.write(b);h.update(b);bytes+=Buffer.byteLength(b);rows++;},
    async finish(){stream.end();await new Promise((resolve,reject)=>{stream.on('finish',resolve);stream.on('error',reject);});return {rows,bytes,sha256:h.digest('hex')};}};
}
async function generate(parent,dest){
  assert(!fs.existsSync(dest),'refuse overwrite '+dest);
  const old=JSON.parse(fs.readFileSync(path.join(parent,'manifest.json'),'utf8'));
  assert(old.schema==='driv3r/corpus_factory/offpolicy-recovery-v34/manifest','wrong parent schema');
  assert(old.generator_sha256===sha(fs.readFileSync(path.join(__dirname,old.generator))),'parent generator source drift');
  for(const d of old.dependencies)assert(d.sha256===sha(fs.readFileSync(path.join(__dirname,d.file))),'parent dependency drift '+d.file);
  for(const [name,stat] of Object.entries(old.shards)){
    const file=path.join(parent,name);assert(fs.statSync(file).size===stat.bytes,'parent shard size '+name);
    assert(await fileSha(file)===stat.sha256,'parent shard checksum '+name);
  }
  fs.mkdirSync(dest,{recursive:true});
  const files={},splits={},geometry=new Map(),inputOwners=new Map(),memoryGroups={};
  for(const split of SPLITS){
    const episodes=new Map(),families=new Map(),worlds=new Set();
    for await(const ep of lines(path.join(parent,split+'_episodes.jsonl'))){
      assert(ep.split===split&&ep.synthetic_host_not_native,'bad parent episode');
      assert(!episodes.has(ep.episode_id),'duplicate episode ID');episodes.set(ep.episode_id,ep);
      if(ep.mode==='clean'){
        const arr=families.get(ep.family_id)||[];arr.push(ep);families.set(ep.family_id,arr);
      }
      const other=geometry.get(ep.geometry_sha256);assert(!other||other===split,'cross-split topology leak');geometry.set(ep.geometry_sha256,split);worlds.add(ep.geometry_sha256);
    }
    const out=writer(path.join(dest,split+'_policy.jsonl'));
    const challenges=split!=='TRAIN'?writer(path.join(dest,split+'_memory_challenge.jsonl')):null;
    const cache=new Map(),unique=new Set(),first=new Map();let count=0,nonzero=0,zero=0,interventions=0,dedup=0,conflicts=0;
    for await(const r of lines(path.join(parent,split+'_decisions.jsonl'))){
      const ep=episodes.get(r.episode_id);assert(ep,'orphan decision');const inp=input(r);
      assert(r.step>=0&&r.step<ep.steps,'bad step');assert(typeof r.target==='string','missing target');
      assert(r.training_weight===0||r.training_weight===1,'bad training weight');
      const id=sha(inp),prior=cache.get(id);
      if(prior!==undefined&&prior!==r.target){conflicts++;throw Error('full model-input label collision '+id);}
      if(prior!==undefined)dedup++;else cache.set(id,r.target);
      const owner=inputOwners.get(id);assert(!owner||owner===split,'cross-split bounded-input leakage');inputOwners.set(id,split);
      const safe={input:inp,action:r.target,weight:r.training_weight};
      assert(!Object.hasOwn(safe,'episode_id')&&!Object.hasOwn(safe,'oracle_action_costs'),'teacher field export');
      out.write(safe);count++;if(safe.weight)nonzero++;else zero++;
      if(r.intervention){assert(safe.weight===0,'intervention must have zero training weight');interventions++;}
      if(ep.mode==='clean'&&r.step===0){
        assert(!first.has(ep.episode_id),'duplicate first action');
        first.set(ep.episode_id,{input:inp,action:r.target,historyless:sha(stripped(inp))});
      }
    }
    const summary={episodes:episodes.size,geometry_groups:worlds.size,decisions:count,nonzero_training_decisions:nonzero,zero_weight_decisions:zero,
      repeated_bounded_inputs:dedup,conflicting_bounded_inputs:conflicts,accepted_wrong_paint_rows_excluded_from_training:interventions,
      qualified_clean_four_way_families:families.size};
    const challengeGroups=[];
    for(const [family,eps] of families){
      assert(eps.length===4,'incomplete matched family '+family);
      const items=eps.map(ep=>first.get(ep.episode_id));assert(items.every(Boolean),'missing clean first step');
      assert(new Set(items.map(x=>x.action)).size===4,'not four distinct first actions');
      assert(new Set(items.map(x=>x.historyless)).size===1,'recent-only non-aliased family');
      assert(new Set(items.map(x=>sha(x.input))).size===4,'full-history indistinguishable family');
      const actions=new Set(items.map(x=>x.action));assert(['MOVE_W','MOVE_E','MOVE_S','BACKTRACK'].every(x=>actions.has(x)),'unexpected 4-way labels');
      challengeGroups.push({family,items:items.sort((a,b)=>sha(a.input).localeCompare(sha(b.input)))});
    }
    if(challenges){
      for(const group of challengeGroups){
        // Identifiers and historical intervention metadata remain outside the model input.
        // Exactly four opposite-label first actions share this same memoryless input.
        for(const v of group.items)challenges.write({input:v.input,historyless_input:stripped(v.input),action:v.action});
      }
      summary.memory_challenge_rows=challengeGroups.length*4;
      summary.historyless_theoretical_best_accuracy=0.25;
      summary.memory_challenge_distinct_contexts=challengeGroups.length;
    }
    files[split+'_policy.jsonl']=await out.finish();
    if(challenges)files[split+'_memory_challenge.jsonl']=await challenges.finish();
    splits[split]=summary;
  }
  const manifest={schema:'driv3r/corpus_factory/trainer-safe-v35/manifest',synthetic_host_not_native_CyberSand:true,
    generator:'trainer_safe_export_v35.js',generator_sha256:sha(fs.readFileSync(__filename)),
    parent_manifest_sha256:sha(fs.readFileSync(path.join(parent,'manifest.json'))),parent_generator:old.generator,
    parent_generator_sha256:old.generator_sha256,parent_shards:old.shards,
    source_note:'v34 off-policy, host-accepted recovery; v35 preserves scenarios/labels but removes teacher-only source metadata from trainer-facing rows',
    model_input_contract:'Only JSON row.input is presented to the model; row.action is the supervised label; row.weight is loss weight (zero means exclude). Never concatenate entire row as a prompt.',
    challenge_contract:'DEV and HELDOUT memory_challenge contain 4-way first-action contexts, full bounded input and a historyless ablation with durable receipts removed. Run only as offline labeled first-step tests; not autonomous closed-loop evaluation.',
    non_goals:['Not a new geometry grammar or independent target SLM training','Not native CyberSand ABI or native gameplay','Not evidence of SLM memory or autonomous policy competence'],
    splits,shards:files};
  fs.writeFileSync(path.join(dest,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  return manifest;
}
module.exports={generate,input,stripped,sha};
if(require.main===module){const [parent,dest]=process.argv.slice(2);assert(parent&&dest,'usage: node trainer_safe_export_v35.js V34_SHARD_DIR FRESH_V35_DIR');
  generate(path.resolve(parent),path.resolve(dest)).then(x=>console.log(JSON.stringify({splits:x.splits,shards:x.shards},null,2))).catch(e=>{console.error(e.stack);process.exitCode=1;});}
