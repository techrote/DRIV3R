'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {sha,stripped,generate}=require('./trainer_safe_export_v35.js');
const read=p=>fs.readFileSync(p,'utf8').trimEnd().split('\n').map(JSON.parse);
const shaFile=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function check(dir){
 const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json')));
 assert.equal(m.schema,'driv3r/corpus_factory/trainer-safe-v35/manifest');
 assert.equal(m.generator_sha256,shaFile(path.join(__dirname,m.generator)));
 const global=new Map(),total={rows:0,nonzero:0,zero:0,challenge:0,conflicts:0};
 for(const [split,s] of Object.entries(m.splits)){
  const rows=read(path.join(dir,split+'_policy.jsonl'));
  assert.equal(rows.length,s.decisions);assert.equal(rows.filter(x=>!x.weight).length,s.zero_weight_decisions);
  const seen=new Map();for(const r of rows){
   assert.deepEqual(Object.keys(r),['input','action','weight']);
   assert.deepEqual(Object.keys(r.input).sort(),['v','p','legal','goal','ok','left','cells','edges','targets','protected','gate','local','recent','recent_positions','durable'].sort());
   assert.ok(r.input.legal.includes(r.action));
   assert.equal(r.input.v,'synthetic-v30');
   const k=sha(r.input),pr=seen.get(k),other=global.get(k);
   assert.ok(pr===undefined||pr===r.action,'conflicting bounded labels');
   assert.ok(!other||other===split,'cross-split bounded leakage');
   seen.set(k,r.action);global.set(k,split);total.rows++;r.weight?total.nonzero++:total.zero++;
  }
  assert.equal(rows.length-seen.size,s.repeated_bounded_inputs);
  if(split!=='TRAIN'){
   const challenge=read(path.join(dir,split+'_memory_challenge.jsonl'));
   assert.equal(challenge.length,s.memory_challenge_rows);const recent=new Map();let full=new Set();
   for(const r of challenge){
    assert.deepEqual(Object.keys(r),['input','historyless_input','action']);
    assert.deepEqual(r.historyless_input,stripped(r.input));
    assert.equal(r.historyless_input.durable.length,0);
    assert.ok(r.input.durable.length>0);
    assert.ok(r.input.legal.includes(r.action));
    const k=sha(r.historyless_input),group=recent.get(k)||new Set();group.add(r.action);recent.set(k,group);
    full.add(sha(r.input));
   }
   assert.equal(recent.size,s.memory_challenge_distinct_contexts);
   for(const grp of recent.values())assert.deepEqual(grp,new Set(['MOVE_W','MOVE_E','MOVE_S','BACKTRACK']));
   assert.equal(full.size,challenge.length);total.challenge+=challenge.length;
  }
 }
 for(const [f,v] of Object.entries(m.shards))assert.deepEqual({rows:read(path.join(dir,f)).length,bytes:fs.statSync(path.join(dir,f)).size,sha256:shaFile(path.join(dir,f))},v);
 console.log('PASS v35 independent dataset qualification',JSON.stringify(total));return m;
}
async function main(){const [parent,a,b]=process.argv.slice(2);if(!a)throw Error('usage: node test_trainer_safe_export_v35.js V34_DIR V35_DIR [SECOND_FRESH_DIR]');
 let m=check(a);if(b){const n=await generate(parent,b);assert.deepEqual(n,m);check(b);for(const f of ['manifest.json',...Object.keys(m.shards)])assert.equal(shaFile(path.join(a,f)),shaFile(path.join(b,f)));console.log('PASS two independent full v35 generations byte-identical');}
}
if(require.main===module)main().catch(e=>{console.error(e.stack);process.exitCode=1;});
module.exports={check};
