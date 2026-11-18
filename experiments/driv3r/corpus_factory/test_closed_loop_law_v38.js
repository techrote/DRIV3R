'use strict';
// Independent full-shard audit of data contract, paired long-run law effects,
// four-way memory challenge and bit-for-bit determinism.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const mod=require('./closed_loop_law_v38.js');
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const assert=(v,m)=>{if(!v)throw Error(m);};
function audit(a,b){
 const m=JSON.parse(fs.readFileSync(path.join(a,'manifest.json'))),n=JSON.parse(fs.readFileSync(path.join(b,'manifest.json')));
 assert(m.schema==='driv3r/corpus_factory/host-live-law-v38/manifest','schema');
 assert(JSON.stringify(m)===JSON.stringify(n),'non-deterministic manifest');
 let rows=0,contexts=0,lawChanged=0,weightZero=0,paint=0,topologies=new Map(),inputs=new Map();
 for(const [file,stat] of Object.entries(m.shards)){
   const f=fs.readFileSync(path.join(a,file)),g=fs.readFileSync(path.join(b,file));
   assert(f.length===stat.bytes&&hash(f)===stat.sha256,'corrupt shard '+file);
   assert(f.equals(g),'non-deterministic shard '+file);
   const list=f.toString().trim().split('\n').map(JSON.parse);
   assert(list.length===stat.rows,'row count '+file);
   const split=file.split('_')[0];
   if(file.endsWith('_policy.jsonl'))for(const x of list){
     rows++;
     assert(JSON.stringify(Object.keys(x).sort())===JSON.stringify(['input','action','weight'].sort()),'teacher leakage');
     const i=x.input,lk=i.law;
     assert(i.v==='synthetic-v30'&&lk.kind==='public_live_tool_mapping','wrong public input');
     assert([0,1,2].includes(lk.change_count),'wrong law transition');
     assert(lk.paint_A_yields!==lk.paint_B_yields,'nonbijective law');
     assert(i.legal.includes(x.action),'illegal label');
     assert([0,1].includes(x.weight),'invalid weight');
     if(!x.weight)weightZero++;
     if(lk.change_count>0)lawChanged++;
     if(x.action.startsWith('PAINT_')){
       paint++;
       const target=i.targets.find(t=>t[0]===i.p);
       assert(target,'no local target for paint');
       const physical=x.action==='PAINT_A'?lk.paint_A_yields:lk.paint_B_yields;
       assert(target[1]===physical,'paint label yields wrong physical material');
       assert(i.local[0][4]==='EMPTY','teacher paints occupied cell');
     }
     const k=hash(JSON.stringify(i)),earlier=inputs.get(k);
     assert(!earlier||earlier.split===split,'cross split model input leakage');
     assert(!earlier||earlier.action===x.action,'contradictory bounded label');
     inputs.set(k,{split,action:x.action});
     const geo=hash(JSON.stringify([i.cells,i.edges,i.protected,i.gate.slice(0,2)]));
     const owner=topologies.get(geo);
     assert(!owner||owner===split,'cross split geometry leakage');topologies.set(geo,split);
   }
   if(file.endsWith('_memory.jsonl')){
     contexts+=list.length;
     for(let j=0;j<list.length;j+=4){const family=list.slice(j,j+4);
       assert(family.length===4,'incomplete memory challenge');
       assert(new Set(family.map(x=>x.action)).size===4,'memory first-action labels not distinctive');
       assert(new Set(family.map(x=>hash(JSON.stringify(x.historyless_input)))).size===1,'memoryless alias violated');
       assert(new Set(family.map(x=>hash(JSON.stringify(x.input)))).size===4,'full memory distinctions absent');
     }
   }
   if(file.endsWith('_challenge.jsonl'))for(const ep of list){
     assert(ep.terminal_success&&ep.law_changes===2&&ep.paints_while_inverted>0,'incomplete adaptive episode');
     assert(ep.g!==undefined&&ep.v!==undefined&&ep.arm!==undefined,'missing reproducible world pointer');
     if(ep.mode==='bad-nav-and-paint')assert(ep.accepted_bad_nav?.accepted&&ep.accepted_bad_paint?.strict_regret>=2,'unproved interventions');
   }
 }
 assert(rows===Object.values(m.splits).reduce((p,s)=>p+s.decision_rows,0),'row count mismatch');
 assert(weightZero===Object.values(m.splits).reduce((p,s)=>p+s.zero_weight,0),'zero weights mismatch');
 assert(contexts===4*(m.splits.DEV.four_way_memory_families+m.splits.HELDOUT.four_way_memory_families),'memory challenge rows mismatch');
 assert(rows>0&&lawChanged>0&&paint>0,'challenge empty');
 return {rows,contexts,paint,lawChanged,weightZero,topologies:topologies.size,sha256:hash(fs.readFileSync(path.join(a,'manifest.json')))};
}
if(require.main===module){const [a,b]=process.argv.slice(2);assert(a&&b,'usage: node test_closed_loop_law_v38.js DIR_A DIR_B');console.log(JSON.stringify(audit(a,b),null,2));}
module.exports={audit};
