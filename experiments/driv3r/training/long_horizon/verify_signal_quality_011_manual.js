"use strict";
// Opt-in CPU regression only: never auto-run this on push / PR.
const assert=require("node:assert/strict");
const {research}=require("./signal_quality_011.js");
const d=research("degraded").evidence;
const v=research("verified").evidence;
for(const p of ["never","always","oracle"]){
  assert.equal(d.summary[p].goals,v.summary[p].goals);
  assert.equal(d.summary[p].utility,v.summary[p].utility);
  assert.deepEqual(d.summary[p].actions,v.summary[p].actions);
}
for(const e of [d,v]){
  assert.equal(e.model.optimizer_updates,1200);
  assert.equal(e.model.training_episodes,640);
  assert.equal(e.development_episodes,1920);
  assert.equal(e.corrupted_receipts_rejected,true);
  assert.equal(e.cohorts["128/64"].always,60);
  assert.equal(e.summary.risk_train.actions.PAINT>=0,true);
}
assert.ok(d.summary.risk_train.goals<d.summary.always.goals);
assert.equal(v.summary.risk_train.goals,v.summary.always.goals);
assert.ok(v.summary.risk_train.planners<v.summary.always.planners);
assert.equal(v.summary.risk_train.utility>v.summary.always.utility,true);
console.log("SIGNAL-QUALITY-011: controls, censor, corrupted receipts, paired outcomes PASS");
