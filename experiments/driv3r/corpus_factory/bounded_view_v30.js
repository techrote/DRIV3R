'use strict';
// Non-oracle bounded public-event view, using only a synthetic host observation.
function receipt(e, i){
  if(e.kind==='GATE_TRANSITION')return [i,e.open?'GATE_OPEN':'GATE_CLOSED',e.at];
  if(e.kind==='ACTION_ACK' && e.accepted && (e.action==='ERASE'||e.action.startsWith('PAINT_')))
    return [i,e.action,e.at];
  if(e.kind==='GOAL_REVISION')return [i,'GOAL_REVISION',e.goal];
  return null;
}
function eventBrief(e){
  return e.kind==='ACTION_ACK'?[e.action,e.at]:e.kind==='MOVE_ACK'?['MOVE',e.at]:
    e.kind==='GATE_TRANSITION'?[e.open?'GATE_OPEN':'GATE_CLOSED',e.at]:[e.kind];
}
function compress(obs,{memory=true,lastPaintOnly=false}={}){
  if(!obs.history?.events?.length || obs.history.events[0].kind!=='REMOTE_SURVEY')throw Error('missing public survey');
  const survey=obs.history.events[0],cells=survey.tiles.map(t=>t.cell),idx=new Map(cells.map((v,i)=>[v,i]));
  const locate=x=>{const n=idx.get(x);if(n===undefined)throw Error('survey index missing '+x);return n;};
  const edges=[];
  for(const t of survey.tiles)for(const n of t.neighbors){const a=locate(t.cell),b=locate(n);if(a<b)edges.push([a,b]);}
  const events=obs.history.events;
  let memories=events.map(receipt).filter(Boolean);
  if(lastPaintOnly){const last=[...memories].reverse().find(x=>x[1].startsWith('PAINT_'));memories=last?[last]:[];}
  if(!memory)memories=[];
  const result={
    v:'synthetic-v30',p:locate(obs.position),legal:obs.legal_actions,
    goal:[survey.goal.quota,locate(survey.goal.exit)],ok:obs.painted_correct,left:obs.remaining,
    cells,edges,targets:survey.paint_targets.map(t=>[locate(t.cell),t.material]),
    protected:survey.protected_cells,
    gate:[locate(survey.live_gate.gate_edge[0]),locate(survey.live_gate.gate_edge[1]),survey.live_gate.initial_open],
    local:obs.local_view.map(x=>[...x.offset,x.reachable?1:0,x.paintable?1:0,x.painted]),
    recent:events.slice(-6).map(eventBrief),recent_positions:obs.history.positions.slice(-6).map(locate),
    durable:memories.map(x=>[x[0],x[1],Array.isArray(x[2])?x[2].map(locate):typeof x[2]==='string'?locate(x[2]):x[2]])
  };
  return result;
}
module.exports={compress};
