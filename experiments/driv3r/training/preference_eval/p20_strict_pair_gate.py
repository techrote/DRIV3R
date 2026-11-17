"""P20 original-producer stop-prefix and label-confound gate; no model training.
Requires source-pinned P19 archive; rechecks original P18/P10 producers using P19.
Not native CyberSand gameplay, and assistant labels are never human preferences.
"""
import argparse,collections,csv,hashlib,json,tempfile,zipfile
from pathlib import Path
import p19_crossmodel_visual_pilot as p19

P19_SHA='6c3df53ac82692fb057556fa874c9f7e7727cf3bac0afed5a2bfe0d604dc484d'
NICHES=('MEANDERER','CURLER','BRANCHER','CHAOTIC_ARTIST')
def sha(b):return hashlib.sha256(b).hexdigest()

def audit(bundle,out):
    if sha(Path(bundle).read_bytes())!=P19_SHA:raise ValueError('P19 original bundle checksum mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle) as z,tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for key,name in [('DRIV3R_P18_ORIGINAL_PRODUCER_REVERIFIED_BLIND_BALLOTS_LOCAL.zip','p18.zip'),('driv3r_preference_P10_real_W16_STOP_clock_model_inference_qualified.zip','p10.zip')]:
            (tmp/name).write_bytes(z.read('original_producer_input_bundles/'+key))
        pairs,lfm=p19.sources(tmp/'p18.zip',tmp/'p10.zip')
        reveal=json.loads(z.read('output/P19_REVEAL_AFTER_RATING.json'))['pairs']
        votes=list(csv.DictReader(z.read('source_and_receipts/P19_MODEL_DERIVED_VISUAL_PILOT.csv').decode().splitlines()))
        source={p['id']:p for p in pairs};seen=set();prefers_long=0;eligible=0;by_pair=collections.defaultdict(list)
        if {r['id'] for r in reveal}!=set(source):raise ValueError('reveal pair mismatch')
        for r in reveal:
            for k,v in [('A','a'),('B','b')]:
                if r[k]['source_json_sha256']!=source[r['id']][v]['source_json_sha256']:raise ValueError('original side SHA mismatch')
        for v in votes:
            key=(v['pair_id'],v['niche']);p=source[v['pair_id']]
            if key in seen:raise ValueError('duplicate vote')
            seen.add(key)
            if v['niche'] not in NICHES or v['rating_source']!='GPT6_ASSISTANT_VISUAL_PILOT_NOT_HUMAN':raise ValueError('invalid rating provenance')
            if v['vote_A_B_TIE_SKIP'] not in ('A','B','TIE','SKIP'):raise ValueError('invalid response')
            if p['cohort']=='P10_W16' and v['vote_A_B_TIE_SKIP'] in ('A','B'):
                eligible+=1
                longer='A' if len(p['a']['positions'])>len(p['b']['positions']) else 'B'
                right=int(longer==v['vote_A_B_TIE_SKIP']);prefers_long+=right;by_pair[p['id']].append(right)
        if len(seen)!=len(pairs)*4:raise ValueError('missing ratings')
        w16=[p for p in pairs if p['cohort']=='P10_W16'];historical=[p for p in pairs if p['cohort']=='P18_HISTORICAL']
        with zipfile.ZipFile(tmp/'p10.zip') as z10:
            for p in w16:
                family,seed=p['block'].split('/')
                def get(mode):
                    b=z10.read(f'new-rollouts/{family}-{seed}-{mode}-intact.json');return json.loads(b),sha(b)
                late,late_sha=get('late');early,early_sha=get('open')
                if {late_sha,early_sha}!=set(p['original_source_hashes']):raise ValueError('original W16 SHA mismatch')
                k=len(early['events'])-1
                if not (k>=0 and k<len(late['events'])-1 and early['events'][-1]['applied']=='STOP' and late['events'][k]['applied']!='STOP'):raise ValueError('not an early STOP contrast')
                if late['trajectory'][:k+1]!=early['trajectory'][:k+1]:raise ValueError('movement differs before STOP')
                if [e['applied'] for e in late['events'][:k]]!=[e['applied'] for e in early['events'][:k]]:raise ValueError('action differs before STOP')
                for j in (late,early):
                    if len(j['events'])+1!=len(j['trajectory']):raise ValueError('bad trajectory length')
                    for i,e in enumerate(j['events']):
                        if e['applied'] not in e['observation']['legal'] or e['observation']['tick']!=i or e['position_after']!=j['trajectory'][i+1]:raise ValueError('invalid source W0 action')
    result={'schema':'driv3r.preference_eval.p20.strict_pair_gate/1','verified_p19_zip_sha256':P19_SHA,'original_pairs':len(pairs),'W16_early_STOP_only_differences':len(w16),'W16_shared_movement_until_STOP':len(w16),'W16_non_tie_assistant_labels':eligible,'W16_assistant_preferred_longer':prefers_long,'W16_fraction_longer':round(prefers_long/eligible,6),'W16_pairs_majority_longer':sum(sum(x)>len(x)/2 for x in by_pair.values()),'W16_pairs_majority_shorter':sum(sum(x)<len(x)/2 for x in by_pair.values()),'W16_pairs_all_tied':len(w16)-len(by_pair),'historical_nonidentical_pairs':len(historical),'historical_independent_world_blocks':len({p['block'] for p in historical}),'original_pretrained_LFM350_trained_vs_stock_identical_paths':lfm['same_path'],'original_LFM350_changed_logit_ticks':lfm['different_logit_ticks'],'human_owner_labels':0,'labels':'GPT6 assistant visual pilot, not fully source-cohort blind','new_root_SLM_forwards':0,'new_optimizer_updates':0,'native_game_accepted':False,'recommendation':'Retain 18 STOP-prefix contrasts only as termination/length controls; do not train aesthetic ranker on these as independent style preferences. Require blind human votes and source-world-disjoint real pretrained model trajectories.'}
    (out/'P20_STRICT_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for i,svg in enumerate(p19.pair_svg(historical),1):(out/f'P20_HISTORICAL_BLIND_{i:02}.svg').write_text(svg)
    with (out/'P20_UNRATED_OWNER_BALLOT.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['anonymous_pair_id','niche','owner_A_B_TIE_SKIP','confidence_1_to_5','notes'])
        for p in historical:
            for niche in NICHES:w.writerow([p['id'],niche,'','',''])
    return result
if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--p19',required=True);x.add_argument('--out',required=True);a=x.parse_args();print(json.dumps(audit(a.p19,a.out),indent=2))
