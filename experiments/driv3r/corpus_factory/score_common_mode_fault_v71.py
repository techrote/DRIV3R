import json,collections,pathlib,sys,hashlib
root=pathlib.Path(sys.argv[1]);counts=collections.defaultdict(collections.Counter); summary=collections.Counter(); collisions=collections.defaultdict(set)
for split in ('TRAIN','DEV','HELDOUT'):
    sf='TRAIN_common_mode_fault_stream.jsonl' if split=='TRAIN' else f'{split}_EVALUATOR_ONLY_common_mode_fault_stream.jsonl'
    cf=f'{split}_common_mode_fault_certificates_EVALUATOR_ONLY.jsonl'
    with (root/sf).open() as es,(root/cf).open() as cs:
        for line,cl in zip(es,cs):
            ep=json.loads(line);c=json.loads(cl); histories={m:{'PRIMARY':None,'SECONDARY':None} for m in ('A','B')}; receipts={m:{'PRIMARY':[],'SECONDARY':[]} for m in ('A','B')}
            for s in ep['steps'][:c['critical_step']]:
                for ev in s['input'].get('events_delta',[]):
                    kind=ev[1]
                    for m in ('A','B'):
                        if kind=='LOCAL_CHANNEL_CHECK_'+m: histories[m][ev[2]]=ev[3]
                        if kind=='REMOTE_MATERIAL_CHECK_'+m and len(ev)==5: receipts[m][ev[2]].append(ev)
            naive_targets=set();dual_targets=set();raw_signature=[]
            for m in ('A','B'):
                r=receipts[m]['PRIMARY']; r2=receipts[m]['SECONDARY']
                if not r:naive_targets.add(m);dual_targets.add(m);raw_signature.append('NONE');continue
                primary=max(r,key=lambda x:x[3]); secondary=max(r2,key=lambda x:x[3]);raw_signature.append(primary[4]+':'+secondary[4])
                primary_inv=histories[m]['PRIMARY']=='ERODED'; secondary_inv=histories[m]['SECONDARY']=='ERODED'
                decode=lambda rec,inv:('ERODED' if rec[4]=='INTACT' else 'INTACT') if inv else rec[4]
                p=decode(primary,primary_inv); s=decode(secondary,secondary_inv)
                if p!='INTACT':naive_targets.add(m)
                if p!='INTACT' or s!='INTACT':dual_targets.add(m)
            broken=set(c['actual_broken']);domain=c['domain'];group=counts[split+':'+domain]
            group['episodes']+=1;group['stale_cal_losses']+=not broken.issubset(naive_targets)
            group['dual_consensus_losses']+=not broken.issubset(dual_targets)
            group['stop_failures']+=bool(broken)
            group['teacher_saves_actions']+=c['always_actions']-c['teacher_actions']
            group['teacher_success']+=c['terminal_success']
            if domain=='COMMON_BUS' and c['flip'] and c['mode']['A']=='PAIRED' and c['mode']['B']=='PAIRED':
                summary['common_flip_both_verified']+=1
                summary['common_flip_dual_consensus_losses']+=not broken.issubset(dual_targets)
            signature=(split,c['geometry_sha256'],c['threshold'],tuple(c['sequence']),tuple(raw_signature),c['mode']['A'],c['mode']['B'])
            collisions[signature].add(c['teacher_first_action'])
summary['raw_signature_collisions_opposite_actions']=sum(len(x)>1 for x in collisions.values())
summary['unique_raw_signatures']=len(collisions)
result={'control_counts':{k:dict(v) for k,v in counts.items()},'summary':dict(summary)}
print(json.dumps(result,indent=2))