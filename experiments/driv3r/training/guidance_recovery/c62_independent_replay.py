"""Independent cold replay of all gallery conditions and aligned/shuffled CPU refit."""
import hashlib,json,numpy as np,torch
from pathlib import Path
import c62_balanced_remote_transfer as c62
from safetensors.torch import save_file
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
c62.c60.SEED=c62.SEED;c62.c60.HORIZON=c62.HORIZON
c62.c60.c59.c58.c55.TICKS=(9,)
base,opt,_=c62.c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(root/'frozen/predecessor_C42')
tr,te,receipts,old=c62.cohorts()
trst,_=c62.c60.c59.c58.c56.states_for_goal(base,tr,'remote')
evst,_=c62.c60.c59.c58.c56.states_for_goal(base,te,'remote')
summary=json.loads((root/'results62/C62-summary.json').read_text())
weights={k:np.array(v,np.float32) for k,v in summary['weights'].items()}
traces=json.loads((root/'results62/C62-gallery-traces.json').read_text())
byid={s[6]['graph_sha']:s for s in evst}
assert all(t['graph'] in byid for t in traces)
for t in traces:
    state=byid[t['graph']]
    out=c62.execute(base,opt,state,c62.ARMS[-1],t['policy'],weights.get(t['policy'],np.zeros(c62.c60.NFEAT,np.float32)),t['drop'],trace=True)
    assert (out['action_sha'],out['ink_sha'],out['path'])==(t['action_sha'],t['ink_sha'],t['path'])
print('cold source-matched gallery replay:',len(traces),'PASS')
episodes=[(s,a) for s in trst for a in c62.ARMS[1:]]
for mode,label in [('aligned','C62-ALIGNED'),('shuffled','C62-SHUFFLED')]:
    model,stats=c62.fair_train(episodes,base,opt,mode)
    assert stats['updates']==summary['training']['updates']
    target=weights[label]
    assert np.array_equal(model.weight.detach().numpy().reshape(-1),target)
    tmp=root/'results62'/f'{label}-independent-refit.safetensors'
    save_file(model.state_dict(),str(tmp))
    digest=hashlib.sha256(tmp.read_bytes()).hexdigest()
    assert digest==summary['checkpoint_sha256'][label]
    tmp.unlink()
    print(label,'exact CPU refit SHA-256:',digest)
