"""Predeclared recording-level label budgets on already-used E01 subjects."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from romsl.experiment import classification_data, classification_scores, extract_features, loader, seed_all, save_checkpoint, write_json
from romsl.pipeline import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', choices=['A','B','C_proxy','D_proxy'], default=['A','B'])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.12)
    device = torch.device('cuda:0')
    actions = ['A03','A06','A12','A13','A14','A17','A18','A26']
    subjects = np.random.default_rng(12345).permutation([f'S{i:02}' for i in range(1,7)]).tolist()
    protocol = {'seeds': [42,43,44], 'experiments':args.experiments, 'recordings_per_class': [1,2,3,6],
                'nested_subject_order': subjects, 'subset_seed': 12345,
                'probe_epochs': 50, 'patience': 5, 'role': 'supplementary post-hoc label-efficiency analysis',
                'unit': 'whole recording, identical selected subjects for A/B and all encoder seeds'}
    write_json(args.output / 'protocol.json', protocol)
    selected_models = []
    for seed, job in zip(protocol['seeds'], args.jobs):
        for exp in args.experiments:
            train = extract_features(job, exp, 'train', device)
            val = extract_features(job, exp, 'val', device)
            features, labels, ids = classification_data(train, actions)
            for budget in protocol['recordings_per_class']:
                indices = [i for i,key in enumerate(ids) if key.split('/')[1] in subjects[:budget]]
                assert len({ids[i] for i in indices}) == 8 * budget
                seed_all(seed)
                head = torch.nn.Linear(128,8).to(device)
                optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
                batches = loader(torch.utils.data.TensorDataset(features[indices],labels[indices]),64,True,seed)
                best, stale, history = -1., 0, []
                name = f'{exp}_seed{seed}_k{budget}'
                for epoch in range(1,51):
                    head.train()
                    for x,y in batches:
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(head(x.to(device)),y.to(device))
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite few-shot loss')
                        loss.backward()
                        optimizer.step()
                    head.eval()
                    score = classification_scores(head,val,actions,device)['macro_f1']
                    history.append({'epoch':epoch,'val_macro_f1':score})
                    if score > best:
                        best, stale = score, 0
                        save_checkpoint(args.output / (name+'.pt'),{'head':head.state_dict(),'epoch':epoch,
                            'seed':seed,'encoder_sha':digest(job/'runs'/exp/'best.pt')})
                    else:
                        stale += 1
                    if stale >= 5:
                        break
                write_json(args.output/(name+'_history.json'),history)
                selected_models.append({'name':name,'experiment':exp,'seed':seed,'budget':budget,
                    'job':str(job),'checkpoint_sha':digest(args.output/(name+'.pt')),
                    'training_recordings':sorted({ids[i] for i in indices})})
                print('FEWSHOT_FIT',name,best,flush=True)
    write_json(args.output/'selection_lock.json',selected_models)
    with (args.output/'test_access_reserved.json').open('x') as handle:
        json.dump({'role':protocol['role'],'selection_lock_sha':digest(args.output/'selection_lock.json')},handle)
    results = []
    for seed,job in zip(protocol['seeds'],args.jobs):
        for exp in args.experiments:
            test = extract_features(job,exp,'test',device)
            for model in selected_models:
                if model['seed'] != seed or model['experiment'] != exp:
                    continue
                path = args.output/(model['name']+'.pt')
                assert digest(path)==model['checkpoint_sha']
                state = torch.load(path,map_location='cpu',weights_only=True)
                assert state['encoder_sha']==digest(job/'runs'/exp/'best.pt')
                head = torch.nn.Linear(128,8).to(device)
                head.load_state_dict(state['head'])
                metrics = classification_scores(head.eval(),test,actions,device)
                results.append({**model,'metrics':metrics})
    write_json(args.output/'results.json',{'protocol':protocol,'results':results})
    print('FEWSHOT_COMPLETED',len(results),flush=True)


if __name__ == '__main__':
    main()
