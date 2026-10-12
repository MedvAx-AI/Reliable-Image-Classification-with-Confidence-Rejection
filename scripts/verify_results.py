"""Independent NumPy/count verification of saved final artifacts and provenance."""
import csv
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np

from src.corruptions import corruption_specs, image_sha256
from src.data import sha256_file, sha256_json, sha256_text_file
from src.policy_artifacts import current_provenance, load_checked_logits, load_frozen_policy, read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root: Path = ROOT) -> dict:
    saved = read_json(root / 'results/run_manifest.json')
    for path, digest in saved['artifacts_sha256'].items():
        require(sha256_file(root / 'results' / path) == digest, f'artifact hash mismatch: {path}')
    policy = load_frozen_policy(root)
    provenance, manifest = current_provenance(root)
    require(saved['policy_sha256'] == sha256_text_file(root/'artifacts/policy.json'), 'policy hash mismatch')
    require(all(saved[k]==v for k,v in provenance.items()), 'run provenance mismatch')
    rows = read_json(root/'results/final_metrics.json')['metrics']
    require(len(rows)==40, 'expected 40 aggregate rows')
    aggregates = {(r['corruption'],r['severity'],r['condition']):r for r in rows}
    require(len(aggregates)==40, 'duplicate aggregate condition')
    data_by_condition = {}
    for spec in corruption_specs():
        key = spec['corruption'],spec['severity']
        name = f"{key[0]}_{key[1]}"
        path = root/f'results/logits/final_test_{name}.npz'
        record = next(r for r in saved['inference_records'] if r['condition']==name)
        require(sha256_file(path)==record['logit_sha256'], 'recorded logit hash mismatch')
        require(sha256_text_file(path.with_suffix('.json'))==record['sidecar_sha256'], 'recorded sidecar hash mismatch')
        data = load_checked_logits(path,expected_split='final_test',manifest=manifest,provenance=provenance)
        meta = data['metadata']
        require((meta['corruption'],meta['severity'])==key, 'condition filename mismatch')
        require(len(meta['image_sha256'])==2000 and sha256_json(meta['image_sha256'])==meta['image_hashes_sha256'],
                'native image hash evidence mismatch')
        z = data['logits'].astype(np.float64)
        y = data['labels']
        prediction = z.argmax(1)
        correct = prediction == y
        computed = {}
        for condition in 'ABCD':
            temp = 1.0 if condition in 'AB' else policy['temperature']
            scaled = (z-z.max(1,keepdims=True))/temp
            logp = scaled-np.log(np.exp(scaled).sum(1,keepdims=True))
            confidence = np.exp(logp.max(1))
            tau = policy['raw']['threshold'] if condition=='B' else policy['calibrated']['threshold'] if condition=='D' else None
            accepted = np.ones(2000,dtype=bool) if tau is None else confidence>=tau
            count, correct_count = int(accepted.sum()),int((accepted&correct).sum())
            bins = np.linspace(0,1,16)
            ece=0.0
            for b in range(15):
                mask=(confidence>=bins[b]) & ((confidence<=bins[b+1]) if b==14 else (confidence<bins[b+1]))
                if mask.any():
                    ece+=float(mask.mean()*abs(correct[mask].mean()-confidence[mask].mean()))
            expected={'n':2000,'correct':int(correct.sum()),'full_set_accuracy':float(correct.mean()),
                      'accepted_count':count,'correct_accepted_count':correct_count,'coverage':count/2000,
                      'rejection_rate':1-count/2000,'accepted_accuracy':correct_count/count if count else None,
                      'selective_risk':1-correct_count/count if count else None,
                      'nll':float(-logp[np.arange(2000),y].mean()),'ece':ece}
            actual=aggregates[(*key,condition)]
            for field,value in expected.items():
                require(actual[field] is None if value is None else abs(actual[field]-value)<1e-12,
                        f'independent metric mismatch: {key}/{condition}/{field}')
            for class_row in actual['per_class']:
                subset=y==class_row['label']
                require(class_row['n']==200 and class_row['accepted_count']==int((accepted&subset).sum())
                        and class_row['correct_accepted_count']==int((correct&accepted&subset).sum()),'per-class count mismatch')
            computed[condition]={'prediction':prediction,'correct':correct,'accepted':accepted,'confidence':confidence}
        data_by_condition[key]={**data,'computed':computed,'id_index':{s:i for i,s in enumerate(data['source_ids'])}}
    seen=set()
    with gzip.open(root/'results/per_image.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=row['corruption'],row['severity']
            source_id=row['source_id']
            identity=(*key,row['condition'],source_id)
            require(identity not in seen,'duplicate per-image row')
            seen.add(identity)
            data=data_by_condition[key]
            index=data['id_index'][source_id]
            decision=data['computed'][row['condition']]
            prediction=int(decision['prediction'][index])
            accepted=bool(decision['accepted'][index])
            require(int(row['true_class'])==int(data['labels'][index]) and int(row['predicted_class'])==prediction,'per-image label mismatch')
            require((row['accepted']=='True')==accepted and (row['correct']=='True')==bool(decision['correct'][index]),'per-image decision mismatch')
            require(row['output']==str(prediction) if accepted else row['output']=='REJECT','user-facing output mismatch')
            require(abs(float(row['raw_confidence'])-data['computed']['A']['confidence'][index])<1e-12
                    and abs(float(row['calibrated_confidence'])-data['computed']['C']['confidence'][index])<1e-12,'per-image confidence mismatch')
    require(len(seen)==80000,'expected 80000 per-image decisions')
    sweeps=list(csv.DictReader((root/'results/threshold_sweep.csv').open(encoding='utf-8',newline='')))
    require(len(sweeps)==2020,'expected 2020 sweep rows')
    for row in sweeps:
        data=data_by_condition[(row['corruption'],row['severity'])]
        decision=data['computed']['A' if row['confidence']=='raw' else 'C']
        mask=decision['confidence']>=float(row['threshold'])
        require(int(row['accepted_count'])==int(mask.sum()) and int(row['correct_accepted_count'])==int((mask&decision['correct']).sum()),'sweep count mismatch')
    report={'aggregate_rows':40,'per_image_rows':len(seen),'sweep_rows':len(sweeps),
            'independently_verified_metrics':True,'frozen_policy_sha256':saved['policy_sha256']}
    print(report)
    return report


if __name__=='__main__':
    verify()
