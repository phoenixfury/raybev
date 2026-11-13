"""Versioned, hash-guarded corrections from the 2026-11-13 visual review.

Only four visually supported class corrections. No deletion or geometry repair.
Source annotations are never overwritten. Run from the project directory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raybev.rosa_dataset import load_annotations

RULES = {
    ('09_30_10_10_54', '10995'): ('Van', 'Car', {'train': 0, 'val': 100}),
    ('11_06_10_07_06', '6568'): ('Van', 'Car', {'train': 170, 'val': 36}),
    ('09_30_10_10_54', '10499'): ('Bus', 'Truck', {'train': 8, 'val': 0}),
    ('11_09_08_01_03', '7304'): ('Bus', 'Truck', {'train': 29, 'val': 0}),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, default=Path('work_dirs/bus_van_visual_audit_20261113/index.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    audit = json.loads(a.audit.read_text())
    documents = {}; changes = []; manifest = {'scope': 'Four class corrections only; geometry and uncertain labels unchanged', 'splits': {}}
    for split in ('train', 'val'):
        path = a.root / f'annotations/fisheye3dod_infos_{split}.pkl'
        if sha(path) != audit['annotation_sha256'][split]:
            raise ValueError('Source differs from visually audited annotations: ' + str(path))
        doc = load_annotations(path); counts = Counter(); before = Counter(); after = Counter()
        for row in doc['data_list']:
            ann = row['annos']; before.update(map(str, ann['gt_names']))
            names = list(map(str, ann['gt_names'])); meshes = list(map(str, ann['gt_mesh']))
            for i, uid in enumerate(ann['gt_ids']):
                key = (str(row['metainfo']['sequence_id']), str(uid))
                if key not in RULES:
                    continue
                old, new, expected = RULES[key]
                if names[i] != old or meshes[i] != 'detzero.' + old.lower():
                    raise ValueError(f'Unexpected source class/mesh for {key}')
                names[i] = new; meshes[i] = 'detzero.' + new.lower(); counts[key] += 1
                changes.append(dict(split=split, token=str(row['token']), index=i,
                                    bag=key[0], track=key[1], old=old, new=new))
            # Recreate string arrays so Truck is never truncated to a short dtype.
            ann['gt_names'] = np.asarray(names, dtype=np.result_type(ann['gt_names'].dtype, 'U10'))
            ann['gt_mesh'] = np.asarray(meshes, dtype=np.result_type(ann['gt_mesh'].dtype, 'U18'))
            if list(ann['gt_names']) != names or list(ann['gt_mesh']) != meshes:
                raise ValueError('String dtype would truncate corrected labels')
            after.update(names)
        for key, (_, _, expected) in RULES.items():
            if counts[key] != expected[split]:
                raise ValueError(f'Track span changed: {key} {split} {counts[key]}')
        documents[split] = doc
        manifest['splits'][split] = dict(source=str(path), source_sha256=sha(path),
            before=dict(before), after=dict(after), changed=sum(counts.values()))
    a.output.mkdir(parents=True)
    for split, doc in documents.items():
        path = a.output / f'fisheye3dod_infos_{split}.pkl'
        with path.open('wb') as f:
            pickle.dump(doc, f, protocol=4)
        # Reload and prove boxes, identity, image paths and all other fields remain intact.
        original = load_annotations(Path(manifest['splits'][split]['source']))
        saved = load_annotations(path)
        assert original['metainfo'] == saved['metainfo']
        for src, dst in zip(original['data_list'], saved['data_list']):
            for key in src:
                if key == 'annos':
                    for field in src[key]:
                        if field not in ('gt_names', 'gt_mesh'):
                            assert np.array_equal(src[key][field], dst[key][field]), field
                else:
                    assert pickle.dumps(src[key], protocol=4) == pickle.dumps(dst[key], protocol=4), key
        manifest['splits'][split]['output_sha256'] = sha(path)
    manifest['changes'] = changes
    (a.output / 'correction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()
