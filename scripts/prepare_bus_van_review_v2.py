"""Remove Bus labels in two user-confirmed bus-free bags; preserve review v1."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import pickle

import numpy as np
from prepare_bus_van_review_v1 import load_annotations, sha

TARGETS = {'11_09_09_40_48', '09_25_13_28_18'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    parent = json.loads((a.source / 'correction_manifest.json').read_text())
    report = dict(reason='User confirms no buses in either bag',
                  targets=sorted(TARGETS),
                  aliases={'BAG_2026_09_25_13_28_18_converted': '09_25_13_28_18'},
                  parent_review=parent, splits={}, removed=[])
    documents = {}
    for split in ('train', 'val'):
        path = a.source / f'fisheye3dod_infos_{split}.pkl'
        if sha(path) != parent['splits'][split]['output_sha256']:
            raise ValueError('Review v1 annotations changed: ' + str(path))
        original = load_annotations(path); doc = copy.deepcopy(original)
        counts = Counter(); seen = set()
        for src, row in zip(original['data_list'], doc['data_list']):
            bag = str(row['metainfo']['sequence_id'])
            if bag not in TARGETS:
                continue
            seen.add(bag); ann = row['annos']; n = len(ann['gt_names'])
            keep = np.asarray([str(name).lower() != 'bus' for name in ann['gt_names']], dtype=bool)
            for i in np.flatnonzero(~keep):
                report['removed'].append(dict(split=split, bag=bag, token=str(row['token']),
                    index=int(i), track=str(ann['gt_ids'][i]), box=ann['gt_boxes'][i].tolist()))
            counts[bag] += int((~keep).sum())
            for field, values in ann.items():
                if not isinstance(values, np.ndarray) or len(values) != n:
                    raise ValueError('Unexpected annotation field: ' + field)
                ann[field] = values[keep]
                assert np.array_equal(ann[field], src['annos'][field][keep])
        if seen != TARGETS:
            raise ValueError('Missing target bag in ' + split)
        # All rows, metadata and non-target annotations are retained exactly.
        for src, row in zip(original['data_list'], doc['data_list']):
            for key in src:
                if key != 'annos' or str(row['metainfo']['sequence_id']) not in TARGETS:
                    assert pickle.dumps(src[key], protocol=4) == pickle.dumps(row[key], protocol=4)
            if str(row['metainfo']['sequence_id']) in TARGETS:
                assert not any(str(x).lower() == 'bus' for x in row['annos']['gt_names'])
        documents[split] = doc
        report['splits'][split] = dict(source_sha256=sha(path), removed_by_bag=dict(counts),
            removed=sum(counts.values()), frames=len(doc['data_list']))
    a.output.mkdir(parents=True)
    for split, doc in documents.items():
        path = a.output / f'fisheye3dod_infos_{split}.pkl'
        with path.open('wb') as f:
            pickle.dump(doc, f, protocol=4)
        saved = load_annotations(path)
        assert pickle.dumps(saved, protocol=4) == pickle.dumps(doc, protocol=4)
        report['splits'][split]['output_sha256'] = sha(path)
    (a.output / 'correction_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['splits'], indent=2))


if __name__ == '__main__':
    main()
