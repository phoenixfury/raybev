#!/usr/bin/env python3
"""Publish a NEW, self-contained ROSA-compatible combined four-bag dataset.

The original annotations, tokens, chronological train/val/gap membership and
ROSA/Fisheye3DOD data_list schema are preserved. For each bag it makes a single
native annotation PKL and collects the four original camera views into
native_images/<bag>/<camera>/<token>.jpg. A separate 800x640 image tree and
identically structured PKLs are published for RayBEV and ROSA.

IMPORTANT: If the existing combined images are 800x800, pass --native-dataset
for EACH bag to obtain actual original native camera JPGs. There is no way to
recover original pixels from pre-resized images. No source is overwritten.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from PIL import Image, ImageOps

CAMERAS=('ARC_F','ARC_R','ARC_SL','ARC_SR')
KEYS={'ARC_F':'arc_f_path','ARC_R':'arc_r_path','ARC_SL':'arc_sl_path','ARC_SR':'arc_sr_path'}


def load_pkl(path):
    # Only load pickles created locally by your own trusted dataset pipeline.
    with Path(path).open('rb') as f: return pickle.load(f)


def write_pkl(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(obj,f,protocol=4)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def safe_rel(path):
    p=Path(str(path))
    if p.is_absolute() or '..' in p.parts or str(p) in ('','.'):
        raise ValueError(f'Nonportable image path: {path}')
    return p


def find_image_root(ds):
    if (ds/'manifest.csv').is_file(): return ds
    if (ds/'dataset'/'manifest.csv').is_file(): return ds/'dataset'
    raise FileNotFoundError(f'No manifest.csv in {ds} or {ds}/dataset')


def load_manifest(ds):
    root=find_image_root(ds)
    with (root/'manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    by_sensor,by_recorder={},{ }
    for r in rows:
        for key,mapping in [('lidar_sensor_timestamp_ns',by_sensor),('timestamp_ns',by_recorder)]:
            if r.get(key):
                stamp=int(r[key]);
                if stamp in mapping: raise ValueError(f'Duplicate {key}={stamp} in {root}')
                mapping[stamp]=r
    return root,rows,by_sensor,by_recorder


def native_manifest_row(sample, tables, strict=True):
    root,rows,by_sensor,by_recorder=tables
    meta=sample.get('metainfo',{})
    # Exact sensor-clock match preferred. Recorder clock only when present.
    for key,mapping in [('acquisition_timestamp_ns',by_sensor),('lidar_sensor_timestamp_ns',by_sensor),('timestamp_ns',by_recorder)]:
        if key in meta and meta[key] is not None:
            row=mapping.get(int(meta[key]))
            if row is not None:return row
    # Last resort: original synchronized source frame ID, never a token guess.
    idx=meta.get('source_frame_id')
    if idx is not None and 0<=int(idx)<len(rows):
        row=rows[int(idx)]
        if strict:
            ts=meta.get('acquisition_timestamp_ns')
            if ts is not None and int(row.get('lidar_sensor_timestamp_ns') or -1)!=int(ts):
                raise ValueError(f'Frame ID/timestamp mismatch at {idx} in {root}: {ts}')
        return row
    raise KeyError(f'Cannot match native manifest row for sample {sample.get("token")} in {root}')


# Camera image paths in existing ROSA annotations are not universally named
# `cam_path`. When a synchronized native manifest is provided, its validated
# image paths are authoritative; the annotation camera entries only need to
# exist so their calibration/pose metadata can be preserved unchanged.
IMAGE_PATH_KEYS=('cam_path', 'img_path', 'image_path', 'data_path')


def validate_camera_rows(sample, *, need_image_paths=False):
    try:
        cams=sample['cam_info']['cam_fisheye']
    except (KeyError,TypeError) as e:
        raise KeyError(f'Sample {sample.get("token")} lacks cam_info.cam_fisheye') from e
    if not isinstance(cams,dict):
        raise TypeError(f'cam_info.cam_fisheye must be a dict for token {sample.get("token")}')
    for cam in CAMERAS:
        if cam not in cams:
            raise KeyError(f'Missing camera {cam} for token {sample.get("token")}; cannot preserve calibration metadata')
        if not isinstance(cams[cam],dict):
            raise TypeError(f'{cam} metadata must be a dict for token {sample.get("token")}')
        if need_image_paths and not any(cams[cam].get(k) for k in IMAGE_PATH_KEYS):
            raise KeyError(
                f'Missing image path for {cam} in token {sample.get("token")}; '
                'supply --native-dataset BAG SYNCHRONIZED_DATASET to resolve paths from manifest.csv'
            )
    return cams


def annotated_camera_path(camera_record, camera, token):
    for key in IMAGE_PATH_KEYS:
        if camera_record.get(key):
            return safe_rel(camera_record[key])
    raise KeyError(f'No image path for {camera} in token {token}')


def set_camera_image_path(camera_record, relative_path):
    """Write the canonical path and synchronize any preexisting path aliases.

    Some dataset converters use `img_path` or `image_path` instead of
    `cam_path`. Keeping the original alias stale would make different ROSA
    loaders silently read different images.
    """
    path=str(relative_path)
    for key in IMAGE_PATH_KEYS:
        if key == 'cam_path' or key in camera_record:
            camera_record[key] = path


def resize_to_800x640(src,dst,expect_size=(1920,1536)):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(src) as im:
        im=im.convert('RGB')
        if expect_size is not None and im.size != tuple(expect_size):
            raise ValueError(f'Expected native {expect_size} but {src} is {im.size}. Pass --native-width/height to match the true sensor images.')
        # The camera LUT assumes aspect-preserving scale and symmetric padding.
        # For original 1920x1536, this performs exactly 800x640, NO padding.
        scale=min(800/im.width,640/im.height)
        w=max(1,round(im.width*scale)); h=max(1,round(im.height*scale))
        res=im.resize((w,h),Image.Resampling.BILINEAR)
        if (w,h)!=(800,640):
            canvas=Image.new('RGB',(800,640))
            canvas.paste(res,((800-w)//2,(640-h)//2));res=canvas
        res.save(dst,'JPEG',quality=93,subsampling=0)


def link_image(src,dst,mode):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if mode=='copy':shutil.copy2(src,dst)
    elif mode=='hardlink':
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
    else:raise ValueError(mode)


def publish_views(root,split,obj,rosa_path):
    # write one pickle and link exact binary copies for both established consumers.
    canon=root/'annotations'/f'fisheye3dod_infos_{split}.pkl'
    write_pkl(canon,obj)
    for view in ('rosa','fisheye3dod','ImageSets-improved-detzero'):
        alias=root/view/canon.name;alias.parent.mkdir(parents=True,exist_ok=True)
        os.link(canon,alias)
    return sha256(canon)


def split_rows(rows):
    return {
        'all':rows,
        'train':[x for x in rows if x['metainfo'].get('temporal_split')=='train'],
        'val':[x for x in rows if x['metainfo'].get('temporal_split')=='val'],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True,help='existing combined_improved_detzero_4bags root')
    p.add_argument('--output',type=Path,required=True,help='new destination (must not already exist)')
    p.add_argument('--native-dataset',nargs=2,action='append',default=[],metavar=('BAG','SYNCHRONIZED_DATASET'),
                   help='repeat for every bag, e.g. 17_09 /path/to/BAG_converted/dataset')
    p.add_argument('--storage',choices=['hardlink','copy'],default='hardlink',help='native image storage; resized always copied')
    p.add_argument('--native-width',type=int,default=1920)
    p.add_argument('--native-height',type=int,default=1536)
    p.add_argument('--skip-native-size-check',action='store_true',help='allow differing sensor dimensions, still aspect-preserve')
    a=p.parse_args()
    base=a.base.resolve(); output=a.output.resolve()
    if output.exists():raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if not base.is_dir():raise FileNotFoundError(base)
    source=base/'annotations/fisheye3dod_infos_all.pkl'
    raw=load_pkl(source)
    if 'data_list' not in raw or 'metainfo' not in raw:raise ValueError('Expected ROSA info dict with data_list and metainfo')
    samples=raw['data_list']
    original={}
    for split in ('train','val'):
        original[split]={str(s['token']) for s in load_pkl(base/'annotations'/f'fisheye3dod_infos_{split}.pkl')['data_list']}
    token_set={str(s['token']) for s in samples}
    if len(token_set)!=len(samples):raise ValueError('Duplicate sample tokens')
    native_paths={name:Path(path).resolve() for name,path in a.native_dataset}
    if len(native_paths)!=len(a.native_dataset):raise ValueError('Duplicate native-dataset bag argument')
    bag_names={str(s['metainfo']['sequence_id']) for s in samples}
    unknown=set(native_paths)-bag_names
    if unknown:raise ValueError(f'--native-dataset unknown bags: {unknown}')
    tables={name:load_manifest(path) for name,path in native_paths.items()}
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.building-',dir=output.parent))
    try:
        native_rows=[]; resized_rows=[]; sizes=Counter(); by_bag=defaultdict(list)
        for n,s in enumerate(samples):
            bag=str(s['metainfo']['sequence_id']);token=str(s['token'])
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',bag):
                raise ValueError('Unsafe bag directory component')
            # Tokens are never interpreted as file paths; preserve token in the PKL.
            file_token=token if re.fullmatch(r'[A-Za-z0-9_-]{1,120}',token) else hashlib.sha256(token.encode()).hexdigest()[:32]
            row=native_manifest_row(s,tables[bag]) if bag in tables else None
            orig_cams=validate_camera_rows(s,need_image_paths=(row is None))
            sn=copy.deepcopy(s);sr=copy.deepcopy(s)
            cams_n=sn['cam_info']['cam_fisheye'];cams_r=sr['cam_info']['cam_fisheye']
            for cam in CAMERAS:
                if row is not None:
                    rel=safe_rel(row[KEYS[cam]]); img_src=tables[bag][0]/rel
                else:
                    rel=annotated_camera_path(orig_cams[cam],cam,token);img_src=base/rel
                if not img_src.is_file():raise FileNotFoundError(f'Native source missing: {img_src}')
                with Image.open(img_src) as testim: size=testim.size
                if size==(800,800):
                    raise ValueError(f'Source {img_src} is already 800x800! Supply --native-dataset {bag} with the original native JPGs')
                sizes[str(size)]+=1
                rel_native=Path('native_images')/bag/cam/f'{file_token}.jpg'
                rel_resize=Path('images_800x640')/bag/cam/f'{file_token}.jpg'
                link_image(img_src,staging/rel_native,a.storage)
                resize_to_800x640(img_src,staging/rel_resize,None if a.skip_native_size_check else (a.native_width,a.native_height))
                set_camera_image_path(cams_n[cam],rel_native.as_posix())
                set_camera_image_path(cams_r[cam],rel_resize.as_posix())
                # Preserve all ROSA calibration fields; do not overwrite intrinsics
                # without knowing their exact semantic representation.
            native_rows.append(sn);resized_rows.append(sr);by_bag[bag].append(sn)
            if (n+1)%500==0:print(f'Processed {n+1}/{len(samples)} frames',flush=True)
        def assert_split(rows):
            for split in ('train','val'):
                got={str(x['token']) for x in split_rows(rows)[split]}
                if got!=original[split]:
                    raise ValueError(f'Existing {split} membership changed! Missing {len(original[split]-got)}, extra {len(got-original[split])}')
        assert_split(native_rows);assert_split(resized_rows)
        hashes={}
        for split,rows in split_rows(resized_rows).items():
            hashes[split]=publish_views(staging,split,dict(metainfo=copy.deepcopy(raw['metainfo']),data_list=rows),True)
        native_root=staging/'native'
        for split,rows in split_rows(native_rows).items():
            publish_views(native_root,split,dict(metainfo=copy.deepcopy(raw['metainfo']),data_list=rows),True)
        # One independent, complete per-bag annotation file, preserving each original sample structure.
        per_bag={}
        for bag,rows in sorted(by_bag.items()):
            folder=staging/'annotations'/'by_bag'
            dst=folder/f'{bag}.pkl';write_pkl(dst,dict(metainfo=copy.deepcopy(raw['metainfo']),data_list=rows))
            per_bag[bag]={'frames':len(rows),'annotations':str(dst.relative_to(staging))}
            for split,subset in split_rows(rows).items():
                write_pkl(folder/f'{bag}_{split}.pkl',dict(metainfo=copy.deepcopy(raw['metainfo']),data_list=subset))
        # Original OCam calibration + ROSA anchor are only copied if they exist.
        for rel in ('calib_results.txt','rosa/anchors/kmeans300_seed3407.npy'):
            if (base/rel).is_file():
                dest=staging/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,dest)
        original_summary=base/'summary.json'
        if original_summary.is_file():shutil.copy2(original_summary,staging/'source_summary.json')
        summary={'source_root':str(base),'frames':len(samples),'bags':per_bag,
                 'input_size_wh':[800,640],'native_sizes':dict(sizes),
                 'image_count':len(samples)*len(CAMERAS),'original_train_tokens':len(original['train']),
                 'original_val_tokens':len(original['val']),'annotation_sha256':hashes,
                 'camera_order':list(CAMERAS),'annotation_schema':'ROSA/Fisheye3DOD data_list preserved',
                 'split_preserved':True,'native_frame_image_paths':'native_images/<bag>/<camera>/<token>.jpg',
                 'resized_frame_image_paths':'images_800x640/<bag>/<camera>/<token>.jpg',
                 'calibration_warning':'Intrinsic metadata is unchanged because native camera coordinates remain authoritative. Any ROSA loader that assumes annotations are raw image dimensions must apply 800x640 pixel scaling or use the original calibration-aware RayBEV LUT.'}
        (staging/'prepare_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if output.exists():raise FileExistsError(output)
        os.rename(staging,output)
        print(json.dumps(summary,indent=2))
    except Exception:
        print(f'Build failed; staging directory retained for inspection: {staging}',flush=True)
        raise

if __name__=='__main__':main()
