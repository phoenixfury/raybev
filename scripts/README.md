# Build RayBEV / ROSA dataset from the four synchronized source datasets

This script **does not read the existing combined annotations or combined dataset**. It uses:

- `manifest.csv`: exact LiDAR sensor timestamps, original four JPG paths, camera timestamps.
- `detzero_output/improved_results.json`: object labels via your **existing, tested** DetZero-to-ROSA `load_sequence()` converter.
- `fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl`: per-bag template metadata used by that converter.
- `calib_results.txt`: OCam projection calibration (provided separately, can be copied from your old dataset; NO old combined PKL is read).

The user-supplied converter modules must be importable:
`build_improved_detzero_rosa_dataset.py`, `append_improved_detzero_rosa_dataset.py` and `ocamcamera.py`/the `ocamcamera` package. Put the new script next to the two converter scripts or specify `--builders-dir`.

## Run with your four bags

```bash
python scripts/build_combined_from_synchronized.py \
  --output /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640_v2 \
  --native-dataset 17_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_17_08_33_47_converted/dataset \
  --native-dataset 16_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/dataset_temporal \
  --native-dataset 30_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_30_10_10_54_converted/dataset_temporal \
  --native-dataset 25_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_25_13_28_18_converted/dataset \
  --ocam-calibration /home/mostafa/mmdetection3d/data/Custom/combined_improved_detzero_4bags_20261105/calib_results.txt \
  --builders-dir /path/containing/original/converter/scripts \
  --sequence-boundaries 17_09 1190 1199 \
  --sequence-boundaries 16_09 1019 1019 \
  --sequence-boundaries 30_09 1069 1078 \
  --sequence-boundaries 25_09 509 518
```

Those split boundaries are the **previous dataset's known frame-index boundaries**, using chronological ordering of the reconstructed usable samples. If any input bag has a different number/order of usable frames, check the input data and choose new boundaries; the script fails rather than silently reusing an old combined PKL. Omit all `--sequence-boundaries` options to compute a **new** chronological train/validation split per sequence (`--val-fraction 0.15 --purge-seconds 5`).

Use optional `--calibration-json /path/to/calibration_baselink.json` to attach full verified per-camera `T_BASELINK_CAMERA` and inverse matrices as provenance in camera records. It is *not* a substitute for OCam calibration. Exact model compatibility still requires a calibration overlay check.

**IMPORTANT**: Your dataset summary listed `dataset_temporal` as the source directory for `16_09` and `30_09`. If you have moved or renamed those sources, adjust the paths accordingly. The script requires a synchronized dataset containing the expected files.

## Output

```
raybev_4bags_native_800x640_v2/
├── native_images/<bag>/arc_{f,r,sl,sr}/<token>.jpg    # original native pixels
├── images_800x640/<bag>/arc_{f,r,sl,sr}/<token>.jpg    # real 800×640 images
├── annotations/
│   ├── fisheye3dod_infos_{all,train,val}.pkl         # combined RESIZED image paths
│   ├── by_bag/<bag>.pkl                              # one ROSA-style file per bag
│   ├── by_bag/<bag>_{train,val}.pkl
│   ├── by_bag_native/<bag>.pkl                       # original native image paths
│   └── native/fisheye3dod_infos_{all,train,val}.pkl
├── rosa/fisheye3dod_infos_{all,train,val}.pkl
├── fisheye3dod/fisheye3dod_infos_{all,train,val}.pkl
├── rosa_native/fisheye3dod_infos_{all,train,val}.pkl
├── rosa/anchors/kmeans300_seed3407.npy             # refitted ONLY from train
├── calib_results.txt
└── summary.json
```

The combined PKLs are compatible with `RosaCombinedDataset`, whose loader reads `sample['cam_info']['cam_fisheye'][CAM]['cam_path']` and `sample['annos']['gt_boxes'] / gt_names`. All six labels are preserved and velocity is ignored by RayBEV's detection head. The script does not train a network or evaluate NDS; use `scripts/train_combined_rosa.py` from RayBEV-Lite to train and log per-epoch mAP and 3D errors.

### Requirements

```bash
pip install numpy pillow
```

In addition, the existing DetZero->ROSA converter may need `mmengine`, `ocamcamera`, `scikit-learn`, and the other dependencies you already use for dataset conversion. This script deliberately delegates the interpretation of DetZero label JSON to your original converter to avoid guessing orientation, annotation source, box coordinate frame, and timestamp semantics.

### Safety and validation

- Never overwrites the output directory. Builds in a new staging directory and renames only on success.
- Does not modify the input synchronized datasets.
- Exact timestamp matching rather than matching frames by token order.
- Rebuilds train-only ROSA anchors and never uses validation/gap labels for anchors.
- Does not silently distort image aspect ratio. Native images must be 1920×1536 by default.
- Outputs `temporal_split` for every sample and separate train/validation annotation files.
- An already-created failed staging directory can be deleted after inspection.
- For reproducible comparisons, do not call a newly generated validation split the same benchmark as an older split unless its tokens have been checked.

