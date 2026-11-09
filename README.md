# RayBEV-Lite v2 — 800×640 input, ±48 m BEV at 0.25 m, six classes

**YOLO26n pretrained backbone option: [TASK_V5_YOLO26N.md](TASK_V5_YOLO26N.md).**
The TDA-oriented convolutional subset retains official YOLO26n layers 0–9,
omitting C2PSA/2D heads. Start fresh with `scripts/launch_v5_yolo26n.sh`;
do not resume R18/MobileNet weights into it. Current runs/defaults are unchanged.

**V4.3 results and next opt-in experiments: [TASK_V4_4.md](TASK_V4_4.md).**
Best v4.3 is epoch 6 (5.31% internal mAP). The approved label-review fork
corrects cargo trucks mislabeled Bus, keeps trams as Bus, and leaves TRAIN
unchanged. A paired early-backbone-freeze experiment is available through
`scripts/launch_v4_4_review_experiment.sh`; it is not an established accuracy gain
or a new inference architecture. Original datasets/checkpoints are preserved.

**Current recipe: [TASK_V4_3.md](TASK_V4_3.md)** — corrects stride-8 sampling
alignment, constrains the complete facing vector, and retains separate height
evidence. Fixed BEV normalization, objectness/foreground separation, TRAIN-only
size priors, ResNet-18, OpenCV/CLIDAR and the offline TRAIN teacher remain.
New output: `work_dirs/raybev_v4_3_r18_geometry_48m_clidar_2x3080ti`.
Initialize from v4.2 epoch-9 `best_map.pt` with `--init-from`, not `--resume`.
The new 136-channel BEV handoff costs more memory; `--height-fusion pooled`
retains 40 channels for an ablation. Historical checkpoints keep their original
lattice/layout through `ModelConfig.from_checkpoint`. Regenerate deployment
exports/LUTs after retraining. New accuracy and TDA4VH throughput still require
measurement; source changes do not establish either.
The [v4.2 foreground/size guide](TASK_V4_2.md) documents the previous run.
The [v4 heading/teacher guide](TASK_V4.md) remains relevant except where superseded.
The MobileNet sections below are historical; their checkpoints remain readable.

**Historical MobileNet heading/size/classification recipe: [TASK_V3.md](TASK_V3.md).**
It supersedes the loss, sampler and three-output head contracts below. Train
fresh into `work_dirs/raybev_v3_1_tasks_48m_clidar_2x3080ti`; do not resume the old
v2 or failed v3 task checkpoint. Recipe 3.1 restores global loss normalization,
reduces auxiliary class CE and uses milder frame-presence repeats after v3
collapsed to zero validation AP. The OpenCV calibration, CLIDAR input convention and BEV
geometry are unchanged. The new head adds four heading-bin logits.

**Earlier architecture and training/deployment guide: [V2.md](V2.md).** Existing v1 checkpoints cannot resume this network. Source changes do not establish trained accuracy or TDA4VH throughput. Older dataset/history sections below describe the original integration.

**Calibration correction:** the four-bag source labels are CLIDAR (now the
default), converted once to BASELINK inside the loader. The active camera
extrinsics now match the converted bags/ROSA; OpenCV intrinsics are unchanged.
Start a fresh ImageNet-initialized run, including when replacing the old v2
server run; do not resume its checkpoint. Old calibration is preserved under
`configs/legacy_v2/` for historical inference. See [V2.md](V2.md).

New-run difficulty weights are `(1,2,4,4,4,4)` for foreground classification,
box regression and quality supervision in class order below. Confidence now
uses `class_score * sqrt(quality)` with fused-score peak selection. Old
checkpoints retain their original scoring through `ModelConfig.from_checkpoint`.
Those earlier loss/scoring changes did not enlarge the learned graphs; the
current task recipe additionally has a small four-logit heading branch.

Architecture: Four native fisheye frames → shared pretrained MobileNetV2-1.0 → geometry-conditioned 32-channel pyramid fusion → OpenCV fisheye LUT sampling across four heights with learned confidence → compact 40-channel pooled BEV → residual multiscale BEV CNN → six-class heatmap, separate eight-channel box tasks, and localization-quality head. No dense depth volume, transformer decoder, LiDAR input, or velocity output.

## Input geometry and resolution

- Image network input is **800×640 pixels (width×height)**, tensor shape `[B,4,3,640,800]`.
- Original camera calibration resolution is **1920×1536 pixels**, which has the same **5:4 aspect ratio**. This is an aspect-preserving scale to 800×640 with **zero padding and zero cropping**.
- Source raw images are resized by `raybev.preprocessing.letterbox_images` (kept for compatibility), which preserves aspect ratio and pads **only if the source aspect ratio differs**. The camera LUT uses the same transform.
- Four-camera order: `ARC_F, ARC_R, ARC_SL, ARC_SR`.
- The calibrated BEV covers **96 m × 96 m total**, `x,y ∈ [-48,48)` metres, resolution **0.25 m/cell**, shape **384×384**.
- BEV heights in `BASELINK`: `z=(0.4,1.4,2.4,3.4)` metres. `BASELINK`: x-forward, y-left, z-up.
- Intrinsics and extrinsics are read from `configs/calibration_baselink.json`. Each `T_BASELINK_CAMERA` maps points from camera to BASELINK. These transforms must match the recording session.
- Lens model: the supplied four-coefficient **OpenCV fisheye/equidistant** polynomial, as confirmed for this rig. OCam metadata in legacy PKLs does not control RayBEV projection.

## Classes and detection head

There are **six independent heatmap channels**, with class IDs fixed as follows:

| ID | Class |
|---:|---|
| 0 | `car` |
| 1 | `pedestrian` |
| 2 | `cyclist` |
| 3 | `truck` |
| 4 | `van` |
| 5 | `bus` |

Classes live in `raybev.config.CLASSES` and `CLASS_TO_ID`.

Output tensor shapes:

- `heatmap_logits`: `[B,6,384,384]`
- `regression`: `[B,8,384,384]` containing `(offset_row, offset_col, z, log_l, log_w, log_h, sin_yaw, cos_yaw)`
- `quality_logits`: `[B,1,384,384]`; decoding combines heatmap and quality scores.
- `decode_detections` produces `[N,9]` detections `(x, y, z, l, w, h, yaw, class_id, confidence)` for each image batch.

**Training labels must distinguish car, truck, van and bus.** A source annotation taxonomy that collapses these classes cannot teach the model their differences without reannotation or trusted relabeling. Existing three-class checkpoints cannot be loaded strictly into the six-channel heatmap head. Older six-class checkpoints with a ten-channel regression output also cannot be strictly loaded into the new eight-channel head; retraining or deliberate head surgery is required. Class IDs remain unchanged. The classes are not additional regression channels.

## Quick start

Python 3.10+ with `torch`, `torchvision`, `numpy` and `pytest`.

```bash
cd raybev_lite
PYTHONPATH=. pytest -q tests/
PYTHONPATH=. python scripts/smoke.py
```

Full-resolution forward pass:

```python
import torch
from raybev import ModelConfig, RayBEVLite, CLASSES
cfg = ModelConfig()
model = RayBEVLite(cfg, 'configs/calibration_baselink.json').eval()
images = torch.randn(1, 4, 3, 640, 800)  # already normalized after resizing
with torch.no_grad():
    heatmap, boxes, quality = model(images)
print(CLASSES)
print(heatmap.shape, boxes.shape, quality.shape)
# (1,6,384,384), (1,8,384,384), (1,1,384,384)
```

## Training

```bash
python scripts/train.py --data /path/to/npz_frames --epochs 100 \
    --batch-size 2 --checkpoint raybev_lite_6class.pt
```

Each `.npz` contains:

- `images`: `[4,3,H,W]`, camera order `ARC_F, ARC_R, ARC_SL, ARC_SR`, uint8 0–255 or float32 0–1, raw unnormalized images. Recommended source resolution 1920×1536. They are resized to 800×640 by the trainer.
- `boxes`: preferred `[N,8]` float32 `(x,y,z,length,width,height,yaw,class_id)` in **CLIDAR by default** (`--box-frame BASELINK` for preconverted labels), `yaw` in radians. `z` represents the **box center**, not bottom. The NPZ trainer also accepts the legacy `[N,10]` format `(x,y,z,l,w,h,yaw,vx,vy,class_id)`, **discards velocity columns**, and retains the original class ID. The target encoder itself strictly requires `[N,8]` in BASELINK after conversion.
- Class IDs must follow the table above. Samples can contain different numbers of objects.

The NPZ script stores class mapping/configuration and initializes the backbone from pretrained weights by default. Use the combined trainers below for validation, balancing, augmentation, and DDP. Decoding includes class-aware rotated BEV NMS.

The default width-1.0 MobileNetV2 loads TorchVision ImageNet V2 weights in all trainers. Use `--backbone-weights PATH` for a local official checkpoint or `--no-pretrained` for an explicit ablation. Direct model construction defaults to no download for offline inference/tests.

## ONNX export and TDA4VH

```bash
pip install onnx
python scripts/export_onnx.py --checkpoint raybev_lite_6class.pt \
    --calibration configs/calibration_baselink.json --output exported/
```

Produces `image_encoder.onnx` for **one 640×800 camera plus its eight-channel geometry map** (reuse four times), `bev_detector.onnx` for `[1,40,384,384]` BEV features, `geometry_lut.npz`, and detailed `metadata.json`. Geometry LUT sampling remains a separate custom processing stage. See [V2.md](V2.md) for exact contracts and TDA4VH integration.

For TDA4VH, measure TIDL-supported operators in the two learned subgraphs and port the fixed bilinear geometric sampler / weighted fusion to a tested C7x or optimized CPU kernel. No claim of successful TIDL deployment or real-time throughput is made.

## Historical changes before v2

- Before v2, BEV bounds changed from ±24 m to ±30 m at 0.30 m/cell. Current v2 uses ±48 m at 0.25 m/cell.
- Regression head: **10 → 8** channels; `vx` and `vy` deleted.
- Ground truth: **8** values per box; optional legacy 10-column NPZ conversion handled in `scripts/train.py`.
- Inference output: **9** values per detection; all velocity-related targets, losses and decoding removed.
- The larger BEV tensor increases the BEV encoder computational workload and memory usage; no TDA4VH throughput is claimed.

## Evaluation notes

Compare to ROSA using the **same split and label taxonomy**. If its dataset does not provide separate `truck`, `van`, and `bus`, you must produce those annotations to measure meaningful six-class AP. Check accuracy, orientation, range, per-class PR curves, memory and four-camera latency on the actual device.

## References

- https://github.com/kdh044/ROSA
- https://github.com/weiyangdaren/Fisheye3DOD
- https://github.com/TexasInstruments/edgeai-tidl-tools

---

# ROSA combined dataset integration (new in this package)

The scripts below **do not modify** your existing combined dataset. They use the same
ROSA/Fisheye3DOD `dict(metainfo=..., data_list=...)` PKL schema and keep all existing
sample fields (including `annos`, `cam_info`, `token`, timestamps, and `temporal_split`).
Per-camera `cam_path` values are added or rewritten in the *new* dataset views. The synchronized native `manifest.csv` is authoritative when supplied; existing camera records do not need a `cam_path` key.

**Important:** The supplied `summary.json`, `VALIDATION.json`, `manifest.csv`, and
`metadata.json` are metadata; the actual 4,488-frame image files and PKL annotations
are *not* included in the ZIP. This package has been tested using a local synthetic
ROSA-layout fixture, not trained or benchmarked on the full real dataset.

## Existing dataset identified in uploaded summary

- Existing combined root:
  `/home/mostafa/mmdetection3d/data/Custom/combined_improved_detzero_4bags_20261105`
- Original frame count **4,488** (train **3,787**, validation **674**, purge **27**)
- Boxes **50,404**, across **four bags**: `17_09`, `16_09`, `30_09`, `25_09`.
- Split membership is preserved exactly, not recalculated.
- Classes: Car, Pedestrian, Cyclist, Truck, Van, Bus.
- The existing validation has **zero Van labels**; it has 7,393 Cars, 313
  Pedestrians, 13 Cyclists, 137 Trucks and 215 Buses. Thus validation mAP
  includes five classes. `Van` AP is null, not zero. This particular five-class
  average is **not directly comparable** to a six-class benchmark mAP.

## 1. Make a combined dataset with both native JPGs and 800×640 JPGs

```bash
# From the root of the extracted raybev_lite folder
python scripts/prepare_combined_rosa.py \
  --base /home/mostafa/mmdetection3d/data/Custom/combined_improved_detzero_4bags_20261105 \
  --output /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640 \
  --native-dataset 17_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_17_08_33_47_converted/dataset \
  --native-dataset 16_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/dataset_temporal \
  --native-dataset 30_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_30_10_10_54_converted/dataset_temporal \
  --native-dataset 25_09 /home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_25_13_28_18_converted/dataset
```

`--native-dataset NAME DATASET` expects the synchronized dataset's `manifest.csv`
and native `images/arc_*/...jpg` files, **not** the recorder's raw H264 files.
The script matches by actual LiDAR acquisition timestamp (from `metainfo`) to
`manifest.csv`, falling back to recorder timestamp and verified `source_frame_id`.
It verifies source image dimensions (default 1920×1536), and refuses to recover
"native" images from already-square 800×800 images. If the native camera really
has another resolution, use `--native-width W --native-height H`; no image is
stretched, and source aspect ratio is preserved by optional letterboxing.
All output is written in a temporary sibling directory, then published to a
**new root**. Existing destinations are refused.

Resulting layout:

```text
raybev_4bags_native_800x640/
├── native_images/
│   ├── 17_09/ARC_F/<token>.jpg
│   ├── 17_09/ARC_R/<token>.jpg
│   ├── 17_09/ARC_SL/<token>.jpg
│   ├── 17_09/ARC_SR/<token>.jpg
│   ├── 16_09/...   30_09/...   25_09/...
├── images_800x640/
│   ├── 17_09/ARC_F/<token>.jpg
│   └── ...
├── annotations/
│   ├── fisheye3dod_infos_all.pkl
│   ├── fisheye3dod_infos_train.pkl
│   ├── fisheye3dod_infos_val.pkl
│   └── by_bag/
│       ├── 17_09.pkl    # all native images/annotations for this bag
│       ├── 16_09.pkl
│       ├── 30_09.pkl
│       ├── 25_09.pkl
│       └── <bag>_{all,train,val}.pkl
├── rosa/
│   ├── fisheye3dod_infos_all.pkl
│   ├── fisheye3dod_infos_train.pkl
│   └── fisheye3dod_infos_val.pkl
├── fisheye3dod/...
├── ImageSets-improved-detzero/...
├── native/rosa/...
├── native/fisheye3dod/...
├── calib_results.txt          # if available from source
├── source_summary.json
└── prepare_summary.json
```

The resized PKLs are in the **same annotation structure** as ROSA and point
at resized images; the native PKLs are suitable for original native images.
However, unmodified ROSA's image pipeline may have additional fixed assumptions
about image shape and pixel intrinsics. The resized PKLs' calibration fields
remain native-calibration fields; when using them in ROSA itself, adapt ROSA's
preprocessing and scale the image-coordinate projection *consistently*.
RayBEV uses the 1920×1536 calibration with its own 800×640 LUT scaling.

### Append another bag with your existing extension script

The script pasted in this chat imports project-local
`build_improved_detzero_rosa_dataset.py` and
`append_improved_detzero_rosa_dataset.py` plus `ocamcamera`, which were **not
included** in the provided files. To keep its original split, anchor fitting,
source-provenance and checksum logic, continue running that existing extension
script in its original Python environment. `scripts/extend_and_prepare.py`
chains your extension script with the above organizer:

```bash
python scripts/extend_and_prepare.py \
  --extender /path/to/your/extend_combined_dataset.py \
  --base /path/to/old_combined \
  --extended-output /path/to/new_extended_combined \
  --prepared-output /path/to/new_extended_native_800x640 \
  --native-dataset 17_09 /path/to/native_17_09_dataset \
  --native-dataset 16_09 /path/to/native_16_09_dataset \
  --native-dataset 30_09 /path/to/native_30_09_dataset \
  --native-dataset 25_09 /path/to/native_25_09_dataset \
  --add-sequence 08_11 /path/to/new_native_synchronized_dataset
```

The wrapper does **not** replace the original extension implementation; it
calls your copy of it directly with its local dependencies. Alternatively,
just execute the original extension command yourself and pass the resulting
new combined root to `prepare_combined_rosa.py`.

## 2. Verify box frames, rotations and camera projection **before** training

The `annos.gt_boxes` / `gt_boxes_3d` arrays are expected to contain:
`(cx,cy,cz,length,width,height,yaw[,vx,vy,...])` with yaw in radians.
The loader intentionally **ignores velocity columns** and uses `gt_names`
for class labels instead of assuming a numeric ROSA label order.

The combined dataset originates from LiDAR/DetZero data in **CLIDAR**, now
the default label frame. `T_BASELINK_CLIDAR` is applied once to centers and
heading vectors before filtering and training; model outputs use BASELINK.
For already converted labels explicitly pass `--box-frame BASELINK`.
Camera and LiDAR transforms default to the same JSON file. The CLIDAR
transform remains **rounded/recovered** source metadata; verify for new rigs.
The `--z-reference` switch identifies center-Z vs bottom-Z labels.

The rig's calibration is OpenCV fisheye, confirmed for these native images.
RayBEV uses the JSON's OpenCV coefficients. Legacy OCam metadata is preserved
as passive PKL state and is not used for projection. Use native 800×640
aspect-preserved images and the matching camera extrinsics.

```bash
python scripts/visualize_calibration.py \
   --root /path/to/raybev_4bags_native_800x640 \
   --ann native/rosa/fisheye3dod_infos_val.pkl \
   --calibration configs/calibration_baselink.json \
   --output calibration_overlays --samples 5
```

## 3. Train RayBEV-Lite on the combined PKLs and evaluate every epoch

Dependencies: Python 3.10+, PyTorch, TorchVision, Pillow, NumPy; optional TensorBoard.

```bash
python scripts/train_combined_rosa.py \
  --root /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640 \
  --train-ann rosa/fisheye3dod_infos_train.pkl \
  --val-ann rosa/fisheye3dod_infos_val.pkl \
  --box-frame CLIDAR \
  --z-reference center \
  --calibration configs/calibration_baselink.json \
  --epochs 24 --batch-size 2 --workers 4 --amp \
  --out work_dirs/raybev_rosa_combined
```

For another recording, supply its matching `--calibration`; optionally set
`--lidar-calibration /path/to/verified_full_calibration.json` if stored separately.

Every epoch produces:

- `epochs.csv`: train/val total, heatmap and regression losses; learning rate;
  active-class mAP, per-class AP at 0.5/1/2/4m and mean AP; center-distance
  precision, recall, TP, FP, FN; mATE/mASE/mAOE; elapsed wall time.
- `history.jsonl`: the same metrics with full class-level nested diagnostics,
  active/excluded classes and metric definitions.
- `tensorboard/`: scalar metrics, if `tensorboard` is installed.
- `last.pt` and `best_map.pt`: model, optimizer, scheduler, epoch, class order and metrics.

**Evaluation caveats:** Center-distance AP with thresholds 0.5/1/2/4 m and
10%-precision/recall gates follows nuScenes-style *ideas*, but these are
**independent lightweight evaluator implementations**, **not verified identical
to ROSA's benchmark evaluator**. TP errors are simple matched-pair means,
not nuScenes' official cumulative-mean error algorithm. The validation has
only 13 Cyclist boxes and zero Van boxes, so class results will have high
uncertainty. No model weights or measured validation results are provided.

### NDS and velocity

Official nuScenes NDS weights mAP, mATE, mASE, mAOE, **mAVE and mAAE**.
The network deliberately has **no velocity and no attributes**, so
**official NDS cannot be reported as a fully measured metric**.
The script outputs `NDS: null` and two explicitly **nonstandard** auxiliary
scores: `geometry_score_3term_nonstandard` and
`NDS_missing_velocity_attribute_zero_score_nonstandard` (missing scores=0).
Do **not** compare these auxiliary scores to ROSA's FDS or official nuScenes NDS.
To reproduce the true ROSA benchmark scores, use its official evaluator on
exported detections with the same active-class set and validation samples.

## 4. Run tests

```bash
PYTHONPATH=. pytest -q tests
# includes end-to-end synthetic ROSA PKL + manifest -> native JPG + 800x640 JPG
# -> loader; and absent-class metric behavior.
```

## Known unverified items

The provided combined dataset's actual PKL content and native images have
not been mounted into this environment. The source schema was inferred from
the user's extension script, uploaded metadata and previously known
RayBEV code; verify on the actual 4-bag files. A numerical overlay/geometry
review, class taxonomy/heading convention review, real training, actual
ROSA evaluator equivalence test, ONNX export test and TDA4VH TIDL compilation
remain necessary.

Per-epoch JSON additionally includes **per-bag** metrics (`per_bag_metrics`),
and the CSV contains per-bag mAP, mATE, mASE and mAOE. Be careful comparing
per-bag mAP scores, as each bag may have different absent classes.
The supplied original extension source is also included at
`scripts/extend_existing_dataset.py`, but requires your three private helper
modules in its Python path to run; `scripts/extend_and_prepare.py` can chain it.

### Camera path compatibility fix (2026-11-08)

If an existing combined ROSA `cam_info.cam_fisheye` camera dictionary lacks
`cam_path`, the organizer now uses native image paths from the timestamp-matched
`--native-dataset` `manifest.csv`. The output always contains a canonical
`cam_path`; any existing `img_path`, `image_path`, or `data_path` alias is also
updated to the same image path. Original camera calibration/pose entries are
left unchanged. Without a supplied native manifest, a recognized image path
field must exist in the source annotation.

A failed build keeps a `.<output>.building-*` directory for inspection. That
folder is separate from the intended output root: the fixed command may be
rerun with the original `--output` unless that root already exists.


## Live progress during combined-dataset training

`train_combined_rosa.py` now reports **training and validation progress separately**
within each epoch. If running in a terminal it uses `tqdm` with batch counts,
ETA, loss, heatmap loss, 3D regression loss, learning rate (training), and
processed images per second. The output also shows when global and per-bag
metrics are being calculated, per-class AP, checkpoint saves, and CUDA peak
allocated memory when CUDA is available.

```bash
pip install tqdm
python scripts/train_combined_rosa.py \
  --root /path/to/raybev_4bags_native_800x640_v2 \
  --box-frame CLIDAR --epochs 24 --batch-size 2 --workers 4 --amp \
  --progress always
```

`--progress auto` (default) shows a live bar in a TTY; when stdout/stderr are
redirected to a file, it prints progress every `--log-every 50` batches instead
of emitting hard-to-read escape sequences. `--progress always` forces a bar
inside VS Code terminals or notebook consoles. `--progress off` uses periodic
text logs. For shorter log intervals, pass `--log-every 10`.

Complete metrics continue to be saved each epoch to `epochs.csv`,
`history.jsonl`, and TensorBoard when installed. The metric definitions are
unchanged; validation classes with zero ground-truth samples are omitted from
the mean. `Van` remains N/A when absent from validation, and official NDS is
not computed because the model intentionally has no velocity/attribute head.

---

# Distributed training on two RTX 3080 Ti GPUs

This release adds `scripts/train_combined_rosa_ddp.py` and
`scripts/launch_2x3080ti.sh`. It uses PyTorch's NCCL DistributedDataParallel
(one independent process per GPU), **not** `nn.DataParallel` and **not** multiple
GPU workers inside a single Python process.

The current v2 training configuration is:

- 4 fisheye cameras, resized to 800×640; `ARC_F, ARC_R, ARC_SL, ARC_SR`.
- Pretrained MobileNetV2-1.0 image encoder; OpenCV LUT; 4 height planes.
- BEV `x,y ∈ [-48,+48)` metres at 0.25 m, grid 384×384.
- Six heatmap classes, eight regression channels and one quality channel; no velocity.
- Direct ROSA-style `rosa/fisheye3dod_infos_{train,val}.pkl` annotation reading.

## Verify the GPUs

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

You should see **two** CUDA devices. Both processes must be able to access
`--root`, `--train-ann`, `--val-ann`, and the camera calibration JSON.

## Start training

Run from the **project root** (not inside `scripts/`):

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_combined_rosa_ddp.py \
  --root /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640_v2 \
  --box-frame CLIDAR \
  --epochs 24 \
  --batch-size 1 \
  --accum-steps 2 \
  --workers 4 \
  --amp \
  --progress always \
  --out work_dirs/raybev_rosa_combined_2x3080ti
```

Alternatively run `bash scripts/launch_2x3080ti.sh`. Override its `ROOT`,
`OUT`, `EPOCHS`, `WORKERS`, `BATCH_SIZE`, `ACCUM_STEPS`, `BOX_FRAME` environment
variables as needed. Additional Python arguments pass through the launcher.

**`--batch-size 1` means one frame (all 4 cameras) per GPU, per microstep.**
The global effective batch size is `2 GPUs × 1 sample × 2 accumulation = 4`.
The two GPUs each maintain their own full copy of the model and optimizer;
VRAM is **not pooled**. AMP is enabled to reduce memory pressure. RTX 3080 Ti
boards are commonly 12 GiB; check the exact GPU with `nvidia-smi`. If you
still get CUDA out-of-memory, retain `--batch-size 1`, reduce workers if host
RAM is constrained, and inspect the geometric sampler's memory profile.

`--workers 4` means **4 DataLoader workers per GPU**, 8 total for training
(and analogous workers for validation). Reduce to 2 if CPU/disk is overloaded.

## Progress and metrics

Only **rank 0 / GPU 0** prints interactive `tqdm` bars and writes output files,
so terminal logs are not duplicated. The per-batch `gpu0_img_s` is throughput
for GPU 0, *not combined GPU throughput*. Both GPU processes participate in
training. For non-interactive terminals, use `--progress off --log-every 25`.

After **each epoch** the script merges two disjoint validation shards and
computes **global** metrics across *every validation frame exactly once*.
A standard padded `DistributedSampler` must **not** be used for validation
because it would duplicate frames when the validation count is uneven.

Files under `--out`:

- `history.jsonl`: complete epoch metrics, per-bag results, active/excluded classes.
- `epochs.csv`: training and validation losses; global mAP; each class's AP,
  precision/recall, mATE, mASE, mAOE, TP/FP/FN; per-bag AP/error summaries.
- `tensorboard/`: scalar curves, if TensorBoard is installed.
- `last.pt`, `best_map.pt`: rank-0, **unwrapped** state dictionaries compatible
  with the model from the single-GPU trainer; optimizer, scheduler, AMP scaler.

If a class has **zero validation ground truth**, its AP is `None`/`N/A` and
it is excluded from the mean. Your uploaded four-bag split has no `van`
validation examples. Standard nuScenes NDS **cannot be computed** without
velocity and attribute predictions; the script retains `NDS=None` and logs
separate geometry-only diagnostics, not a fake official NDS.

Training uses `DistributedSampler(..., drop_last=True)` to avoid duplicating
training examples; with 3,787 frames, one training frame is omitted *per
epoch* and the omitted frame varies with the epoch's shuffle seed.

## Resume training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_combined_rosa_ddp.py \
  --root /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640_v2 \
  --box-frame CLIDAR \
  --epochs 24 --batch-size 1 --accum-steps 4 --workers 4 \
  --amp --progress always \
  --out work_dirs/raybev_v2_48m_clidar_2x3080ti \
  --resume work_dirs/raybev_v2_48m_clidar_2x3080ti/last.pt
```

Resume only an interrupted **corrected CLIDAR** run with the same total
`--epochs`, full recipe, world size, class list, label frame and both calibration
hashes. Do not resume the pre-correction server checkpoints. Switching from
single-GPU to two-GPU training on resume is not supported.

## Validation of the DDP implementation

```bash
pytest -q tests/
torchrun --standalone --nproc_per_node=2 tests/smoke_distributed_runtime.py
torchrun --standalone --nproc_per_node=2 tests/smoke_raybev_ddp_cpu.py
```

These are test-only CPU/gloo runs (small synthetic tensors); they do **not**
benchmark NCCL, CUDA, real dataset throughput or GPU memory consumption.
For real use, profile both RTX GPUs and ensure that your calibrated OpenCV fisheye
projection agrees with actual image overlays before interpreting mAP.
