# V5: pretrained YOLO26n convolutional RayBEV

This is a **new backbone experiment**, not a demonstrated accuracy improvement.
Existing ResNet/MobileNet runs, defaults and checkpoint loading are preserved.
The new launcher explicitly selects `--backbone yolo26n` and a new output.

## What is used

The official Ultralytics **YOLO26n detection** checkpoint supplies every weight
and running statistic in layers 0–9, including C3k2 and the residual SPPF.
Taps are layer 4 (/8, 128 channels), 6 (/16, 128), and 9 (/32, 256).
The existing RayBEV geometry-gated pyramid projects them to 32 channels.

The deliberate deployment profile is `yolo26n_conv_sppf`, **not the complete
Ultralytics detector**. C2PSA self-attention (layer 10), the PAN neck, and all
COCO 2D detection heads are omitted. None of their unused parameters are
registered or optimized. This avoids attention MatMul/Softmax in the learned
deployment graphs. YOLO's 2D NMS-free head, DFL changes and MuSGD recipe do not
automatically transfer to the RayBEV 3D task: its 3D heads/losses/optimizer/NMS
remain the current RayBEV implementation.

The rest remains: four native 800×640 cameras, **OpenCV fisheye** calibration,
CLIDAR source labels converted once to BASELINK, ±48 m at 0.25 m (384×384), four
separate height planes, frozen-statistics BEV normalization, objectness × class
probability × sqrt(quality), class-relative sizes, axis/facing heading, and the
offline TRAIN-only ROSA heading/track supervision. Trams remain Bus; the approved
heavy-review-v1 validation fork is used. TRAIN labels remain unchanged.

The attached diagram's YOLO11n / OCam / ±30 m values are **not** used. Also, neck
width stays 32 for a controlled backbone comparison, not the diagram's 48/64.

## Input and pretrained-weight contract

The public model and ONNX inputs still take **ImageNet-normalized RGB**:
`(RGB_uint8 / 255 - mean) / std`, with mean `(0.485,0.456,0.406)` and std
`(0.229,0.224,0.225)`. Inside the YOLO encoder, `x*std+mean` restores RGB [0,1].
Do not send raw RGB [0,1] directly to this model; do not BGR-swap, normalize
twice, run Ultralytics prediction preprocessing, or letterbox to a square.
The loader's native 800×640 aspect ratio and stride-8 pixel origin 0 are retained.

BN epsilon/momentum match Ultralytics (0.001/0.03), not PyTorch defaults.
Every retained parameter/buffer loads strictly; missing, wrong-scale, wrong-task
or fused checkpoints fail. A failed download never falls back to random weights.
The default cache is pinned by SHA256. Local trusted matching `.pt` files are
architecture-checked and their actual SHA is recorded in `backbone_initialization`.
Original Ultralytics `.pt` files are Python pickles: load trusted sources only.

Bundled pretrained file: `weights/yolo26n.pt` (5.3 MiB).

```
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
SHA256: 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

The reference dependency is pinned in `requirements-yolo26.txt` to
`ultralytics==8.4.182`. The adapter also supports **8.4.208**, tested separately
with Python 3.8 / Torch 2.4 / TorchVision 0.19. If your server already has
8.4.208, keep it: update the RayBEV source; no downgrade is needed. Other package
versions still fail explicitly until tested. Actual runtime versions are recorded
in checkpoint/export profiles, separately from original pretrained provenance.
8.4.182 is installed in the local `openmmlab` environment. Install/verify it
in a separate or cloned server environment before switching jobs; do not upgrade
Torch/NumPy/MMCV in an environment being used by the running training process.
Legacy backbones do not import Ultralytics.

If startup reports `requires ultralytics==8.4.182; found 8.4.208`, the server
has the original exact-version guard. Sync the updated `raybev/` and `scripts/`
source; leave 8.4.208 installed and rerun the fresh launch without `--resume`.
The Ultralytics Settings creation message is harmless. Verification includes
36 targeted tests per package release, two-rank CPU DDP under 8.4.208, and
bit-for-bit identical full-resolution outputs after strict cross-version reload
(`analysis/yolo26n_version_compatibility.json`). No installed environment was
changed to perform these checks; 8.4.208 was tested via an isolated wheel.

## Start a new training run

From the RayBEV project directory, with the training environment activated:

```bash
python -m pip install -r requirements-yolo26.txt
python scripts/prepare_yolo26n.py
bash scripts/launch_v5_yolo26n.sh
```

`prepare_yolo26n.py` verifies the bundled file without network access; it only
downloads when missing. Copy `weights/yolo26n.pt`, `datasets/heavy_review_v1`,
the teacher JSON, calibration and updated source together to an offline server.
The launcher itself requires the pretrained file for a fresh run and makes no
network request. It also verifies the annotation fork before starting torchrun.

ROOT is auto-detected under the repository's `data/` or `data/Custom/` directory.
Override for another server layout:

```bash
ROOT=/absolute/path/to/raybev_4bags_native_800x640_v2 \
  bash scripts/launch_v5_yolo26n.sh
```

Default: 2 GPUs, batch 4/GPU, accumulation 1, 24 epochs, LR 2e-4, pretrained
stage LR multiplier 0.5, 500 optimizer-step warmup, AMP, all image stages
trainable. Projection/BEV/head weights start fresh. Output:
`work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti`.
Training prints and checkpoints the pretrained file SHA and exact subset.

**Do not use `--init-from` or `--resume` with a ResNet/MobileNet checkpoint.**
Their image features and parameter layouts are incompatible. Once a YOLO run
exists, resume that SAME run/recipe with:

```bash
bash scripts/launch_v5_yolo26n.sh \
  --resume work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti/last.pt
```

After training, RayBEV inference/export/resume builds the architecture offline
and loads its own strict state dictionary; the original COCO weights are not
needed. The pinned Ultralytics Python module is still needed to build YOLO layers.

## Compute and deployment

Measured at the unchanged full resolution, per four-camera frame:

| Backbone experiment | Whole-model parameters | Convolution GMACs |
| --- | ---: | ---: |
| Current ResNet-18 | 11,372,305 | 77.809 |
| YOLO26n conv + SPPF | 1,299,249 | 10.839 |

YOLO image encoding is 7.341G and BEV detection 3.498G convolution MACs.
Counts exclude sampling, pooling, resize, sigmoid/elementwise operations,
memory transfers and NMS; they are **not board latency or FPS**.
The concatenated BEV handoff is still 136×384×384 = 19.125 MiB at INT8.
See `analysis/yolo26n_profile.json` and the prior
`analysis/v4_2_epoch9/v4_3_profile.json`.

Use the existing `scripts/export_onnx.py --checkpoint ...` after retraining.
Exports record the precise backbone subset, input conversion and initialization.
Explicit channel splits avoid dynamic Shape/Gather/Div slicing in the YOLO graph.
The full-size learned subgraphs contain no self-attention, MatMul, Softmax or
GridSample. Native-shape ONNX Runtime/PyTorch parity is checked in
`analysis/yolo26n_native_export.json`.

Still required on **TDA4VH**: matching-SDK TIDL compile/offload report, SiLU
(Sigmoid×Mul)/Split/SPPF-MaxPool/Resize support checks, INT8 calibration and
accuracy evaluation, custom host/C7x LUT sampler and postprocessing, and actual
latency/peak-memory measurements. ONNX export alone does not prove TIDL support.

## Validation and comparison

Focused tests cover exact official pretrained feature/BN parity, normalization,
four-camera shapes, every trainable parameter's gradient, frozen weights/BN,
optimizer groups, strict reload without COCO re-download, wrong checkpoints,
download failure, combined teacher/track training, and ONNX parity.
149 focused tests passed. The two-rank CPU DDP smoke also passed with both
unfrozen and early-frozen YOLO stages (`RAYBEV_BACKBONE=yolo26n`).
Three full-resolution GPU AMP updates on actual TRAIN Car/Truck/Bus frames,
with the verified ROSA teacher and track targets, also passed; see
`analysis/yolo26n_smoke.json`. The default 65536 loss scale backed off twice
(to 16384) before all selected heavy-vehicle updates were finite. The existing
trainer already skips overflowing updates and backs off the scale; those skips
are not accepted optimizer steps. This is a smoke check, not a trained model
or an accuracy/convergence measurement. No production run was launched.

Compare trained YOLO and R18 checkpoints using the **same heavy-review-v1 GT**,
especially Truck/Bus AP and recall, size errors, class confusion and heading AOE
at matched recall. A smaller pretrained backbone does not resolve remaining
TRAIN label noise or prove improvement over ROSA. Do not compare a reviewed-GT
YOLO score directly against an original-GT R18/ROSA score.

Official architecture/weights: [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26)
and [Ultralytics release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Ultralytics uses [AGPL-3.0 / Enterprise licensing](https://www.ultralytics.com/license);
review the applicable terms before proprietary deployment.
