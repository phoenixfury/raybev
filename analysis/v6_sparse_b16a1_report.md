# V6 Q100/L4/C64, batch 16/GPU: checkpoint analysis

## Verdict

This checkpoint is a regression from V5.1, not yet an accurate lightweight ROSA.
It has both an insufficient optimizer-update budget and genuine weaknesses in
heading, heavy-class generalization, precise localization, and false positives.
More updates deserve a controlled test; simply resuming this completed run at
its terminal learning rate is not a solution. No training code, annotations,
calibration, or model weights were modified during this analysis.

## Checkpoint and training contract

- Run: `work_dirs/raybev_v6_yolo_rosa_q100_l4_b16a1`.
- All 24 epochs finished. `best_map.pt`: epoch 23; `best_heading.pt`: epoch 19;
  `last.pt`: epoch 24. Best logged mAP: **3.2946%**; final: **3.2882%**.
- Actual recipe: 16 scenes/GPU, two GPUs, accumulation 1, global batch 32,
  AdamW head LR 0.0002, backbone multiplier 0.1, warmup 500 optimizer steps.
- 2,849 successful optimizer updates, versus 2,856 scheduled: seven AMP skips.
  All checkpoint tensors are finite. Scaler ended at 1024, not persistent failure.
- The original global-batch-8/24-epoch recipe scheduled 11,376 updates. Increasing
  the global batch fourfold without increasing epochs cut updates about fourfold.
- Final head LR: 4.004e-6; backbone LR: 4.004e-7. The schedule is exhausted.
  Strict resume correctly rejects changing epochs or the optimizer recipe.
- Official YOLO26n initialization is recorded and verified. All image stages
  were trainable. Measured relative parameter L2 changes from official weights:
  stage8 0.92%, stage16 2.54%, stage32 2.64%. The backbone is not frozen.
- Local camera/LiDAR calibration hashes and full semantic TRAIN/VAL annotation
  signatures match the checkpoint. Labels remain reviewed-v4 CLIDAR, converted
  once to BASELINK. These checks confirm consistency, not a new visual GT audit.

## Best-epoch validation

The same reviewed VAL split contains 674 scenes: Car 7,345, Pedestrian 313,
Cyclist 13, Truck 188, Bus 152, Van 0. Van AP is unavailable, not zero.

| Metric | V5.1 best logged | V6 best logged |
| --- | ---: | ---: |
| mAP | 5.815% | 3.295% |
| Car AP | 27.018% | 15.697% |
| Pedestrian AP | 2.055% | 0.776% |
| Truck / Bus / Cyclist AP | 0 / 0 / 0 | 0 / 0 / 0 |

V6 logs mATE 1.278 m, mASE 0.605, mAOE 0.873 rad. These are the existing
lightweight diagnostics, not official ROSA FDS or nuScenes devkit metrics.
Macro error means include a 1.0 sentinel for an active class with no matches;
their apparent improvements are not sufficient evidence of better heading.

At score 0.1, epoch 23:

| Class | TP / GT | FP | Recall |
| --- | ---: | ---: | ---: |
| Car | 3,084 / 7,345 | 14,842 | 41.99% |
| Truck | 5 / 188 | 1,713 | 2.66% |
| Bus | 3 / 152 | 1,812 | 1.97% |

The Bus AOE of 10.86 degrees describes only three matches. It is not a successful
Bus heading result. Lowering the score to 0.01 produces only eight Truck matches
and the same three Bus matches; at score 0.3 neither heavy class has a match.

The last five epochs gain only 0.062 mAP percentage points; epochs 15 to 24 gain
0.183 points. Validation loss changes 11.287 to 11.181 over that interval. TRAIN
loss changes 7.571 to 6.962, but TRAIN/VAL totals have different objectives and
sampling, so their numerical gap alone is not an overfitting measurement.

## Fresh inference and equal-recall comparison

Read-only native-resolution GPU AMP inference reproduces mAP **3.2922%** on all
674 scenes. Small differences from server logs are consistent with numerical
and NMS differences. Local GPU is an RTX 3060 Laptop, not the training hardware.

At exactly 40% Car recall, using score-ranked predictions and 2 m matching:

| Metric | V5.1 | V6 |
| --- | ---: | ---: |
| Precision | 59.47% | 21.66% |
| Directed AOE | 32.00 degrees | 55.66 degrees |
| Axis AOE | 8.88 degrees | 22.62 degrees |
| Greater-than-90-degree flip rate | 15.21% | 25.26% |

At 30% recall, AOE is 25.78 versus 53.15 degrees and precision 80.0% versus
47.80%. Equal recall does not imply identical matched GT subsets. The V5.1
prediction cache has identical frame tokens and reviewed evaluation labels.
These compare checkpoints, not isolated architecture effects: initialization,
training history, and update budgets differ.

Car localization is coarse: AP at 0.5 m is zero, at 1 m 2.85%, at 2 m 17.26%,
at 4 m 42.68%. At score 0.1, Car recall falls from 57.34% within 20 m to 27.74%
at 20–35 m and 5.46% beyond 35 m. Corresponding heading errors are 53.44,
65.87, and 89.62 degrees.

## Scoring and image-evidence ablations

Identical checkpoint and VAL labels; all geometry remains unchanged:

| Decoder variant | mAP |
| --- | ---: |
| Normal | 3.2922% |
| Remove quality multiplier | 3.2244% |
| Disable NMS | 3.2925% |
| Lower retained score floor from 0.01 to 0.001 | 3.2922% |

Truck and Bus AP stay zero in all variants. The quality multiplier and NMS are
not the principal cause of heavy-class failure. Removing quality increases
false positives; it should not be presented as an accuracy fix.

Artificial input ablations: zero all image-pyramid features -> mAP 0%; replace
every scene's pyramid with the first VAL scene's pyramid -> mAP 0.0456%.
These out-of-distribution probes show the model is using image evidence; they
do not prove that its semantic features are adequate or that calibration is
physically perfect.

## Heavy-class proposal, size, and classification failures

To distinguish missing centers from classification, raw queries were assigned
one-to-one to GT using XY distance only, without classes or score filtering.
Assignments over 2 m were rejected. This is a **geometric oracle diagnostic**,
not detection recall and not a relabeling of predictions.

| Final-stage diagnostic | Truck | Bus |
| --- | ---: | ---: |
| VAL geometric candidates / GT | 55 / 188 | 61 / 152 |
| Correct class among these candidates | 8 / 55 (14.5%) | 2 / 61 (3.3%) |
| Candidates classified Car | 34 / 55 | 38 / 61 |
| Candidates classified Truck | 8 / 55 | 14 / 61 |
| Median predicted / GT length | 0.693 | 0.327 |
| Median predicted / GT width | 0.745 | 0.568 |
| Median predicted / GT height | 0.693 | 0.516 |
| Mean true-class probability | 0.122 | 0.038 |

The nearby Bus candidates average 3.92 m length versus 10.07 m for their GT;
they are mostly Car-like queries, not successfully identified buses. Across
stages, Bus median relative length shrinks from 0.423 to 0.327 on VAL. The
decoder is reinforcing the wrong interpretation, not just suppressing an
otherwise correct large Bus box during postprocessing.

A fixed, class-balanced **143-scene TRAIN diagnostic subset** was evaluated
without augmentation, teacher, DN, or GT model inputs. It contains 43 Truck and
49 Bus boxes. All have a final raw candidate within 2 m. Conditional class
accuracy is 62.8% Truck / 59.2% Bus, versus 14.5% / 3.3% on VAL. TRAIN median
relative lengths are 0.924 / 0.752 versus 0.693 / 0.327 on VAL. Even TRAIN subset
AP is only 3.59% Truck / 6.03% Bus: the model is not perfectly fitting TRAIN.
This selected subset is not the entire TRAIN split; its macro mAP is not
directly comparable to VAL, particularly because it includes Van GT.

The sampler is active, unlike V5.1's old weak frame-repeat recipe: this run
accumulated 30,869 Truck and 31,660 Bus object exposures. These are repeated
examples, not unique objects. Missing sampling alone does not explain failure.

## Confirmed heading-objective limitations

`sparse_model.py` normalizes the doubled-axis prediction before
`sparse_loss.py` applies SmoothL1. This creates an antipodal stationary point:
an exact 90-degree physical axis error is a 180-degree doubled-vector error.
For normalized prediction [0,1] and that target, loss is 1.5 but its gradient
with respect to the pre-normalization prediction is approximately zero.
At 89 degrees the first-coordinate gradient is only -0.0349; at 60 degrees
it is -0.8660. Other model gradients can escape this; it is not proof that
the whole model is frozen or the sole explanation for the observed AOE.

The existing toy fit offsets anchors by **180 degrees**. Their doubled axes
are therefore initially correct: that test checks front/back learning, not
recovery from the 90-degree axis antipode. A separate axis-recovery test is
needed. Regressing unnormalized doubled-axis components, with normalization
reserved for geometry/decoding, is a candidate fix to verify.

Facing decoding also uses hard argmax/hemisphere selection between stages.
Decoded yaw has no gradient to facing logits; explicit facing classification
still trains them. Later geometric losses cannot train earlier front/back
choices through this hard switch. This is a limitation, not demonstrated
evidence that all heavy-class failures originate there.

Quality is supervised on matched positives only, with target
`exp(-XY_error_metres - mean_abs_log_size_error)`. Background queries receive
classification negatives but no quality negatives. This deserves calibration
work, but the measured ablation shows that removing quality is not a solution.

## Next controlled experiments

1. Correct the heading objective and add both 90-degree axis and 180-degree
   facing learnability tests. Keep Q100/L4/C64 while testing this change.
2. Compare against an unchanged-architecture, update-matched control. With
   global batch 32, 96 epochs schedules 11,424 updates, approximately the
   original 11,376. This also quadruples scene exposure; it is not statistically
   identical training or a guaranteed optimum. Monitor VAL and preserve best
   checkpoints. Alternatively reduce global batch back to eight for 24 epochs.
3. For continuation from V6, implement an explicit audited warm-start/new-
   schedule path before using it. Ordinary strict resume cannot change this
   completed recipe or revive its learning rate. Do not silently edit a saved
   checkpoint to bypass the checks.
4. Target heavy-class semantic and geometric generalization: test lightweight
   top-down multi-scale semantic fusion and/or object-region/learned-offset
   sampling. Use the raw-candidate diagnostics to verify improvement before
   merely increasing class weights or query count. These are next hypotheses,
   not fixes already proven by this analysis.
5. Expose complete TRAIN/VAL component losses, per-class errors, match counts,
   precision/recall, and heading by range in console/CSV. Keep no-match metrics
   clearly distinguished from real measurements. Preserve the same reviewed
   labels and evaluator when comparing experiments.

One hundred queries is not a cardinality ceiling for the audited max 35 VAL
objects per frame, but that does not guarantee adequate spatial proposal
coverage. Do not claim query count or more training alone resolves the failure.
No TDA4VH speed or INT8 accuracy conclusion follows from these tests.
