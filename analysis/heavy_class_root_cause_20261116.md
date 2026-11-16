# Heavy-class path audit: V9 epoch 23

The failure is a mixture of uncertain supervision, poor unseen-track generalization,
and geometry-conditioned query classification. It is not established to be a broken
YOLO backbone, camera/class-order swap, or primarily NMS suppression.

## Checks and scope

Inspected preprocessing, class mapping, backbone/FPN, fitted anchors, proposals,
attention/image sampling, all decoder stages, final-box classifier, denoising,
matching, focal/foreground/quality/geometry losses, weak extents and decoding.

Fresh CPU inference with the unchanged V9 epoch-23 weights covered 24 TRAIN frames
and 139 VAL frames. VAL includes **every 114 Truck annotation and every 23 Bus
annotation**, plus 31 Van annotations sampled across tracks. Van subset results
are not full-VAL metrics. TRAIN frame selection was track-stratified, not random
population evaluation. Baseline true-positive counts for Truck/Bus reproduce
the logged 12/114 and 2/23 at score >=0.1 and class-aware XY<2 m.

## Demonstrated findings

1. **Unresolved labels dominate the validation Truck pool.** The previous manual
   ledger retains 73 unresolved Truck annotations and 41 supported ones. Fresh
   inference detects 1/73 unresolved versus 11/41 supported. An unresolved label
   is not proven wrong; do not count every disagreement as a network mistake or
   delete these VAL objects to improve scores. TRAIN also retains 431 unresolved
   Truck, 375 Van and 113 Bus annotations, plus taxonomy-pending cases.

2. **The classifier works on sampled TRAIN objects but generalizes poorly.** In
   the 24-frame TRAIN sample it detects 25/25 Truck, 13/13 Van and 9/9 Bus objects.
   No Truck/Bus VAL track ID overlaps TRAIN. In contrast, 138/225 VAL Van boxes
   share TRAIN track IDs: bag `11_06_10_07_06`/110 and `11_09_09_00_30`/1400.
   These repeated identities make overall Van results look more general than they
   are. Exact IDs do not establish all physical-identity overlap or independence.

3. **Anchor fitting is only spatially balanced, not shape-balanced.** Recovering
   the exact TRAIN annotation origins of all immutable anchors gives 192 Car,
   12 Pedestrian, 2 Cyclist, 6 Truck, 7 Van and 6 Bus anchors. Of the 64 proposal
   size/yaw priors, **57 are Car-derived**, 1 Truck, 1 Van, 2 Bus and 3 Pedestrian.
   The fitting routine uses inverse-square-root class frequencies for spatial
   clustering but picks the final nearest real annotation without class weighting.
   These priors are not permanent class labels, but can bias initial extent and
   therefore which image evidence is sampled during refinement.

4. **The last-stage resampling patch does not rescue the failure by itself.**
   Among the nearest eligible VAL queries within 2 m of Truck GT, correct argmax
   counts are 18/108 baseline, 20/108 with the final fresh-box residual removed,
   19/108 when that residual samples the exact GT cuboid, and 16/108 with the
   explicit final head-state geometry contribution removed. These interventions
   keep the trained weights fixed. They do not remove geometry information from
   earlier attention/refinement. They do not demonstrate that the image backbone
   lacks heavy-object features, or justify deleting the residual in a production run.

5. **Correct query initialization can help some real heavy objects.** An additional
   small, supported-label probe inserted exact GT-seeded denoising-style queries,
   with ordinary query/attention rules and unchanged weights. For six supported
   VAL Truck observations, nearest-normal-query classes were correct for 4/6,
   versus 6/6 GT-seeded queries. For two supported Bus observations: 0/2 versus
   2/2. For four Van observations: 1/4 versus 1/4. GT seeding supplies privileged
   center/size/yaw information; **these are diagnostic interventions, not valid
   detector scores or proof that a particular new anchor scheme will fix it**.

6. **Later refinement does not consistently improve heavy recognition.** Counts
   of Truck annotations with any nearby supported Truck-winning query across
   stages are 35, 32, 28, 27. This is candidate availability, not one-to-one recall.
   At the final stage the logged audit gives 81 wrong-argmax, 15 below-score, 6
   no-geometry and 12 retained. Lower thresholds cannot repair wrong argmax.

7. **Frame balancing does not balance object-level supervision.** Epoch 23 sees
   94,866 Car, 5,446 Truck, 3,825 Van and 2,332 Bus targets. Selecting a heavy track
   still trains all surrounding cars. Foreground CE uses weights 1 for Car and 2
   for heavy classes, while all classes share the global object denominator.
   Focal negatives and co-occurring Car positives remain abundant. Target counts
   alone do not measure gradient contributions, so imbalance is a supported
   contributing mechanism, not a fully isolated causal result.

8. **The backbone is not simply missing/untrained.** Trusted COCO convolutional
   weights load strictly; the original 2D Detect/PAN heads are intentionally
   discarded. RayBEV's six-way heads train from scratch. Retained convolutional
   weight cosine similarities to initialization remain about 0.95–1.00; FPN
   residuals are nonzero. No nonfinite model tensors were found. These checks do
   not certify feature separability. Sparse samples also have no occlusion test:
   a camera-visible 3D point can sample a foreground car covering a heavy object.

9. **Heavy-specific geometry and direct image-class supervision are limited.**
   `heavy_endpoint_weight` is 0 in the actual full-V9 run. YOLO weak extents
   supervise measured boundaries, not a pretrained heavy semantic classifier.
   Final-box resampling is P3-only and detaches its sampling box coordinates.
   Quality is shared across classes and cannot repair their ordering. Bus facing
   is independently problematic: accepted tram detections are often reversed by
   about 180 degrees. Do not globally reverse otherwise-correct vehicle headings.

## Recommended order of repairs

1. Human review using all cameras and temporal context; retain an uncertainty
   queue. Include a reverse audit of heavy-looking original Cars. Verify cuboid
   association/geometry separately; size is not a class label. Keep a fixed,
   published VAL revision and never tune its labels to model predictions.
2. Controlled fresh experiment with class-independent, shape-balanced anchor and
   proposal size hypotheses. Preserve adequate Car coverage and measure both
   raw coverage and heavy classification at each stage.
3. Isolate object-level foreground class balancing/objectness-vs-semantic losses;
   do not just multiply every rare-class/background gradient or raise heavy logits.
   Log per-class train/VAL classification and supervision coverage.
4. Evaluate richer object-aligned image context and early refinement supervision,
   not only another final-layer residual. Run matching/geometry ablations one at
   a time; training assignment regret alone is already small.
5. Test endpoint geometry loss only against reviewed, credible cuboids. Audit
   tram directed headings separately. Compare false positives at fixed recall
   and report seen/unseen tracks, not just macro mAP.

No training weights, source annotations or network implementation were changed
by this audit. Temporary raw observations and GT-seeded probe results are in
`/tmp/raybev_heavy_path_audit.json` and `/tmp/raybev_heavy_oracle_queries.json`.
