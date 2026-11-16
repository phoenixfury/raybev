# Local desktop annotation reviewer

Run `review_heavy_annotations.py` on the laptop's desktop, using the existing
RayBEV Python environment. It needs NumPy, Pillow, OpenCV, CPU PyTorch and Tkinter;
no model weights, Ultralytics, GPU, browser, LiDAR files or network connection.
On Linux, Tkinter may need the distribution's `python3-tk` package (or Conda Tk).
Keep the script inside this RayBEV project: it imports `raybev/` and
`prepare_heavy_annotation_review.py`.

## Inputs and start

### Remaining issues after the v5 audit

From the project directory, run:

```bash
bash scripts/review_remaining_v5.sh
```

On a laptop with images elsewhere:

```bash
ROOT=/path/to/combined_bags_camera_ray_dedup_v2 bash scripts/review_remaining_v5.sh --font-size 24
```

This is equivalent to `python scripts/review_heavy_annotations.py --remaining-issues --root /path/to/data`.
Copy the v5 release and `analysis/v10_manual_review_v5/{followup_review.jsonl,input_contract.json}`
along with the project if reviewing on another machine. Input hashes must match the audited
v5 TRAIN/VAL/extents/calibration; the ledger uses final v5 indices **after** previous removals.

Only **126 flagged objects** appear, including 49 Cars, 5 Pedestrians, oversized Vans,
and frames belonging to two mixed-class track IDs. The reason is displayed separately
in the UI; vehicle-sized Pedestrians open first. These are review flags, not automatic
class corrections or certified cuboid errors. Some flags overlap.

Issue mode defaults to **single-frame** edits because mixed track IDs may reflect
identity switches. Enable “Label entire bag + track” only after checking identity;
that optional operation still includes unflagged frames of that track. Removal always
affects just one frame. Dimensions preserve center/yaw; use Uncertain or removal when
the association cannot be established rather than guessing a replacement cuboid.

Completed issues leave the initial **unknown** filter. Select Status **all** to revisit
them, or **removed** to inspect removals; Undo and autosave remain available.
Progress is scoped to these 126 flags, not the full annotation pool.

New autosave: `work_dirs/manual_v5_followup/decisions.json`.
Export defaults to **`datasets/combined_bags_camera_ray_manual_review_v6/`**, including
complete TRAIN/VAL annotations and synchronized YOLO extents, not just flagged objects.
The v5 files and earlier completed review decisions are never overwritten. This follow-up
state contains only new decisions against v5; keep the v5 release for earlier provenance.
`--summary-only` checks the queue without opening a window; `--state`, `--output`,
and input paths can be overridden. Existing output directories are not overwritten.

### Full heavy-object review

Have these available on the laptop:

- The dataset's `images/` tree, preserving relative paths. `--root` is its parent.
- The three current v4 files: TRAIN pickle, VAL pickle and
  `train_yolo_extents_v1.json`.
- `configs/calibration_baselink.json`, unchanged.

From the RayBEV project directory:

```bash
python scripts/review_heavy_annotations.py \
  --root /path/to/combined_bags_camera_ray_dedup_v2
```

The UI defaults to a larger **20-point** font. Set `--font-size 24` for larger
text (range 10–48). **Ctrl + / Ctrl -** changes the font while the window is open.
Controls are in a scrollable right-hand panel, with class buttons first. Use its
vertical scrollbar to reach filters, context, track-wide edits and dimensions.
Controls wrap and stay above their row backgrounds. Camera titles and
image-overlay labels are enlarged too.

Use the **− / +** buttons above the right sidebar to resize text without keyboard
shortcuts. The window title must show **[font-v4]** and the effective point/pixel
size; the terminal also prints the absolute script path. If either is absent,
close the window and launch the updated script, not an older copy. Fonts are
assigned directly to widgets and dedicated button/checkbutton styles. Sizes use
fixed 96-DPI pixel conversion (20 pt = 27 px) to avoid desktop DPI discrepancies.
The script also tests actual rendered font heights and rejects bitmap fonts such
as `fixed`, which can silently stay at 13 pixels for every requested size. It
prefers a verified scalable Liberation Sans or another available scalable family.

TRAIN/VAL/extents default to `datasets/combined_bags_camera_ray_heavy_review_v4/`.
Override them with `--train-ann`, `--val-ann`, `--extents` if stored elsewhere.
In this checkout, a concrete launch is:

```bash
python scripts/review_heavy_annotations.py \
  --root ../../data/Custom/combined_bags_camera_ray_dedup_v2
```

The current v4 input contains **4,494** existing heavy annotations: **4,132 TRAIN**,
**362 VAL**. The queue also includes other-class frames of those same bag/track
identities, so a track correction does not leave its Car-labeled frames behind.
Unrelated Car tracks remain excluded. No BEV/range filter hides any selected track.
Add `--classes car truck van bus` to include the whole Car pool too; use a separate
`--state` for a different seed-class selection.

By default the script starts with previously **unresolved / taxonomy-policy**
objects from `annotation_review.jsonl` beside the TRAIN pickle, together with
objects manually flagged Uncertain. Already manually confirmed objects leave this
priority group. Use `--review-ledger /path/to/annotation_review.jsonl` if the ledger
is stored separately; copy it to the laptop with the annotations. Without a ledger,
only saved Uncertain decisions can identify unknowns. `--start-with all` restores
normal track ordering/resume. The Status filter also offers **unknown**.

## Review controls

Four panels show the selected object in all cameras. Object crops enlarge to fill
their panel while preserving aspect ratio, rather than staying at native size.
Enlargement improves visibility but cannot recover missing image detail.
Green curved lines are its
projected 3D cuboid; the yellow cross is its projected center. These projections
do not establish which vehicle is visible in front of an occluded target.
Use **Uncertain** instead of labeling the foreground occluder.

- `1` Car, `2` Truck, `3` Van, `4` Bus, `5` Pedestrian, `6` Cyclist.
- `K` confirms the current class; `U` flags uncertain and keeps its current label.
- Left/right arrows navigate without marking reviewed. `Z` undoes the last action.
- Buttons provide the same operations. Shortcuts are disabled while typing notes.
- Default: a label button saves and advances. Disable “Next after saving” to stay.
- Filters select split, original class, progress status, and bag/track/token text.
- Relabeled cars stay in the original queue so corrections can be revisited.
- Object crops include adjustable context. Toggle off for full-frame views.
  Double-click a panel to open the annotated image at native resolution with scrollbars.
- **Label entire bag + track** and **Across TRAIN + VAL** are enabled by default.
  Setting a class (or confirming the current class) saves it to every frame of
  that bag/track, including frames whose original class differs. Filters do not
  limit that scope. A count confirmation is required; Undo reverses the whole edit.
  Disable either checkbox for single-frame or current-split edits. The same numeric
  ID in another bag is never included; missing/negative IDs remain frame-only.
  Decisions persist on restart and exports synchronize affected TRAIN YOLO records.
  With next-after-saving enabled, a track edit advances to the next pending object
  instead of revisiting all its now-confirmed frames. **Uncertain stays frame-only**
  so one occluded view does not mark the whole track uncertain.
- Optional L/W/H entries prescribe size in metres. They save with a class/uncertain
  action; centers and yaw stay fixed. They are not a calibrated cuboid fit. No
  center/yaw editing is offered. Bottom-referenced input
  datasets are class-only.
- **Remove from this frame** removes only this object in the currently displayed
  frame, even when both track-wide checkboxes are enabled. Confirm the prompt.
  Other frames, images and the track ID are untouched. Removal autosaves, survives
  restart, and is skipped by later track relabels; Undo restores the prior decision.
  Removed objects are hidden normally; use Status **removed** to inspect them.
  Export drops the selected row from every aligned object field and removes its
  linked TRAIN YOLO extents. It retains the frame even if no objects remain, and
  keeps other objects/extent associations aligned after indices shift. Originals
  are never overwritten; removal takes effect in the new exported annotations.
- Notes also save with a class/uncertain action; navigation alone does not save
  unsent note/dimension field edits.

Decisions autosave atomically to
`work_dirs/heavy_manual_review/decisions.json`, including the last position and up
to 100 undo actions. Run the same command to resume. Source hashes prevent mixing
progress with another annotation revision. Use a new `--state` for new inputs.

## Export and move to the server

Click **Export new annotations + YOLO** and choose a NEW directory name, e.g.
`datasets/combined_bags_camera_ray_manual_review_v1`. Existing directories are
never overwritten. Partial exports are not committed. Unreviewed and uncertain
objects keep their current labels; they are not deleted or automatically ignored.

The exported folder contains:

```
fisheye3dod_infos_train.pkl
fisheye3dod_infos_val.pkl
train_yolo_extents_v1.json
review_manifest.json
review_decisions.json
review_queue.jsonl
```

Export verifies ALL annotations for unintended field/geometry changes, synchronizes
linked YOLO GT classes and the TRAIN fingerprint, then validates all retained weak
extents. Measured 2D boxes are never relabeled as new detector evidence. If a size
edit invalidates an existing extent association, that record is dropped and counted
in the manifest rather than reusing stale proof. If no valid extent remains, export
fails instead of producing an unusable extent file.

Copy the **whole export folder** to the training server, and copy this script and
`prepare_heavy_annotation_review.py` if the server doesn't have them. Keep image
ROOT unchanged. Server-side verification requires no deleted original extent JSON:

```bash
python scripts/review_heavy_annotations.py \
  --root ../../data/combined_bags_camera_ray_dedup_v2 \
  --verify-export datasets/combined_bags_camera_ray_manual_review_v1
```

Point training's `TRAIN_ANN`, `VAL_ANN`, and `EXTENT_ANNOTATIONS` at the three exported
files and choose a fresh `OUT`. **Do not run the old v4 correction-manifest verifier
on a desktop export**: this format uses `review_manifest.json` and the verification
command above. Existing image data and original annotation inputs are never overwritten.

Headless inventory or export from saved decisions:

```bash
python scripts/review_heavy_annotations.py --root /dataset --summary-only
python scripts/review_heavy_annotations.py --root /dataset --export-only \
  --output datasets/combined_bags_camera_ray_manual_review_v2
```

These modes need no display. Export-only does not read images, so you can also move
the original three annotation inputs and the saved decision file to the server and
export there. Tkinter is imported only for the desktop window.
