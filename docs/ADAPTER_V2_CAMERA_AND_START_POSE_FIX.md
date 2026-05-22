# Adapter v2 Camera and Start Pose Fix

Date: 2026-05-22
Branch: `piper-lerobot-adapter-v2`

## 1. Global Camera Fix

### Issue

The old `USBCamera` auto-detection iterated through all `/dev/video*` devices in
natural sort order. Since RealSense sub-devices (/dev/video2, /dev/video4) appear
before the 5MP USB Camera (/dev/video6), the old code would pick the first
non-black RealSense stream — which is a depth/IR stream, not the real global RGB
camera.

Additionally, the old `USBCamera` did not check whether the captured frame was
black (mean < 5), so it could select a metadata-only stream.

### Fixes applied

1. **Black frame detection in `camera/rs_camera.py`** — After the warmup loop in
   `USBCamera.__init__`, a new check computes `cv2.cvtColor(frame, COLOR_BGR2GRAY).mean()`.
   If mean < 5.0, the candidate is skipped with message `"black frame"`.

2. **Auto-detection already prefers USB cameras** — `_auto_video_candidates()` in
   `rs_camera.py` already sorts non-RealSense devices first. With the black frame
   check, the auto path now correctly lands on `/dev/video6`.

3. **`scripts/debug_global_camera.py`** — New diagnostic script that tests every
   `/dev/video*` candidate, prints device name (from sysfs), warmup frames, mean/min/max
   pixel values, saves a sample jpg, and recommends the correct device for `--global-camera`.

4. **Black frame guard in `record_adapter_v2.py`** — `is_frame_black()` checks
   global camera frames before allowing recording to start. SPACE key prints a
   warning if the global frame is black.

5. **Camera device printed before recording** — `create_global_camera()` now
   prints the selected device path.

### Camera inventory (2026-05-22)

| Device | Type | Opened | Mean | Black? | Use |
|---|---|---|---|---|---|
| /dev/video0 | Intel RealSense Depth | No | — | — | — |
| /dev/video1 | Intel RealSense Depth | No | — | — | — |
| /dev/video2 | Intel RealSense Depth | Yes | 108.9 | No | **Avoid** (depth/IR) |
| /dev/video3 | Intel RealSense Depth | No | — | — | — |
| /dev/video4 | Intel RealSense Depth | Yes | 81.9 | No | **Avoid** (depth/IR) |
| /dev/video5 | Intel RealSense Depth | No | — | — | — |
| **/dev/video6** | **5MP USB Camera** | **Yes** | **128.7** | **No** | **USE THIS** |
| /dev/video7 | 5MP USB Camera | No | — | — | — |

### Verification

```
Auto selected device: /dev/video6
5 frames: mean=125.5, 125.5, 125.5, 125.5, 125.3
All frames non-black: True
```

### Recommended command

```bash
# Auto works now, or explicitly:
python3 scripts/record_adapter_v2_mirror.py \
  --can-port can0 \
  --global-camera /dev/video6 \
  --start-pose-file config/adapter_v2_start_pose.json \
  --start-guard-mode zone \
  --dry-run
```

## 2. Adapter v2 Start Pose

### Issue

Adapter v2 needs its own collection start pose, independent of the old 10-demo
baseline `STANDARD_START_QPOS` in `adapter_v2/schema.py`. The old baseline
start pose must not be overwritten.

### Fixes applied

1. **`scripts/adapter_v2_capture_start_pose.py`** — New read-only script that:
   - Connects to Piper can0
   - Reads current qpos (no motion)
   - Prints qpos and gripper value
   - Warns if gripper < 0.09 m
   - Requires user to type `SAVE` before writing
   - Saves to `config/adapter_v2_start_pose.json`

2. **`--start-pose-file` in `record_adapter_v2.py`** — New CLI option that loads
   expected start pose from a JSON file. If not provided, falls back to schema
   `STANDARD_START_QPOS`.

3. **Metadata updated** — Each episode metadata now includes `start_pose_file`
   and `start_guard_mode` fields.

### Current adapter v2 start pose (pending user confirmation)

```
qpos: [0.019816, 0.004588, -0.003698, -0.071329, 0.017514, 0.024509, 0.0994]
gripper: 0.099400 m (OK, >= 0.09 zone minimum)
```

### To save the start pose

```bash
conda activate piper_act
python3 scripts/adapter_v2_capture_start_pose.py --can-port can0
# Type SAVE when prompted
```

## 3. Protected baseline (not modified)

| Asset | Status |
|---|---|
| `outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/` | Untouched |
| `data/lerobot_dataset_piper_bottle_old_singlecam_10demo/` | Untouched |
| `inference/deploy.py` | Untouched |
| `frozen_success/` | Untouched |
| `tag act-10demo-success-before-adapter-v2` | Untouched |
| `adapter_v2/schema.py` STANDARD_START_QPOS | Unchanged (new start pose in separate config file) |

## 4. Files changed / created

| File | Action |
|---|---|
| `camera/rs_camera.py` | Modified — black frame check in USBCamera warmup |
| `scripts/debug_global_camera.py` | **New** — camera diagnostic tool |
| `scripts/adapter_v2_capture_start_pose.py` | **New** — start pose capture tool |
| `scripts/record_adapter_v2.py` | Modified — `--start-pose-file`, `is_frame_black()`, camera device print, metadata fields |
| `adapter_v2/schema.py` | Previously modified — `StartGuardMode`, zone constants |
| `adapter_v2/start_pose.py` | Previously modified — zone/strict dual mode |
| `scripts/adapter_v2_check_start_pose.py` | Previously modified — `--mode` flag |

## 5. Next dry-run command

After saving the start pose:

```bash
conda activate piper_act

python3 scripts/record_adapter_v2_mirror.py \
  --can-port can0 \
  --global-camera /dev/video6 \
  --start-pose-file config/adapter_v2_start_pose.json \
  --start-guard-mode zone \
  --dry-run
```

Expected:
- Selected global camera: /dev/video6 (not black)
- Start guard compares against adapter v2 start pose from config file
- User can press C to re-check
- SPACE only works after GUARD PASS (with camera black-frame check)
- No dataset written
- No robot motion sent

## 6. One-demo record command (after dry-run passes)

```bash
python3 scripts/record_adapter_v2_mirror.py \
  --can-port can0 \
  --global-camera /dev/video6 \
  --start-pose-file config/adapter_v2_start_pose.json \
  --start-guard-mode zone \
  --num-episodes 1 \
  --fps 10
```
