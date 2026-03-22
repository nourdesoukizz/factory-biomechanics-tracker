# Factory Biomechanics Tracker — Design Document

> **Purpose**: This document is the single source of truth for building the factory-biomechanics-tracker pipeline. Claude Code should reference this document at every phase before writing code.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Data Flow](#4-data-flow)
5. [Phase 1 — Environment & Ingestion](#phase-1--environment--ingestion)
6. [Phase 2 — Detection & Tracking](#phase-2--detection--tracking)
7. [Phase 3 — Pose Estimation & Biomechanics](#phase-3--pose-estimation--biomechanics)
8. [Phase 4 — Task Classification](#phase-4--task-classification)
9. [Phase 5 — Metrics Engine](#phase-5--metrics-engine)
10. [Phase 6 — Output Generation](#phase-6--output-generation)
11. [Phase 7 — Reporting](#phase-7--reporting)
12. [Phase 8 — Test Suite](#phase-8--test-suite)
13. [Configuration Reference](#configuration-reference)
14. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
15. [What We'd Do With More Time](#what-wed-do-with-more-time)

---

## 1. Project Overview

### Goal
Build a computer vision pipeline that ingests a factory floor video and outputs:
- An **annotated MP4** with per-person tracking IDs, task labels, and live metric overlays
- A **per-person metrics CSV/JSON** with task counts, active/idle time, movement, and zone dwell times
- A **PDF summary report** auto-generated from metrics
- Frame-level **heatmaps** of spatial presence per worker

### Selected Tasks (minimum 2 of 3)
| Task | Selected | Detection Strategy |
|---|---|---|
| Pick & Place small object into tray | ✅ | Wrist proximity + hand velocity + finger keypoint convergence |
| Lift & Place a tray | ✅ | Both wrist elevation + torso bend angle + object bounding box |
| Moving a rack | ✅ (bonus) | Full-body displacement + arm extension + large object tracking |

### Selected Metrics (minimum 2 of 3)
| Metric | Selected | Notes |
|---|---|---|
| Active vs. Idle Time | ✅ | Based on pose velocity thresholding |
| Total Movement | ✅ | Cumulative centroid displacement across frames |
| Time in Frame Zones | ✅ (bonus) | Frame divided into configurable grid zones |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│              Video File (MP4/AVI/MOV)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    DETECTION LAYER                          │
│   YOLOv8x-pose  →  Person bboxes + 17 keypoints/frame      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    TRACKING LAYER                           │
│   ByteTrack  →  Consistent person IDs across frames        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  BIOMECHANICS LAYER                         │
│   Joint angle computation, velocity, RULA risk scoring     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               TASK CLASSIFICATION LAYER                     │
│   Rule-based classifier with confidence scores per task    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   METRICS ENGINE                            │
│   Per-person aggregation: time, movement, zones, tasks     │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
┌──────────▼──────────┐       ┌──────────▼──────────────────┐
│   ANNOTATED VIDEO   │       │    METRICS OUTPUT           │
│   (MP4 + overlays)  │       │  CSV / JSON / PDF / Heatmap │
└─────────────────────┘       └─────────────────────────────┘
```

### Key Design Principle
**Modular pipeline with no shared mutable state between stages.** Each stage takes a well-defined input and emits a well-defined output. This makes each component independently testable and replaceable.

---

## 3. Directory Structure

```
factory-biomechanics-tracker/
├── config/
│   └── default.yaml              # All thresholds, zone definitions, model paths
├── data/
│   └── input/                    # Drop input videos here
├── output/
│   ├── annotated/                # Annotated output videos
│   ├── metrics/                  # CSV and JSON per run
│   ├── heatmaps/                 # Per-person heatmap PNGs
│   └── reports/                  # Auto-generated PDF reports
├── src/
│   ├── __init__.py
│   ├── ingestion.py              # Video reader, frame iterator
│   ├── detection.py              # YOLOv8 wrapper (bboxes + keypoints)
│   ├── tracking.py               # ByteTrack integration, ID management
│   ├── biomechanics.py           # Joint angles, velocity, RULA scoring
│   ├── classification.py         # Task classifier (rule-based + confidence)
│   ├── metrics.py                # Per-person metric aggregation
│   ├── annotation.py             # Frame overlay drawing logic
│   ├── heatmap.py                # Spatial heatmap generation
│   └── reporting.py              # PDF report generation
├── pipeline.py                   # Main orchestrator — runs full pipeline
├── requirements.txt
├── README.md
└── DESIGN.md                     # This document
```

---

## 4. Data Flow

### Per-Frame Data Contract

Every frame produces a list of `PersonFrame` objects:

```
PersonFrame:
  - track_id: int
  - bbox: [x1, y1, x2, y2]
  - keypoints: dict[str, (x, y, confidence)]  # 17 COCO keypoints
  - centroid: (x, y)
  - frame_index: int
  - timestamp_sec: float
```

### Per-Person Accumulated State

The metrics engine maintains a `PersonState` object per track_id:

```
PersonState:
  - track_id: int
  - frame_history: List[PersonFrame]
  - joint_angle_history: List[dict]
  - velocity_history: List[float]
  - task_log: List[TaskEvent]
  - zone_log: List[ZoneEvent]
  - is_active: bool
```

### Output Data Contract

```
PersonMetrics:
  - track_id: int
  - total_active_frames: int
  - total_idle_frames: int
  - active_ratio: float
  - total_movement_px: float
  - zone_dwell_times: dict[str, float]   # zone_id -> seconds
  - task_counts: dict[str, int]          # task_name -> count
  - task_events: List[TaskEvent]         # start_frame, end_frame, task, confidence
  - avg_rula_score: float
  - peak_rula_score: float
```

---

## Phase 1 — Environment & Ingestion

### Goals
- Set up reproducible environment
- Load video, validate it, expose a clean frame iterator

### Implementation

**`requirements.txt`** must pin:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
PyYAML>=6.0
lapx>=0.5.2          # ByteTrack dependency via ultralytics
matplotlib>=3.7.0
reportlab>=4.0.0     # PDF generation
scipy>=1.11.0        # Kalman smoothing
```

**`src/ingestion.py`**
- Class `VideoReader` wrapping `cv2.VideoCapture`
- Properties: `fps`, `total_frames`, `width`, `height`, `duration_sec`
- Method `iter_frames()` → yields `(frame_index, timestamp_sec, frame_bgr)`
- Validates file exists and is readable before iteration
- Logs video metadata on init

**`config/default.yaml`** — ingestion section:
```yaml
ingestion:
  input_path: "data/input/factory.mp4"
  skip_frames: 0          # process every Nth frame (0 = all)
  max_frames: null        # null = full video
```

### Exit Criteria
- Can iterate all frames of input video without error
- Frame metadata (index, timestamp) is accurate

---

## Phase 2 — Detection & Tracking

### Goals
- Detect all persons per frame with bounding boxes
- Assign consistent track IDs across frames using ByteTrack
- Smooth keypoints using Kalman filtering to reduce jitter

### Implementation

**`src/detection.py`** — `PoseDetector` class
- Wraps `ultralytics.YOLO("yolov8x-pose.pt")`
- Method `detect(frame_bgr)` → returns `List[PersonFrame]` (without track_id yet)
- Filters detections below `conf_threshold` (default: 0.4)
- Maps YOLO keypoint indices to named keypoints dict:
  ```
  KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
  ]
  ```

**`src/tracking.py`** — `PersonTracker` class
- Uses ByteTrack via ultralytics built-in tracker (`tracker="bytetrack.yaml"`)
- Method `update(detections, frame_bgr)` → returns `List[PersonFrame]` with `track_id` populated
- Maintains `track_history: dict[int, deque]` of last N centroids for trajectory drawing
- **Kalman smoothing**: maintain a `KalmanFilter` per keypoint per track_id using `scipy.signal` — smooth x,y positions over a rolling window of 5 frames

**`config/default.yaml`** — tracking section:
```yaml
tracking:
  conf_threshold: 0.4
  iou_threshold: 0.5
  keypoint_conf_threshold: 0.3
  kalman_window: 5
  max_disappeared_frames: 30
```

### Exit Criteria
- Each person in video has a stable, consistent track_id
- Keypoints don't jitter visibly on smoothed output
- No ID switches on persons who remain continuously visible

---

## Phase 3 — Pose Estimation & Biomechanics

### Goals
- Compute joint angles from keypoints each frame
- Compute per-person velocity (centroid movement between frames)
- Compute RULA-approximated ergonomic risk score per frame

### Implementation

**`src/biomechanics.py`** — `BiomechanicsAnalyzer` class

**Joint Angle Computation** — `compute_joint_angles(keypoints) -> dict[str, float]`

Compute angles (in degrees) for:
- `elbow_left`: angle at left_elbow between left_shoulder → left_elbow → left_wrist
- `elbow_right`: angle at right_elbow between right_shoulder → right_elbow → right_wrist
- `shoulder_left`: angle at left_shoulder between left_hip → left_shoulder → left_elbow
- `shoulder_right`: same for right side
- `trunk_flex`: angle between vertical axis and the line connecting mid_hip → mid_shoulder (torso forward lean)
- `wrist_height_left`: normalized y-position of left_wrist relative to shoulder (positive = above shoulder)
- `wrist_height_right`: same for right

Use `np.arctan2` for vector angle computation. Only compute when keypoint confidence > threshold.

**Velocity Computation** — `compute_velocity(current_frame, previous_frame) -> float`
- Euclidean distance between centroids in pixels
- Normalized to pixels/second using `fps`

**RULA Score Approximation** — `compute_rula_score(joint_angles) -> float`
- Simplified RULA Group A (arms + wrists):
  - Upper arm score: 1-4 based on shoulder angle
  - Lower arm score: 1-2 based on elbow angle
  - Wrist score: 1-3 based on wrist height
  - Sum → normalized to 1-7 scale
- Return score with label: `LOW` (1-2), `MEDIUM` (3-4), `HIGH` (5-6), `VERY_HIGH` (7)

**`config/default.yaml`** — biomechanics section:
```yaml
biomechanics:
  idle_velocity_threshold_px_per_sec: 15.0
  trunk_flex_threshold_deg: 20.0
  wrist_above_shoulder_threshold: 0.0   # normalized units
```

### Exit Criteria
- Joint angles are geometrically plausible for visible postures
- RULA scores increase when workers bend significantly
- Velocity correctly reads ~0 when person is stationary

---

## Phase 4 — Task Classification

### Goals
- Classify each person's activity per frame into one of: `pick_and_place`, `lift_and_place`, `move_rack`, `idle`, `other`
- Output a confidence score (0.0–1.0) per classification
- Log task events with start/end frame

### Implementation

**`src/classification.py`** — `TaskClassifier` class

**Design**: Rule-based classifier with confidence weighting. Each task has a set of binary conditions; confidence = fraction of conditions met.

**Task: `pick_and_place`**

Conditions (each worth equal weight):
1. At least one wrist is below shoulder height
2. Elbow angle between 60°–130° (reaching/grasping posture)
3. Wrist velocity is low (< `pick_velocity_threshold`) — precise movement
4. Trunk flex < 30° (not a heavy lift)
5. Person centroid is relatively stationary (< `stationary_threshold` px movement)

Confidence = (conditions_met / 5)

Trigger event when confidence > 0.6 for at least N consecutive frames (`min_task_duration_frames`).

**Task: `lift_and_place`**

Conditions:
1. Both wrists are rising (negative y velocity) OR both wrists above waist
2. Shoulder angle > 45° (arms raised)
3. Trunk flex > `trunk_flex_threshold_deg` (forward lean before lift)
4. Centroid velocity low (not walking while lifting)
5. RULA score ≥ MEDIUM

Confidence = (conditions_met / 5)

Trigger event when confidence > 0.6 for at least N consecutive frames.

**Task: `move_rack`**

Conditions:
1. Both arms extended (elbow angle > 140°)
2. Centroid velocity > `movement_velocity_threshold` (person is walking)
3. Trunk flex < 15° (upright pushing posture)
4. Wrists at approximately hip height

Confidence = (conditions_met / 4)

**Event Logging**

`TaskEvent`:
```
- track_id: int
- task: str
- start_frame: int
- end_frame: int
- confidence: float
- duration_sec: float
```

Use a state machine per person: `CANDIDATE → ACTIVE → COMPLETED`. Prevent double-counting if same task re-triggers within a cooldown window.

**`config/default.yaml`** — classification section:
```yaml
classification:
  confidence_threshold: 0.6
  min_task_duration_frames: 8
  task_cooldown_frames: 15
  pick_velocity_threshold: 30.0
  stationary_threshold: 20.0
  movement_velocity_threshold: 60.0
```

### Exit Criteria
- Each task produces reasonable event counts on the video
- No task fires continuously without pause (cooldown working)
- Confidence scores vary — not all 1.0 or 0.0

---

## Phase 5 — Metrics Engine

### Goals
- Aggregate per-person data across all frames into final metrics
- Compute zone dwell times based on configurable frame zones

### Implementation

**`src/metrics.py`** — `MetricsEngine` class

**Active vs Idle Time**
- A person is `ACTIVE` in a frame if centroid velocity > `idle_velocity_threshold` OR a task is currently firing
- A person is `IDLE` otherwise
- `active_ratio = active_frames / total_frames_seen`

**Total Movement**
- Sum of frame-to-frame centroid Euclidean distances (in pixels)
- Also store in normalized units (% of frame width) for comparability

**Zone Dwell Time**
- Frame is divided into a configurable grid (default: 3×2 = 6 zones, labeled A1–C2)
- Each frame, the person's centroid is mapped to a zone
- Accumulate seconds per zone

**`config/default.yaml`** — zones section:
```yaml
zones:
  grid_cols: 3
  grid_rows: 2
  labels:
    - ["A1", "B1", "C1"]
    - ["A2", "B2", "C2"]
```

**`PersonMetrics` Assembly**

`build_metrics(track_id, person_state) -> PersonMetrics`

Computed at end of video run. Stored as both:
- `output/metrics/metrics_{run_id}.csv` — one row per person
- `output/metrics/metrics_{run_id}.json` — full nested structure with task event details

### Exit Criteria
- CSV has one row per tracked person with all metric columns populated
- Active ratio is between 0.0 and 1.0 for all persons
- Zone dwell times sum to approximately total time seen

---

## Phase 6 — Output Generation

### Goals
- Write annotated video with per-person overlays
- Write per-person spatial heatmaps

### Implementation

**`src/annotation.py`** — `FrameAnnotator` class

Per-frame, draw (in this layer order):

1. **Zone grid** — semi-transparent colored rectangles with zone labels (drawn once, composited)
2. **Person bounding box** — color-coded by track_id, thickness 2
3. **Skeleton overlay** — connect keypoints with lines using COCO skeleton definition; opacity proportional to keypoint confidence
4. **Centroid trajectory** — draw last 30 centroid positions as fading trail
5. **Track ID label** — `Person {id}` above bounding box, bold white text with dark background
6. **Task label** — current active task name + confidence bar below ID label; color-coded by task type
7. **RULA badge** — small colored circle (green/yellow/orange/red) top-right of bbox
8. **Mini metric panel** — bottom-left corner of frame, semi-transparent: shows current active/idle ratio and movement for each tracked person
9. **Frame counter + timestamp** — top-right of frame

**Color scheme** (consistent across all outputs):
```
track colors: cycle through [#E74C3C, #3498DB, #2ECC71, #F39C12, #9B59B6, #1ABC9C]
pick_and_place: #3498DB (blue)
lift_and_place: #E67E22 (orange)
move_rack: #9B59B6 (purple)
idle: #95A5A6 (gray)
RULA LOW: #2ECC71
RULA MEDIUM: #F1C40F
RULA HIGH: #E67E22
RULA VERY_HIGH: #E74C3C
```

**`src/heatmap.py`** — `HeatmapGenerator` class
- Accumulates centroid positions across all frames per person into a 2D histogram
- Uses `matplotlib` with `hot` colormap, overlaid on first video frame (grayscale)
- Saves PNG per person: `output/heatmaps/heatmap_person_{id}.png`
- Also saves combined heatmap across all persons

**Video Writing**
- Use `cv2.VideoWriter` with `mp4v` codec
- Match input fps and resolution
- Write to `output/annotated/annotated_{run_id}.mp4`

### Exit Criteria
- Annotated video plays smoothly with all overlays visible and readable
- Heatmaps correctly reflect where each person spent most time
- Colors are consistent between video and heatmaps

---

## Phase 7 — Reporting

### Goals
- Auto-generate a polished PDF summary report from metrics

### Implementation

**`src/reporting.py`** — `ReportGenerator` class using `reportlab`

**PDF Structure:**

```
Page 1: Cover
  - Title: Factory Floor Analysis Report
  - Run date, video filename, duration, person count

Page 2: Summary Table
  - One row per person: ID, active%, total movement, top task, RULA avg

Page 3+: Per-Person Detail (one page per person)
  - Person ID header
  - Active/Idle pie chart (matplotlib → embed as image)
  - Zone dwell time bar chart
  - Task event timeline (horizontal bar chart, one row per task type)
  - RULA score trend (line chart)
  - Heatmap thumbnail

Final Page: Methodology Note
  - Brief explanation of detection model, tracking algorithm, task classification logic
```

Save to: `output/reports/report_{run_id}.pdf`

### Exit Criteria
- PDF is readable and professional
- All charts render correctly with no placeholder text
- Report is auto-generated with zero manual steps

---

## Pipeline Orchestrator

**`pipeline.py`** — `run_pipeline(config_path)` function

```
1. Load config from YAML
2. Initialize VideoReader → log metadata
3. Initialize PoseDetector, PersonTracker, BiomechanicsAnalyzer, TaskClassifier, MetricsEngine
4. Initialize FrameAnnotator, VideoWriter
5. For each frame:
   a. Detect persons
   b. Update tracker → assign IDs
   c. Compute biomechanics per person
   d. Classify tasks per person
   e. Update metrics engine
   f. Annotate frame
   g. Write annotated frame to video
6. Finalize metrics → write CSV + JSON
7. Generate heatmaps
8. Generate PDF report
9. Log summary to console
```

**CLI Entry Point:**
```bash
python pipeline.py --config config/default.yaml --input data/input/factory.mp4
```

---

## Configuration Reference

Full `config/default.yaml`:

```yaml
ingestion:
  input_path: "data/input/factory.mp4"
  skip_frames: 0
  max_frames: null

tracking:
  conf_threshold: 0.4
  iou_threshold: 0.5
  keypoint_conf_threshold: 0.3
  kalman_window: 5
  max_disappeared_frames: 30

biomechanics:
  idle_velocity_threshold_px_per_sec: 15.0
  trunk_flex_threshold_deg: 20.0
  wrist_above_shoulder_threshold: 0.0

classification:
  confidence_threshold: 0.6
  min_task_duration_frames: 8
  task_cooldown_frames: 15
  pick_velocity_threshold: 30.0
  stationary_threshold: 20.0
  movement_velocity_threshold: 60.0

zones:
  grid_cols: 3
  grid_rows: 2

output:
  annotated_video: true
  metrics_csv: true
  metrics_json: true
  heatmaps: true
  pdf_report: true
  run_id: null  # auto-generated from timestamp if null
```

---

## Phase 8 — Test Suite

### Goals
- Validate every module independently without needing a real video
- Catch regressions if any module is modified
- Give Claude Code a clear signal when something is broken

### Structure

```
tests/
├── conftest.py                  # Shared fixtures
├── test_ingestion.py
├── test_biomechanics.py
├── test_classification.py
├── test_metrics.py
├── test_annotation.py
├── test_heatmap.py
└── test_pipeline_integration.py
```

Use `pytest`. No external video needed — all tests use synthetic fixtures.

---

### `conftest.py` — Shared Fixtures

```python
# Synthetic PersonFrame with controllable keypoints
@pytest.fixture
def person_frame_standing():
    # Returns a PersonFrame with upright posture keypoints
    # shoulders above hips, arms at sides, elbows ~170deg

@pytest.fixture
def person_frame_reaching():
    # Returns a PersonFrame with one wrist extended forward
    # elbow angle ~90deg, wrist below shoulder

@pytest.fixture
def person_frame_lifting():
    # Returns a PersonFrame with both wrists raised above shoulders
    # trunk forward lean, shoulder angle >45deg

@pytest.fixture
def person_frame_walking():
    # Returns a PersonFrame with high centroid velocity
    # arms slightly extended, upright posture

@pytest.fixture
def mock_video_path(tmp_path):
    # Creates a 3-second synthetic MP4 (black frames, 30fps)
    # using cv2.VideoWriter — no real factory video needed

@pytest.fixture
def default_config():
    # Loads config/default.yaml and returns as dict
```

---

### `test_ingestion.py`

| Test | What It Validates |
|---|---|
| `test_video_reader_opens_valid_file` | VideoReader initializes without error on mock video |
| `test_video_reader_metadata` | fps, width, height, total_frames, duration_sec are correct |
| `test_iter_frames_yields_correct_count` | Number of frames yielded matches total_frames |
| `test_iter_frames_timestamp_increases` | Each frame timestamp is greater than previous |
| `test_video_reader_raises_on_missing_file` | FileNotFoundError raised on bad path |
| `test_skip_frames_reduces_output` | skip_frames=2 yields ~1/3 of total frames |

---

### `test_biomechanics.py`

| Test | What It Validates |
|---|---|
| `test_elbow_angle_straight_arm` | ~170° when arm is straight |
| `test_elbow_angle_bent_arm` | ~90° when arm is bent at right angle |
| `test_trunk_flex_upright` | ~0° when spine is vertical |
| `test_trunk_flex_leaning` | >20° when torso tilted forward |
| `test_wrist_height_above_shoulder` | Returns positive value when wrist above shoulder |
| `test_wrist_height_below_shoulder` | Returns negative value when wrist below shoulder |
| `test_velocity_stationary` | Returns ~0 when centroid unchanged between frames |
| `test_velocity_moving` | Returns correct px/sec given known displacement and fps |
| `test_rula_low_score_upright` | Standing posture scores 1–2 |
| `test_rula_high_score_lifting` | Overhead reach posture scores 5–7 |
| `test_rula_missing_keypoints` | Handles low-confidence keypoints gracefully, no crash |
| `test_compute_joint_angles_returns_all_keys` | Output dict contains all expected angle keys |

---

### `test_classification.py`

| Test | What It Validates |
|---|---|
| `test_pick_and_place_fires_on_correct_posture` | Confidence > 0.6 on `person_frame_reaching` fixture |
| `test_pick_and_place_does_not_fire_when_walking` | Confidence < 0.6 when centroid velocity is high |
| `test_lift_and_place_fires_on_correct_posture` | Confidence > 0.6 on `person_frame_lifting` fixture |
| `test_lift_and_place_does_not_fire_when_standing` | Confidence < 0.6 on `person_frame_standing` fixture |
| `test_move_rack_fires_on_walking_with_extended_arms` | Confidence > 0.6 on `person_frame_walking` fixture |
| `test_task_cooldown_prevents_double_counting` | Same task does not fire twice within cooldown window |
| `test_min_duration_prevents_flash_events` | Task with < min_task_duration_frames does not produce event |
| `test_confidence_is_between_0_and_1` | All task confidences are valid floats in [0.0, 1.0] |
| `test_task_event_has_required_fields` | TaskEvent contains track_id, task, start_frame, end_frame, confidence |
| `test_idle_when_no_task_active` | Returns `idle` when no task condition met |

---

### `test_metrics.py`

| Test | What It Validates |
|---|---|
| `test_active_ratio_all_active` | active_ratio = 1.0 when all frames have high velocity |
| `test_active_ratio_all_idle` | active_ratio = 0.0 when all frames have zero velocity |
| `test_active_ratio_mixed` | active_ratio ~0.5 for half active, half idle frame sequence |
| `test_total_movement_zero_stationary` | total_movement_px = 0 when centroid never moves |
| `test_total_movement_known_displacement` | Returns correct sum for known centroid trajectory |
| `test_zone_assignment_correct` | Centroid in top-left maps to zone A1 |
| `test_zone_dwell_time_sums_to_total` | Sum of all zone dwell times ≈ total time seen |
| `test_task_counts_correct` | task_counts matches number of TaskEvents per task type |
| `test_metrics_output_has_all_fields` | PersonMetrics contains all required fields |
| `test_multiple_persons_tracked_independently` | Two persons with different track_ids produce separate metrics |

---

### `test_annotation.py`

| Test | What It Validates |
|---|---|
| `test_annotate_frame_returns_same_shape` | Output frame shape matches input frame shape |
| `test_annotate_frame_modifies_pixels` | Output frame differs from input (something was drawn) |
| `test_no_crash_on_missing_keypoints` | Annotator handles None/empty keypoints without error |
| `test_no_crash_on_single_person` | Works correctly with exactly one tracked person |
| `test_no_crash_on_zero_persons` | Works correctly with empty person list |
| `test_track_id_color_cycles` | Different track IDs get different colors |

---

### `test_heatmap.py`

| Test | What It Validates |
|---|---|
| `test_heatmap_output_is_png` | Output file is a valid PNG |
| `test_heatmap_shape_matches_frame` | Heatmap dimensions match input video dimensions |
| `test_heatmap_nonzero_where_person_present` | Pixels are nonzero at centroid locations |
| `test_combined_heatmap_includes_all_persons` | Combined heatmap has nonzero pixels for all person positions |

---

### `test_pipeline_integration.py`

| Test | What It Validates |
|---|---|
| `test_pipeline_runs_on_mock_video` | Full pipeline completes without error on 3-second synthetic video |
| `test_pipeline_produces_annotated_video` | Output MP4 exists and is nonzero size |
| `test_pipeline_produces_csv` | Output CSV exists and has at least one data row |
| `test_pipeline_produces_json` | Output JSON is valid and parseable |
| `test_pipeline_produces_heatmaps` | At least one heatmap PNG exists in output dir |
| `test_pipeline_produces_pdf` | Output PDF exists and is nonzero size |
| `test_pipeline_run_id_namespaces_outputs` | All output files contain the run_id string |

---

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_biomechanics.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run fast (skip integration test)
pytest tests/ -v -k "not integration"
```

### Exit Criteria for Phase 8

- All unit tests pass (0 failures)
- Integration test passes on synthetic video
- Coverage report shows ≥ 80% coverage across `src/`
- No test uses a real video file — all use fixtures or synthetic data

---

## Design Decisions & Tradeoffs

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Detection model | YOLOv8x-pose | Best accuracy/speed tradeoff with pose built-in | MediaPipe (faster but less accurate on crowds), OpenPose (slower) |
| Tracking | ByteTrack | State-of-the-art multi-object tracking; handles occlusion well | DeepSORT (requires ReID, slower), SORT (no occlusion handling) |
| Task classification | Rule-based + confidence | Interpretable, debuggable, no training data needed | ML classifier (would need labeled data we don't have) |
| Keypoint smoothing | Kalman filter per keypoint | Removes temporal jitter without introducing latency | Gaussian blur over time (loses responsiveness) |
| Ergonomics scoring | RULA approximation | Industry-standard framework; signals domain knowledge | Custom scoring (less credible without validation) |
| PDF generation | ReportLab | Pure Python, no browser dependency | WeasyPrint (HTML→PDF, heavier), pdfkit (requires wkhtmltopdf) |

---

## What We'd Do With More Time

1. **ReID model** — use a re-identification network to recover track IDs after long occlusions
2. **Learned task classifier** — label 200 clips and train a temporal CNN (e.g. SlowFast) for higher precision
3. **3D pose lifting** — use a model like MotionBERT to lift 2D keypoints to 3D for more accurate joint angles
4. **Object detection integration** — detect trays and racks explicitly; use proximity to person hands to confirm task interactions
5. **Real-time mode** — refactor pipeline to process live RTSP camera stream with sub-second latency
6. **Multi-camera support** — fuse views from multiple angles to resolve occlusions and improve zone accuracy
7. **Fatigue modeling** — track RULA score trend over shift duration to flag cumulative ergonomic risk
8. **Web dashboard** — serve metrics and annotated clips via FastAPI + simple frontend for ops teams

---

*Document version: 1.0 — Generated for Measurement Labs take-home assessment*