<div align="center">

# Factory Biomechanics Tracker

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Temporal_CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose_Estimation-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_Pipeline-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/sklearn-K--Means-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF_Reports-FF6F61?style=for-the-badge)

![Tests](https://img.shields.io/badge/Tests-62%20Passing-2ECC71?style=flat-square)
![Persons Tracked](https://img.shields.io/badge/Persons%20Tracked-52-3498DB?style=flat-square)
![Frames](https://img.shields.io/badge/Frames%20Processed-9025-E67E22?style=flat-square)
![RULA](https://img.shields.io/badge/RULA-Full%20Scoring-9B59B6?style=flat-square)
![Tasks](https://img.shields.io/badge/Tasks-3%20Detected-E74C3C?style=flat-square)
![Classifier](https://img.shields.io/badge/Classifier-Learned%20CNN-1ABC9C?style=flat-square)

*Human pose estimation and activity recognition for factory floor analysis. Tracks worker biomechanics, classifies tasks using a learned temporal CNN, and generates productivity metrics from video.*

</div>

---

## Demo

<p align="center">
  <img src="output/annotated/preview.gif" alt="Annotated output preview" width="100%">
</p>

<p align="center"><em>8-second preview of the annotated output — bounding boxes, skeletons, task labels, RULA badges, zone grid, and live metrics. Full 5-minute video available in <code>output/annotated/</code>.</em></p>

---

## Development Timeline

> **Total development time: 3 hours** | 28 source files | 62 tests | 9025 frames processed

```mermaid
gantt
    title Development Timeline - 3 Hours End to End
    dateFormat HH:mm
    axisFormat %H:%M
    todayMarker off

    section Foundation
    Project structure and YAML config             :done, f1, 00:00, 10m
    Dependency management                         :done, f2, after f1, 5m
    Shared data contracts and models              :done, f3, after f2, 10m

    section Detection and Tracking
    Video ingestion with frame iterator           :done, c1, after f3, 10m
    YOLOv8m-pose detection wrapper                :done, c2, after c1, 10m
    ByteTrack multi-object tracking               :done, c3, after c2, 15m

    section Biomechanics
    Joint angle computation with arctan2          :done, c4, after c3, 10m
    Centroid velocity in pixels per second        :done, c4b, after c4, 5m
    RULA ergonomic risk scoring                   :done, c5, after c4b, 10m

    section Task Classification
    Rule-based classifier for 3 tasks             :done, c6, after c5, 10m
    State machine with cooldown logic             :done, c6b, after c6, 5m

    section Metrics and Output
    Metrics engine for active idle and zones      :done, c7, after c6b, 10m
    Eight-layer frame annotator                   :done, o1, after c7, 10m
    Spatial heatmap generator                     :done, o2, after o1, 5m
    PDF report with embedded charts               :done, o3, after o2, 10m
    Pipeline orchestrator and CLI                 :done, o4, after o3, 5m

    section Learned Upgrades
    Full RULA with lookup Tables A B C            :crit, u1, after o4, 10m
    1D Temporal CNN task classifier               :crit, u2, after o4, 15m
    ReID post-tracking with appearance matching   :crit, u3, after o4, 10m
    2D Kalman filters for position and velocity   :crit, u3b, after u3, 5m
    Foreshortening detection and smoothing        :crit, u4, after u1, 5m

    section Validation and Delivery
    62-test suite with synthetic fixtures         :done, t1, after u4, 10m
    First-pass run on 9025 frames                 :active, t2, after t1, 12m
    Train TemporalTaskNet on 46k features         :active, t3, after t2, 2m
    Final run with learned classifier             :active, t4, after t3, 13m
    Documentation and README                      :done, t5, after t4, 3m
```

## Pipeline Execution Per Video

> Single run on 1920x1080 at 30fps, 9025 frames, GPU-accelerated via MPS

```mermaid
gantt
    title Pipeline Execution - 13 Minutes per Video
    dateFormat mm:ss
    axisFormat %M:%S
    todayMarker off

    section Ingestion
    Load and validate video metadata              :active, p1, 00:00, 2s

    section Per-Frame Processing Loop
    YOLOv8m-pose detection on MPS GPU             :p2a, 00:02, 11m
    ByteTrack ID assignment and ReID merge        :p2b, 00:02, 11m
    2D Kalman smoothing on 17 keypoints           :p2c, 00:02, 11m
    Joint angles and neck and trunk scoring       :p3a, 00:02, 11m
    Full RULA scoring with Table C lookup         :p3b, 00:02, 11m
    Velocity and foreshortening confidence        :p3c, 00:02, 11m
    Learned CNN inference on 30-frame buffer      :p4a, 00:02, 11m
    State machine update and event logging        :p4b, 00:02, 11m
    Metrics accumulation for zones and movement   :p5, 00:02, 11m
    Annotate frame with 8 layers and write MP4    :p6, 00:02, 11m

    section Post-Processing Export
    Export metrics CSV with 1 row per person       :crit, p7a, 11:04, 2s
    Export metrics JSON with task event detail     :crit, p7b, after p7a, 2s
    Generate 52 spatial heatmaps and combined     :crit, p7c, after p7b, 25s
    Generate PDF report with charts               :crit, p7d, after p7c, 35s
```

## Training Pipeline - Self-Supervised from Video

> No manual labels required. The video trains its own classifier.

```mermaid
gantt
    title Learned Classifier Training Pipeline
    dateFormat mm:ss
    axisFormat %M:%S
    todayMarker off

    section Data Collection via First-Pass Pipeline
    Run full pipeline with rule-based labels      :done, d1, 00:00, 12m
    Collect 46142 feature records at 11 dims      :done, d2, 00:00, 12m

    section Feature Preparation
    Group records by track ID and sort            :done, fe1, 12:00, 2s
    Create 8997 sliding windows at stride 5       :done, fe2, after fe1, 3s
    Z-score normalize per feature channel         :done, fe3, after fe2, 1s
    Save normalization stats for inference        :done, fe4, after fe3, 1s

    section Unsupervised Label Refinement
    K-means clustering with 8 clusters            :done, lr1, after fe4, 4s
    Compare cluster majority to pseudo-labels     :done, lr2, after lr1, 1s
    Correct low-confidence disagreements          :done, lr3, after lr2, 1s
    Stratified 80-20 train validation split       :done, lr4, after lr3, 1s

    section Model Training
    TemporalTaskNet 3 Conv1d layers 24K params    :crit, mt1, after lr4, 40s
    Adam optimizer with class-weighted loss       :crit, mt2, after lr4, 40s
    Temporal jitter and Gaussian noise augment    :crit, mt3, after lr4, 40s
    Early stopping on val loss patience 10        :crit, mt4, after lr4, 40s

    section Artifacts
    Save best model weights to task_model.pt      :done, out1, after mt1, 1s
    Save feature stats to JSON                    :done, out2, after mt1, 1s
```

---

## Setup

```bash
pip install -r requirements.txt
```

> Requires **Python 3.10+**. The YOLOv8m-pose model is downloaded automatically on first run.

## Usage

### Full Pipeline (with pre-trained model)

```bash
python pipeline.py --input data/input/video.mp4
```

### Training Workflow (from scratch)

```bash
# Step 1: First pass — extract features with rule-based pseudo-labels
python pipeline.py --input data/input/video.mp4 --extract-features

# Step 2: Train the learned classifier on extracted features
python pipeline.py --train

# Step 3: Run final pipeline with learned classifier (enable in config)
python pipeline.py --input data/input/video.mp4
```

### CLI Options

```bash
python pipeline.py --help
  --config CONFIG     Path to YAML config (default: config/default.yaml)
  --input INPUT       Path to input video (overrides config)
  --extract-features  Save per-frame features for classifier training
  --train             Train learned classifier from extracted features
```

---

## Outputs

All outputs are namespaced by run ID (`YYYYMMDD_HHMMSS`):

| Output | Location | Description |
|---|---|---|
| Annotated Video | `output/annotated/annotated_{run_id}.mp4` | Video with bboxes, skeletons, task labels, RULA badges, zone grid, metric overlays |
| Metrics CSV | `output/metrics/metrics_{run_id}.csv` | One row per tracked person with all metrics |
| Metrics JSON | `output/metrics/metrics_{run_id}.json` | Full nested structure with task event detail |
| Heatmaps | `output/heatmaps/heatmap_person_{id}_{run_id}.png` | Per-person spatial presence heatmaps |
| PDF Report | `output/reports/report_{run_id}.pdf` | Cover, summary table, per-person charts, methodology |

---

## Architecture

```
Video → YOLOv8m-pose → ByteTrack + ReID → Biomechanics (Full RULA)
  → Learned Task Classifier (1D Temporal CNN) → Metrics Engine
  → [Annotated Video, CSV, JSON, Heatmaps, PDF]
```

Each module in `src/` is independently importable. Data flows through `PersonFrame → PersonState → PersonMetrics`.

---

## Key Components

### Task Classification (Learned)

The system uses a **1D Temporal CNN** (TemporalTaskNet) trained on the input video itself:
- Extracts 11 features per frame: 7 joint angles + velocity + centroid xy + RULA score
- Processes 30-frame sliding windows (~1 second at 30fps)
- Architecture: 3 Conv1d layers → BatchNorm → GlobalAvgPool → FC (24K parameters)
- Initial pseudo-labels from rule-based heuristics, refined via K-means clustering
- Falls back to rule-based for the first ~1 second per person (buffer warmup)

**Tasks detected**: `pick_and_place` `lift_and_place` `move_rack` `idle`

### RULA Ergonomic Scoring (Full)

Implements the complete RULA worksheet (McAtamney & Corlett, 1993):
- **Group A**: Upper arm (1-6), lower arm (1-3), wrist (1-4), wrist twist
- **Group B**: Neck (1-6), trunk (1-6), legs (1-2)
- Standard lookup **Tables A, B, C** for final score (1-7)
- Muscle use and force/load adjustments

### Tracking (ByteTrack + ReID)

- **ByteTrack** via ultralytics for primary tracking
- **Post-tracking ReID**: HSV histogram appearance matching + spatial prediction + temporal penalty
- **2D Kalman filters** per keypoint (state: position + velocity) for smoothing
- **90-frame** disappearance tolerance (3 seconds at 30fps)

### 2D Pose Mitigation

- **Foreshortening detection**: Flags unreliable angles when limbs point toward/away from camera
- **Temporal smoothing**: 5-frame weighted moving average on joint angles
- **Body proportion validation**: Detects implausible 2D projection artifacts

---

## Tests

```bash
pytest tests/ -v                           # All tests
pytest tests/ -k "not integration"         # Fast (skip integration)
pytest tests/ --cov=src --cov-report=term  # With coverage
```

> All tests use synthetic data — no real video needed. **62 tests passing.**

---

## Configuration

All settings in `config/default.yaml`:

| Section | Controls |
|---|---|
| `ingestion` | Input path, frame skipping, max frames |
| `tracking` | Detection thresholds, ReID weights, Kalman settings |
| `biomechanics` | Velocity thresholds, full RULA toggle, angle smoothing |
| `classification` | Confidence, duration, cooldown thresholds |
| `learned_classification` | Enable/disable learned model, training hyperparams |
| `zones` | Grid dimensions for dwell tracking |
| `output` | Toggle each output type on/off |

---

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Detection model | YOLOv8m-pose | Best accuracy/speed tradeoff with built-in pose, GPU-accelerated |
| Tracking | ByteTrack + ReID | Handles occlusion; ReID merges fragmented tracks (**190 → 52 persons**) |
| Task classification | Learned 1D CNN | Trained on video itself via pseudo-labels; generalizes better than hard rules |
| Keypoint smoothing | 2D Kalman (pos+vel) | Tracks position and velocity; predicts during brief occlusions |
| Ergonomic scoring | Full RULA (Groups A+B) | Industry-standard with proper lookup tables, not linear approximation |
| PDF generation | ReportLab | Pure Python, no browser dependency, embeds matplotlib charts |

---

## Current Limitations

| Area | Limitation | Impact |
|---|---|---|
| Pose estimation | 2D keypoints only, no depth | Foreshortening reduces angle accuracy when limbs point toward or away from camera |
| Task classification | Trained on pseudo-labels from a single video | Model may not generalize well to different factory layouts or camera angles |
| Object interaction | No explicit object detection | Tasks inferred from pose alone, cannot confirm if worker is actually holding a tray or rack |
| Tracking | ReID uses color histograms only | Workers in identical uniforms may be confused after long occlusions |
| RULA scoring | Wrist twist not detectable in 2D | Wrist twist defaults to neutral, underestimating risk for twisting motions |
| Scalability | Single-threaded frame processing | Processes one frame at a time, no batch GPU inference or multiprocessing |
| Camera | Single fixed viewpoint | Occlusions from one angle cannot be resolved without additional views |

---

## What I Would Do With More Time

```mermaid
gantt
    title Extended Roadmap - Next Phases
    dateFormat HH:mm
    axisFormat %H:%M
    todayMarker off

    section Delivered
    Full pipeline with learned classifier         :done, a1, 00:00, 60m
    Full RULA and ReID and 2D mitigation          :done, a2, 01:00, 60m
    Training and validation and final outputs     :done, a3, 02:00, 60m

    section Phase A - Depth and Object Awareness
    MotionBERT for 2D to 3D pose lifting         :active, pa1, 03:00, 25m
    Fine-tune YOLOv8 to detect trays and racks   :pa2, after pa1, 20m
    Hand-object proximity task confirmation      :pa3, after pa2, 15m

    section Phase B - Supervised Classification
    Manually label 200 video clips               :pb1, after pa3, 20m
    Train SlowFast dual-pathway temporal CNN     :pb2, after pb1, 25m
    Fatigue model with RULA trend over shift     :pb3, after pb2, 15m

    section Phase C - Production Deployment
    Real-time RTSP stream under 1s latency       :pc1, after pb3, 20m
    FastAPI dashboard with live metrics           :pc2, after pc1, 20m
    Multi-camera fusion via triangulation        :pc3, after pc2, 15m
    Docker container and GitHub Actions CI       :pc4, after pc3, 5m
```

### Phase A - Depth and Object Awareness
- **3D pose lifting** with MotionBERT to convert 2D keypoints to 3D joint positions, eliminating foreshortening issues and giving accurate joint angles regardless of camera angle
- **Object detection** for trays, racks, and tools using a fine-tuned YOLOv8, confirming task interactions by checking hand-object proximity rather than relying on pose alone

### Phase B - Supervised Classification
- **Hand-labeled training data** from 200 annotated video clips with ground-truth task boundaries, replacing pseudo-labels with verified ones
- **SlowFast temporal CNN** processing both slow posture and fast motion pathways for more precise task segmentation
- **Fatigue modeling** tracking each worker's RULA score trend across the full shift to flag cumulative ergonomic risk before injury occurs

### Phase C - Production Deployment
- **Real-time mode** processing live RTSP camera streams with sub-second latency using batch GPU inference
- **Web dashboard** with FastAPI backend serving live metrics, annotated video clips, and heatmaps to a browser-based ops dashboard
- **Multi-camera fusion** combining views from multiple cameras to resolve occlusions and improve zone accuracy through triangulation
- **Docker and CI/CD** for containerized deployment with GitHub Actions running the test suite on every push
