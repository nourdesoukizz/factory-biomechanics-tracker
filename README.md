# Factory Biomechanics Tracker

Human pose estimation and activity recognition system for factory floor analysis. Tracks worker biomechanics, classifies tasks using a learned temporal CNN, and generates productivity metrics from video.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The YOLOv8m-pose model is downloaded automatically on first run.

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

## Outputs

All outputs are namespaced by run ID (`YYYYMMDD_HHMMSS`):

| Output | Location | Description |
|---|---|---|
| Annotated Video | `output/annotated/annotated_{run_id}.mp4` | Video with bboxes, skeletons, task labels, RULA badges, zone grid, metric overlays |
| Metrics CSV | `output/metrics/metrics_{run_id}.csv` | One row per tracked person with all metrics |
| Metrics JSON | `output/metrics/metrics_{run_id}.json` | Full nested structure with task event detail |
| Heatmaps | `output/heatmaps/heatmap_person_{id}_{run_id}.png` | Per-person spatial presence heatmaps |
| PDF Report | `output/reports/report_{run_id}.pdf` | Cover, summary table, per-person charts, methodology |

## Architecture

```
Video → YOLOv8m-pose → ByteTrack + ReID → Biomechanics (Full RULA)
  → Learned Task Classifier (1D Temporal CNN) → Metrics Engine
  → [Annotated Video, CSV, JSON, Heatmaps, PDF]
```

Each module in `src/` is independently importable. Data flows through `PersonFrame → PersonState → PersonMetrics`.

## Task Classification (Learned)

The system uses a **1D Temporal CNN** (TemporalTaskNet) trained on the input video itself:
- Extracts 11 features per frame: 7 joint angles + velocity + centroid xy + RULA score
- Processes 30-frame sliding windows (~1 second at 30fps)
- Architecture: 3 Conv1d layers → BatchNorm → GlobalAvgPool → FC (24K parameters)
- Initial pseudo-labels from rule-based heuristics, refined via K-means clustering
- Falls back to rule-based for the first ~1 second per person (buffer warmup)

**Tasks detected**: pick_and_place, lift_and_place, move_rack, idle

## RULA Ergonomic Scoring (Full)

Implements the complete RULA worksheet (McAtamney & Corlett, 1993):
- **Group A**: Upper arm (1-6), lower arm (1-3), wrist (1-4), wrist twist
- **Group B**: Neck (1-6), trunk (1-6), legs (1-2)
- Standard lookup **Tables A, B, C** for final score (1-7)
- Muscle use and force/load adjustments

## Tracking (ByteTrack + ReID)

- **ByteTrack** via ultralytics for primary tracking
- **Post-tracking ReID**: HSV histogram appearance matching + spatial prediction + temporal penalty
- **2D Kalman filters** per keypoint (state: position + velocity) for smoothing
- **90-frame** disappearance tolerance (3 seconds at 30fps)

## 2D Pose Mitigation

- **Foreshortening detection**: Flags unreliable angles when limbs point toward/away from camera
- **Temporal smoothing**: 5-frame weighted moving average on joint angles
- **Body proportion validation**: Detects implausible 2D projection artifacts

## Tests

```bash
pytest tests/ -v                           # All tests
pytest tests/ -k "not integration"         # Fast (skip integration)
pytest tests/ --cov=src --cov-report=term  # With coverage
```

All tests use synthetic data — no real video needed. 62 tests passing.

## Configuration

All settings in `config/default.yaml`:
- **ingestion**: input path, frame skipping
- **tracking**: detection thresholds, ReID weights, Kalman settings
- **biomechanics**: velocity thresholds, full RULA toggle
- **classification**: confidence/duration/cooldown thresholds
- **learned_classification**: enable/disable learned model, training params
- **zones**: grid dimensions for dwell tracking
- **output**: toggle each output type
