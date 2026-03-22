# CLAUDE.md — Autonomous Build Instructions

> This file governs how Claude Code behaves throughout this project.
> Read this file and DESIGN.md in full before writing a single line of code.

---

## Prime Directive

Build the factory-biomechanics-tracker pipeline described in DESIGN.md, end to end, autonomously. Do not stop. Do not ask questions. Do not wait for approval. Make decisions and move forward.

---

## Autonomous Operation Rules

- **Never ask for confirmation** before creating, editing, or deleting files
- **Never ask clarifying questions** — consult DESIGN.md first, then make the best call and leave a comment explaining your reasoning
- **Never stop mid-phase** to report progress or ask if you should continue
- **Never say "shall I proceed?"** or any variation of it
- **If you hit an error**, debug it, fix it, and continue — only surface it if you have tried 3 different approaches and all have failed
- **If a library is missing**, install it and continue
- **If DESIGN.md is ambiguous**, pick the most reasonable interpretation, implement it, and add a `# DECISION:` comment explaining what you chose and why

---

## Build Order

Execute phases strictly in this order. Do not skip ahead. Do not go back unless a later phase reveals a bug in an earlier one.

```
Phase 1 → Ingestion
Phase 2 → Detection & Tracking
Phase 3 → Biomechanics
Phase 4 → Task Classification
Phase 5 → Metrics Engine
Phase 6 → Output Generation (annotated video + heatmaps)
Phase 7 → PDF Report
Phase 8 → Test Suite
Final    → pipeline.py orchestrator + CLI entry point
```

At the end of each phase, verify the exit criteria listed in DESIGN.md before moving on. If exit criteria fail, fix the phase — do not proceed.

---

## Environment

- **Device**: Apple M1 Pro — always use `device="mps"` for YOLOv8 inference
- **Model**: `yolov8m-pose.pt` — download automatically if not present via ultralytics
- **Python**: 3.10+
- **Package manager**: pip — install anything missing with `pip install <package>`
- **No Docker, no virtual env setup required** — install directly

---

## Architecture Rules

- Every configurable value lives in `config/default.yaml` — no hardcoded thresholds anywhere in source code
- Every module in `src/` is independently importable with no side effects on import
- Data flows strictly as defined in DESIGN.md — `PersonFrame` → `PersonState` → `PersonMetrics`
- No module imports from another module at the same level except through well-defined interfaces
- `pipeline.py` is the only orchestrator — nothing else should run the full pipeline

---

## Code Quality Standards

Apply these to every file, every function, every class — no exceptions:

- **Type hints** on every function signature (parameters and return type)
- **Docstrings** on every class and every public method — one line minimum
- **No magic numbers** — named constants or config values only
- **No print statements** — use Python `logging` module throughout (`logging.getLogger(__name__)`)
- **Error handling** — wrap I/O and model inference in try/except with informative error messages
- **Early returns** over deeply nested if/else
- **Single responsibility** — if a function is doing two things, split it

---

## File & Output Conventions

- All source code lives in `src/`
- All outputs go to `output/annotated/`, `output/metrics/`, `output/heatmaps/`, `output/reports/`
- Run ID is a timestamp string: `YYYYMMDD_HHMMSS` — used as suffix on all output files
- Never overwrite existing output files — always use run ID to namespace them
- Config is always loaded from `config/default.yaml` unless `--config` CLI flag overrides it

---

## What We Are Building — Quick Reference

**Input**: Factory floor video (MP4)

**Tasks to detect**:
- `pick_and_place` — wrist proximity + low velocity + elbow angle
- `lift_and_place` — wrist elevation + shoulder angle + trunk flex
- `move_rack` — arm extension + centroid velocity + upright posture

**Metrics to compute**:
- Active vs. Idle time (velocity thresholding)
- Total movement (cumulative centroid displacement in pixels)
- Time in frame zones (3×2 configurable grid, labeled A1–C2)

**Outputs**:
1. Annotated MP4 — bounding boxes, skeletons, track IDs, task labels, RULA badges, zone grid, metric overlay
2. `metrics_{run_id}.csv` — one row per person, all metrics
3. `metrics_{run_id}.json` — full nested structure with task event detail
4. `heatmap_person_{id}.png` + `heatmap_combined.png`
5. `report_{run_id}.pdf` — cover, summary table, per-person charts, methodology note

---

## Annotation Overlay Spec

Draw layers in this exact order (bottom to top):
1. Zone grid (semi-transparent)
2. Bounding box (color by track ID)
3. Skeleton (keypoint lines, opacity = keypoint confidence)
4. Centroid trajectory trail (last 30 frames, fading)
5. Track ID label + task label + confidence bar
6. RULA badge (colored circle: green/yellow/orange/red)
7. Global metric panel (bottom-left, semi-transparent)
8. Frame counter + timestamp (top-right)

Track ID colors — cycle through in order:
`#E74C3C, #3498DB, #2ECC71, #F39C12, #9B59B6, #1ABC9C`

Task label colors:
- `pick_and_place` → `#3498DB`
- `lift_and_place` → `#E67E22`
- `move_rack` → `#9B59B6`
- `idle` → `#95A5A6`

RULA badge colors:
- LOW (1–2) → `#2ECC71`
- MEDIUM (3–4) → `#F1C40F`
- HIGH (5–6) → `#E67E22`
- VERY_HIGH (7) → `#E74C3C`

---

## Never Do

- Never use `input()` or any interactive prompt in code
- Never hardcode file paths — always use config or CLI args
- Never commit model weights to the repo — download at runtime
- Never generate placeholder functions with `pass` and move on — implement fully or flag explicitly with `# TODO:` and a reason
- Never use `print()` — use `logging`
- Never silently swallow exceptions — always log them

---

## When You Are Done

After Phase 7 is complete and pipeline.py runs end to end:

1. Verify all 5 output types are generated correctly
2. Verify the annotated video plays with all overlays visible
3. Verify CSV has one row per tracked person
4. Verify PDF opens and renders all charts
5. Write a clean `README.md` with: setup instructions, how to run, output description, and a note on design decisions
6. Report completion with a summary of what was built, any `# DECISION:` choices made, and any known limitations