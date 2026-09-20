# Ultrasound Video Frame Extraction & Mode Classification
 
A rule-based computer vision pipeline that converts raw ultrasound videos into a clean, mode-labeled image dataset — with **zero manual annotation**.
 
Built at **DEEPECHO** · Author: Ilyas Bejja · Supervisor: Mohammed Najid
 
---
 
## Overview
 
Ultrasound exams mix several distinct display modes in a single recording — B-mode (grayscale), color Doppler, pulsed-wave (PW) Doppler, measurement/caliper overlays, and occasional split-screen views. Sorting these frames by hand doesn't scale to large clinical datasets.
 
This pipeline automates the entire process: from raw `.mp4`/`.mkv` video files to a structured, six-class labeled image dataset plus a summary CSV of processing statistics — using deterministic, explainable image-processing rules rather than a trained model (since no labeled data exists yet to train one).
 
## Features
 
- **Adaptive frame extraction** via ffmpeg scene-change detection — no fixed sampling rate, no wasted storage on static frames
- **Automatic ROI detection** — locates and crops the active ultrasound display region without any manual bounding box
- **Six-class classification cascade** using HSV color-space thresholding:
  - `b-mode` — plain grayscale imaging (default class)
  - `doppler-mode` — color flow overlays (red/blue)
  - `measurement-mode` — yellow caliper/annotation overlays
  - `pw-doppler` — spectral velocity waveform (golden/orange)
  - `split_frame` — dual-view split screen
  - `non-usable` — blank/corrupted/menu frames
- **Fully interpretable** — every decision traces back to a pixel-ratio threshold, not a learned weight
- **Dockerized** — reproducible execution with no manual Python/ffmpeg setup
## Architecture
 
The pipeline runs as a three-stage process orchestrated by `model.py` and implemented by the `cls_frame` class in `main_class.py`:
 
```
1. Frame Extraction        2. ROI Detection            3. Mode Classification
   extract_vd_frames()  →     get_best_coords()      →    classify()
   ffmpeg scene-change        brightness + contour         HSV color-rule cascade
   sampling                   analysis
```
 
### Stage A — Frame Extraction
```bash
ffmpeg -i <video> -vf "select='gt(scene,0.02)'" -vsync vfr frame_%04d.png
```
Keeps only frames where ffmpeg's scene-change metric exceeds `0.02`, so static portions of the exam contribute few frames while probe movement or mode switches contribute proportionally more.
 
### Stage B — Automatic ROI Detection
1. `detect_brightest_frame()` scores every frame by center-to-border brightness ratio and picks the best-illuminated one as reference.
2. That frame is grayscaled, binarized (threshold ≥ 15), and cleaned up with a 30×30 morphological closing operation.
3. The largest contour's bounding box becomes the crop rectangle, reused for every frame in the video.
### Stage C — Classification Cascade
Every cropped frame is tested against five ordered HSV/intensity checks; the first match wins:
 
| Priority | Check | Rule | Saved as |
|---|---|---|---|
| 1 | `detect_measurement_mode()` | Yellow-pixel ratio ≥ 0.2% (H ≈ 32–35°) | `measurement-mode` |
| 2 | `detect_non_usable_frame()` | Dark-pixel ratio ≥ 95% (intensity < 10) | `non-usable` |
| 3 | `detect_split_frame()` | Vertical dark separator ratio ≥ 97% | `split_frame` |
| 4 | `detect_pw_doppler()` | Golden spectral pixels > 2000 (lower half) | `pw-doppler` |
| 5 | `detect_doppler_mode()` | Red/blue color-flow ratio over threshold | `doppler-mode` |
| 6 | *(default — no match)* | — | `b-mode` |
 
Rarer, more specific overlays are checked first so they aren't shadowed by the more generic B-mode/Doppler default paths.
 
## Output Structure
 
```
data/
├── input/
│   └── <exam>/<video>.mp4         # raw videos, one folder per exam
├── current/
│   └── <video>_frames/*.png       # frames extracted before classification
├── output/
│   └── <exam>/
│       ├── b-mode/
│       ├── doppler-mode/
│       ├── measurement-mode/
│       ├── pw-doppler/
│       ├── split_frame/
│       └── non-usable/
└── results.csv                    # one row per processed video
```
 
### `data/results.csv` schema
 
| Column | Meaning |
|---|---|
| `folder_name` | Exam/output sub-folder processed |
| `video_id` | Original video filename |
| `video_duration` | Video length in seconds (via ffprobe) |
| `n_extracted_frames` | Frames kept by the scene-change filter |
| `extraction_duration` | Time spent extracting frames |
| `getting_coord_duration` | Time spent on ROI detection |
| `classification_duration` | Time spent classifying all frames |
| `n_b-mode`, `n_doppler-mode`, `n_measurement-mode`, `n_pw-doppler_frame`, `n_split_frame`, `n_non-usable` | Per-class frame counts |
 
## Getting Started (Docker)
 
The pipeline is published on Docker Hub, so no local Python or ffmpeg install is required.
 
**1. Prepare your input folder:**
```
data/
└── input/
    ├── video1.mkv
    ├── video2.mkv
    ├── patient_study_01/
    │   └── video3.mp4
    └── patient_study_02/
        └── video4.mp4
```
Supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.mpg`, `.mpeg`. Loose videos dropped directly into `data/input/` are automatically grouped into their own sub-folder on first run.
 
**2. Run the container:**
```bash
docker run --rm \
  -v /path/to/your/local/data:/model/data \
  b0urb0n/ultrasound-classifier:3.0
```
 
The container scans `data/input/`, extracts keyframes, classifies them into the six output sub-folders, and writes `data/results.csv`. `data/current/`, `data/output/`, and `data/results.csv` are generated automatically — you only need to create and populate `data/input/`.
 
## Validation & Results
 
### ROI Detection Accuracy
Evaluated on 58 raw videos across multiple machine brands:
 
| Videos tested | Failures | Accuracy |
|---|---|---|
| 58 | 5 | **91.37%** (53/58) |
 
All 5 failures occurred on Voluson-family and Mindray machines with low UI/window contrast or bright static UI elements near the ROI border.
 
### Classification Cascade Accuracy
Manually verified frame-by-frame across four machine brands (766 frames total):
 
| Class | Correct / Total | Accuracy |
|---|---|---|
| B-mode | 624/624 | 100.00% |
| Non-usable | 35/37 | 94.59% |
| Doppler | 24/32 | 75.00% |
| PW-Doppler | 4/6 | 66.67% |
| Split-frame | 30/65 | 46.15% |
| Measurement-mode | 0/2 | 0.00% |
| **Overall** | **717/766** | **93.60%** |
 
By machine: Samsung (99.27%) and Mindray (98.30%) classify most reliably; Voluson is solid (92.31%); Hitachi is the outlier (83.48%), driven almost entirely by weak split-screen detection on its UI layout. Split-frame is the clearest weak point overall and the main target for threshold recalibration; PW-Doppler and measurement-mode results are based on very small samples and need re-verification on a larger batch.
 
### Performance
Aggregated across 19 fully-run videos:
 
| Metric | Value |
|---|---|
| Total raw video duration | 8,573.9 s (≈ 142.9 min) |
| Total frames extracted | 22,685 |
| Total pipeline time (extraction + ROI + classification) | 2,811.2 s (≈ 46.9 min) |
| Processing-to-video-duration ratio | 32.8% (≈ 3× faster than real time) |
| Mean classification cost | 0.053 s/frame |
| Mean ROI-detection cost | 0.028 s/frame |
 
Both ROI-detection and classification time scale almost perfectly linearly with the number of extracted frames (Pearson correlation ≈ 0.99), so the pipeline scales predictably with dataset size.
 
## Design Rationale: Why Rule-Based?
 
1. **No labeled data exists yet** — this pipeline is what creates the first labeled dataset, so a supervised model isn't an option at this stage.
2. **Vendor overlays are a strong, low-variance signal** — Doppler colors, caliper yellow, and PW-Doppler golden traces are rendered consistently by a given machine's rendering engine, so simple HSV thresholding is reliable without training data.
3. **Full traceability** — every decision maps to an explicit numeric threshold, which makes debugging and recalibration straightforward via the `detect_*_threshold()` helper methods.
HSV is used instead of BGR because it separates chromatic information (hue) from illumination (value), making color-overlay detection robust to brightness variation across machines and gain settings.
 
## Limitations & Future Work
 
- **Threshold sensitivity** — all six thresholds are hand-tuned and may need recalibration for different vendor color palettes or compression settings.
- **Mutually exclusive cascade** — a frame matching two overlays at once (e.g., a caliper on top of a Doppler frame) is always routed to the higher-priority class; multi-label output could be a future improvement.
- **Single-frame ROI** — the crop rectangle is computed once from the brightest frame; probe/window resizing mid-exam would invalidate it for the rest of the video.
- **Bootstrapping a learned model** — once enough auto-labeled frames accumulate, this output becomes ideal training data for a lightweight supervised classifier (e.g., a CNN) to handle edge cases the hand-crafted rules miss.
- **Planned ROI mitigations**: adaptive binarization (e.g., Otsu's method) instead of a fixed threshold, a resolution-relative morphological kernel size, and a per-vendor calibration profile selected automatically from video metadata.
## Repository
 
[github.com/ilyasbejja/ultrasound_exploration_model](https://github.com/ilyasbejja/ultrasound_exploration_model)
 
## License
 
Add your license here.
