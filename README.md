# Surgical Tool Tracking

**Real-time-ready detection and multi-object tracking of laparoscopic instruments.** This project combines a custom-trained YOLO detector with ByteTrack to answer a practical question in computer-assisted surgery: *which instruments are visible, where are they, and which detections belong to the same instrument over time?*

![Annotated tracking demo](docs/assets/tracking-demo.jpg)

This frame shows two tracked instruments with class labels, confidence scores,
persistent IDs, and the measured display FPS. The full demo video is generated
locally because the surgical video assets are too large for the repository.

## Results at a glance

| Area | Current result | Protocol |
|---|---:|---|
| Detection | `mAP50: 0.9257` | Colab validation split |
| Detection | `mAP50-95: 0.5517` | Colab validation split |
| Detection | `P: 0.9092`, `R: 0.8844` | Colab validation split |
| Tracking | Not yet measured with ground truth | Annotated sequence required |
| Runtime | `4.99 FPS`, `182.86/378.18 ms p50/p95` | Local CPU, 640 px |

The detection values are validation results from the 50-epoch Colab run; they
are not a final independent test result. The best mAP50-95 was reached at
epoch 49. The current local CPU benchmark is kept in the generated report and
is not presented as a real-time result.

## Why this project matters

The project is deliberately small and measurable: one inference entry point, one diagnostic evaluator, and one utility for creating reproducible video clips. It demonstrates machine learning, geometry in image coordinates, temporal association, experimental protocol, and deployment trade-offs rather than only showing a model prediction.

## Demo

The main demonstration should be a 15-30 second video showing:

1. several tools entering and leaving the image;
2. bounding boxes, classes, confidence, and persistent track IDs;
3. an occlusion or crossing case;
4. the measured processing FPS and source FPS;
5. one failure case, kept visible and explained.

Keep one annotated video or GIF in the repository, preferably under `docs/assets/`, and link to the full-resolution MP4 when it is too large for GitHub. A short demo with a visible limitation is more credible than a long video with only successful frames.

## What this demonstrates

- object detection with confidence filtering and seven surgical-tool classes;
- persistent multi-object identities with ByteTrack;
- reproducible video inference and annotated output;
- performance measurement instead of an unverified "real-time" claim;
- a clear boundary between diagnostic proxies and scientific accuracy metrics.

## Project structure

```text
main.py                 # inference and optional live preview
src/tracker.py          # YOLO + ByteTrack video pipeline
src/evaluate_tracking.py# latency, throughput and tracking diagnostics
src/cut_video.py        # reproducible frame-accurate clip creation
models/best.pt          # local weights, intentionally ignored by Git
data/samples/           # local source videos, intentionally ignored by Git
reports/                # generated reports, useful locally but not source code
```

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the virtual-environment interpreter explicitly in VS Code if the terminal
does not activate it: `venv\Scripts\python.exe`.

## Model and data assets

The trained weights and surgical videos are intentionally excluded from Git
because they are large and may have dataset-specific distribution terms. Put
the exported `best.pt` at `models/best.pt` and a local clip at
`data/test_video.mp4`. For a public repository, provide a documented release
link for the weights and a short legally redistributable demo clip or GIF.

The current local model is YOLO11s trained on Cholec80. Its validation summary
is recorded in `reports/training_yolo11s_v1/summary.md`.

## Run inference

```powershell
python main.py `
  --model models/best.pt `
  --source data/test_video.mp4 `
  --output outputs/tracked_video.mp4 `
  --confidence 0.25 `
  --imgsz 640 `
  --device 0
```

Add `--display` for an OpenCV preview. `--device 0` selects the first CUDA
GPU; omit it to let Ultralytics choose. Lowering `--imgsz` can increase
throughput, but every speed/accuracy trade-off must be benchmarked.

## Measure the system

```powershell
python src/evaluate_tracking.py `
  --model models/best.pt `
  --source data/test_video2.mp4 `
  --report-dir reports/tracking_video2 `
  --imgsz 640 `
  --device 0
```

The report contains `summary.md`, `summary.json`, `frame_metrics.csv`, and
`track_metrics.csv`. It measures source FPS, inference FPS, p50/p95 inference
latency, detection coverage, confidence, track duration, ID reappearance gaps,
and a boolean verdict for reaching the source FPS.

The current local CPU report is about 5-6 inference FPS on a 30 FPS video. The
annotated MP4 can still be encoded at 30 FPS, but that does not mean the
system is real-time. A credible real-time claim must state hardware, input
resolution, model version, batch size, and whether capture, inference,
annotation, and display are included.

## Metrics to present

For detection, report mAP@50, mAP@50:95, precision, recall, and per-class
results on a held-out test split. The existing archive values are not enough
without the dataset split and evaluation protocol.

For tracking, annotate a short representative sequence with ground-truth
boxes and identities, then report HOTA, IDF1, MOTA, ID switches, and track
fragmentation. Detection coverage, confidence, longest track, and ID gaps are
useful diagnostics, but they are not tracking accuracy.

For deployment, report mean/p50/p95 latency, end-to-end FPS, source FPS,
hardware, resolution, and memory. The target is usually at least 25 FPS for a
low-latency live display, but the correct target depends on the clinical use
case; offline analysis has no 25 FPS requirement.

## Recommended next milestones

1. Add a small, documented ground-truth test split with identity labels.
2. Benchmark CPU, CUDA, and smaller input sizes using the same clip and report
	the result in a table.
3. Add a short failure analysis: occlusion, glare, tool overlap, blur, and
	class imbalance.
4. Export one short demo video and one compact report that can be understood
	without opening the source code.

This is a research/engineering portfolio project, not a clinical device. No
clinical performance or safety claim should be inferred from these videos.
