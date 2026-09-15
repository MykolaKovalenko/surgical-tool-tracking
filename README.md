# Surgical Tool Tracking

**Real-time-ready detection and multi-object tracking of laparoscopic instruments.** This project combines a custom-trained YOLO detector with ByteTrack to answer a practical question in computer-assisted surgery: *which instruments are visible, where are they, and which detections belong to the same instrument over time?*

![Annotated tracking demo](experiments/docs/assets/tracking-demo.jpg)

![YOLO + ByteTrack demo](experiments/docs/assets/test_video2_demo.gif)

This frame shows two tracked instruments with class labels, confidence scores,
persistent IDs, and the measured display FPS. The full demo video is produced
locally because the surgical video assets are too large for the repository.

## Results at a glance

| Area | Current result | Protocol |
|---|---:|---|
| Detection | `mAP50: 0.9298` | v2 Colab validation split |
| Detection | `mAP50-95: 0.5501` | v2 Colab validation split |
| Detection | `P: 0.9371`, `R: 0.8912` | v2 Colab validation split |
| Tracking | Not yet measured with ground truth | Annotated sequence required |
| Runtime | `10.76 FPS`, `90.98/96.36 ms p50/p95` | Local CPU, 640 px |

The detection values are validation results from the 100-epoch v2 Colab run;
they are not a final independent test result. The v2 result is compared with
the 50-epoch v1 baseline in `experiments/reports/training_yolo11s_v2/summary.md`. The
current local CPU benchmark is not presented as a real-time result: 10.76 FPS
is still below the 30 FPS source rate.

### Measured `test_video2` run

| Metric | Value |
|---|---:|
| Source duration | 25.0 s |
| Source FPS | 30.0 |
| Processed frames | 750 |
| Detection coverage | 99.33% |
| Mean detections/frame | 1.5053 |
| Mean confidence | 0.6975 |
| Unique track IDs | 37 |
| Longest track | 293 frames |
| Inference FPS, CPU | 10.18 |
| Latency p50 / p95 | 94.33 / 110.32 ms |

This is a runtime and behavior report, not tracking accuracy. `test_video2` has
no official tracking labels, so precision, recall, MOTA, IDF1, and HOTA are not
reported for it. The full report is saved under `outputs/test_video2_report/`.

## Why this project matters

The project is deliberately small and measurable: one inference entry point, one diagnostic evaluator, and one utility for creating reproducible video clips. It demonstrates machine learning, geometry in image coordinates, temporal association, experimental protocol, and deployment trade-offs rather than only showing a model prediction.

## Demo

The main demonstration should be a 15-30 second video showing:

1. several tools entering and leaving the image;
2. bounding boxes, classes, confidence, and persistent track IDs;
3. an occlusion or crossing case;
4. the measured processing FPS and source FPS;
5. one failure case, kept visible and explained.

Keep one short annotated video or GIF under `experiments/docs/assets/`. A 10-second
GIF preview is already generated from `test_video2`:

```markdown
![YOLO + ByteTrack demo](experiments/docs/assets/test_video2_demo.gif)
```

The full local result is `outputs/test_video2_tracked.mp4`. Publish it only if the
source video is legally redistributable; otherwise publish the GIF only when its
use is permitted, or link to the original source instead.

## What this demonstrates

- object detection with confidence filtering and seven surgical-tool classes;
- persistent multi-object identities with ByteTrack;
- reproducible video inference and annotated output;
- performance measurement instead of an unverified "real-time" claim;
- a clear boundary between diagnostic proxies and scientific accuracy metrics.

## Project structure

```text
main.py                # only runtime entry point
src/tracker.py         # YOLO + ByteTrack implementation
models/best.pt         # local model weights
data/                  # local input videos and samples
outputs/               # current generated result only
experiments/           # optional preparation, evaluation, and training tools
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

The current local model is YOLO11s v2 trained on Cholec80. Its validation and
comparison summary is recorded in `reports/training_yolo11s_v2/summary.md`.

## Prepared local video data

The extracted image sequences are also assembled locally as plain 10 FPS videos
for repeatable experiments. No tracking or model output is embedded in these files:

```text
data/
  video41/
    video41_10fps.mp4
    video41_tracking_gt.csv
  video42/
    video42_10fps.mp4
    video42_tracking_gt.csv
  video43/
    video43_10fps.mp4
    video43_tracking_gt.csv
  video44/
    video44_10fps.mp4
    video44_tracking_gt.csv
  video45/
    video45_10fps.mp4
    video45_tracking_gt.csv
```

The CSV files contain the project-specific reference boxes and reconstructed
track IDs for the corresponding frame sequence. They are targets for evaluation,
not official tracking ground truth. Use a prepared video as input to the model,
then export predictions to a separate output folder:

```powershell
python main.py `
  --model models/best.pt `
  --source data/video41/video41_10fps.mp4 `
  --output outputs/video41_tracked.mp4 `
  --device cpu
```

For cutting experiments, keep the original frame sequence or cut the prepared
10 FPS MP4 consistently. The CSV frame names refer to the original PNG names.

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

## Optional experiments

The simple public workflow is only video -> YOLO -> ByteTrack. Tools beyond that
path are kept in `experiments/tools/` so they do not obscure the main project.
Use them only when preparing a benchmark or training dataset.

```text
experiments/tools/cut_video.py
experiments/tools/evaluate_tracking.py
experiments/tools/evaluate_tracking_reference.py
experiments/tools/prepare_yolo_dataset.py
experiments/tools/reconstruct_tracking_gt.py
experiments/configs/
experiments/docs/
experiments/reports/
experiments/notebooks/
experiments/archives/
```

The reference CSVs are project-specific targets reconstructed from frame
annotations, not official tracking ground truth. Keep the source frame order when
cutting clips; the current CSVs use the original PNG frame names.

## Optional runtime measurement

```powershell
python experiments/tools/evaluate_tracking.py `
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

## Presentation guidance

For detection, report mAP@50, mAP@50:95, precision, recall, and per-class
results on a held-out test split. The existing archive values are not enough
without the dataset split and evaluation protocol.

Use a real, legally distributable video for the main qualitative demo. Without
official temporal identity labels, do not present reconstructed tracking metrics
as clinical or official tracking accuracy.

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
