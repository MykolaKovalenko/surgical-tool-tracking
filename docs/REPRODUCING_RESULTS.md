# Reproducing the results

This document separates the three experiments in the project: detector
training, video inference, and temporal tracking diagnostics.

## 1. Detector training

The current baseline was trained in Google Colab with:

| Parameter | Value |
|---|---|
| Model | YOLO11s pretrained |
| Image size | 640 px |
| Epochs | 50 |
| Batch size | 16 |
| Seed | 42 |
| Validation split | `val` |

The exported `best.pt` belongs in `models/best.pt`. Do not compare a new model
using a different split or image size without recording that change.

## 2. Offline video inference

```powershell
venv\Scripts\python.exe main.py `
  --model models\best.pt `
  --source data\test_video.mp4 `
  --output outputs\tracked_video.mp4 `
  --confidence 0.25 `
  --imgsz 640 `
  --device cpu
```

The output video contains detections, classes, ByteTrack IDs, an active-tool
count, and an instantaneous display FPS. The display FPS is useful for a demo
but is not the official benchmark.

## 3. Performance and tracking diagnostics

```powershell
venv\Scripts\python.exe src\evaluate_tracking.py `
  --model models\best.pt `
  --source data\test_video2.mp4 `
  --report-dir reports\tracking_video2 `
  --confidence 0.25 `
  --imgsz 640 `
  --device cpu
```

Use `summary.md` for a compact report and `summary.json` or the CSV files for
further analysis. The official runtime result must include hardware, input
resolution, model, confidence threshold, and whether the measurement includes
video decoding, annotation, and writing.

## What is and is not measured

The detector metrics in the repository are validation metrics from the Colab
run. The current tracking report contains behavioral proxies such as track
duration and ID gaps. HOTA, IDF1, MOTA, ID switches, and fragmentation require
manually annotated ground-truth identities and are not yet available.