"""Generate practical detection and tracking diagnostics for a video."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def evaluate_video(
    model_path: str | Path,
    source_path: str | Path,
    report_dir: str | Path,
    *,
    confidence: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
    sample_every: int = 1,
) -> dict[str, Any]:
    """Run tracking and write frame-level and aggregate diagnostics.

    The report contains proxy tracking measurements. MOTA, IDF1, and HOTA
    require ground-truth trajectories and are intentionally not estimated.
    """
    model_path = Path(model_path)
    source_path = Path(source_path)
    report_dir = Path(report_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")

    report_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = report_dir / "frame_metrics.csv"
    track_csv = report_dir / "track_metrics.csv"
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(str(model_path))
    class_counts: Counter[str] = Counter()
    track_frames: Counter[int] = Counter()
    track_classes: defaultdict[int, Counter[str]] = defaultdict(Counter)
    track_last_frame: dict[int, int] = {}
    track_gaps: list[int] = []
    confidence_sum = 0.0
    detection_count = 0
    frames_with_detections = 0
    processed_frames = 0
    inference_seconds = 0.0
    inference_latencies: list[float] = []
    rows: list[dict[str, Any]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index = processed_frames
            processed_frames += 1
            if frame_index % sample_every != 0:
                continue

            started = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=confidence,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]
            inference_elapsed = time.perf_counter() - started
            inference_seconds += inference_elapsed
            inference_latencies.append(inference_elapsed)
            boxes = result.boxes
            detections = 0 if boxes is None else len(boxes)
            frames_with_detections += int(detections > 0)
            frame_confidences: list[float] = []
            frame_ids: list[int] = []

            if boxes is not None and detections:
                confidences = boxes.conf.cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [-1] * detections
                )
                for confidence_value, class_id, track_id in zip(
                    confidences, classes, ids
                ):
                    label = str(model.names[int(class_id)])
                    frame_confidences.append(float(confidence_value))
                    frame_ids.append(int(track_id))
                    class_counts[label] += 1
                    confidence_sum += float(confidence_value)
                    detection_count += 1
                    if track_id >= 0:
                        track_frames[track_id] += 1
                        track_classes[track_id][label] += 1
                        if track_id in track_last_frame:
                            gap = frame_index - track_last_frame[track_id] - sample_every
                            if gap > 0:
                                track_gaps.append(gap)
                        track_last_frame[track_id] = frame_index

            rows.append(
                {
                    "frame": frame_index,
                    "time_seconds": round(frame_index / source_fps, 3)
                    if source_fps
                    else None,
                    "detections": detections,
                    "track_ids": ";".join(str(track_id) for track_id in frame_ids),
                    "mean_confidence": round(
                        sum(frame_confidences) / len(frame_confidences), 5
                    )
                    if frame_confidences
                    else 0.0,
                }
            )
    finally:
        capture.release()

    with frame_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    track_rows = []
    for track_id, frame_total in sorted(track_frames.items()):
        class_counter = track_classes[track_id]
        dominant_class, dominant_count = class_counter.most_common(1)[0]
        track_rows.append(
            {
                "track_id": track_id,
                "frames": frame_total,
                "duration_seconds": round(
                    frame_total * sample_every / source_fps, 3
                )
                if source_fps
                else None,
                "dominant_class": dominant_class,
                "class_changes_proxy": sum(class_counter.values()) - dominant_count,
            }
        )
    with track_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "track_id",
            "frames",
            "duration_seconds",
            "dominant_class",
            "class_changes_proxy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(track_rows)

    latency_ms = sorted(value * 1000 for value in inference_latencies)
    latency_p50 = latency_ms[len(latency_ms) // 2] if latency_ms else 0.0
    latency_p95 = latency_ms[min(int(len(latency_ms) * 0.95), len(latency_ms) - 1)] if latency_ms else 0.0
    unique_tracks = len(track_frames)
    summary: dict[str, Any] = {
        "video": str(source_path),
        "model": str(model_path),
        "tracker": tracker,
        "confidence_threshold": confidence,
        "imgsz": imgsz,
        "device": device or "auto",
        "source_fps": round(source_fps, 3),
        "source_frames_metadata": source_frames,
        "processed_frames": processed_frames,
        "duration_seconds": round(processed_frames / source_fps, 3)
        if source_fps
        else None,
        "frames_with_detections": frames_with_detections,
        "detection_coverage": round(
            frames_with_detections / len(rows), 4
        )
        if rows
        else 0.0,
        "total_detections": detection_count,
        "mean_detections_per_frame": round(detection_count / len(rows), 4)
        if rows
        else 0.0,
        "mean_detection_confidence": round(
            confidence_sum / detection_count, 4
        )
        if detection_count
        else 0.0,
        "unique_track_ids": unique_tracks,
        "longest_track_frames": max(track_frames.values(), default=0),
        "tracks_seen_once": sum(value == 1 for value in track_frames.values()),
        "reappearance_gaps": len(track_gaps),
        "largest_reappearance_gap_frames": max(track_gaps, default=0),
        "class_detection_counts": dict(class_counts),
        "inference_fps_sampled": round(
            len(rows) / inference_seconds, 2
        )
        if inference_seconds
        else 0.0,
        "inference_latency_p50_ms": round(latency_p50, 2),
        "inference_latency_p95_ms": round(latency_p95, 2),
        "realtime_at_source_fps": bool(
            inference_seconds and len(rows) / inference_seconds >= source_fps
        ),
        "ground_truth_metrics": None,
        "note": "MOTA, IDF1, and HOTA require annotated ground-truth tracks.",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_markdown_summary(report_dir / "summary.md", summary)
    return summary


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise human-readable report."""
    classes = summary["class_detection_counts"] or {"none": 0}
    class_lines = "\n".join(
        f"| {label} | {count} |" for label, count in sorted(classes.items())
    )
    content = f"""# Tracking report

## Video

| Metric | Value |
|---|---:|
| Duration | {summary['duration_seconds']} s |
| Source FPS | {summary['source_fps']} |
| Processed frames | {summary['processed_frames']} |
| Detection coverage | {summary['detection_coverage']:.1%} |
| Mean detections/frame | {summary['mean_detections_per_frame']} |
| Mean confidence | {summary['mean_detection_confidence']} |
| Inference FPS | {summary['inference_fps_sampled']} |
| Inference latency p50 | {summary['inference_latency_p50_ms']} ms |
| Inference latency p95 | {summary['inference_latency_p95_ms']} ms |
| Realtime at source FPS | {summary['realtime_at_source_fps']} |

## Tracking proxies

| Metric | Value |
|---|---:|
| Unique track IDs | {summary['unique_track_ids']} |
| Longest track | {summary['longest_track_frames']} frames |
| Tracks seen once | {summary['tracks_seen_once']} |
| Reappearance gaps | {summary['reappearance_gaps']} |
| Largest gap | {summary['largest_reappearance_gap_frames']} frames |

## Classes

| Class | Detection count |
|---|---:|
{class_lines}

## Important limitation

MOTA, IDF1, and HOTA are not computed because this video has no ground-truth
trajectories. The values above measure system behavior, not tracking accuracy.
See `frame_metrics.csv` for frame-level details.
See `track_metrics.csv` for one row per observed track ID.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run the diagnostics CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/tracking"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    args = parser.parse_args()
    summary = evaluate_video(
        args.model,
        args.source,
        args.report_dir,
        confidence=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        sample_every=args.sample_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()