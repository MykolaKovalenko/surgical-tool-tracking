#!/usr/bin/env python3
"""Compare reconstructed tracking references against model predictions.

This script is designed for the workflow used in this project:
- the GT/reference directory contains per-video CSVs built from annotation boxes,
- the prediction directory contains per-video tracking outputs from a model,
- the script computes frame-level matching, precision/recall, MOTA, and a simple
  ID-switch proxy for each video.

Important note:
    The files under the reference directory are used as targets for evaluation.
    They are not official clinical ground truth; they are project-specific
    reference tracks that make the benchmark reproducible and comparable.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate target/reference tracking outputs against model predictions."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional video ID or basename to evaluate, for example 'video_41' or '41'.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory containing the reference tracking CSVs (one per video).",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing the predicted tracking CSVs (one per video).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the summary CSV. Defaults to gt_dir/metrics_summary.csv.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to match predicted boxes to reference boxes.",
    )
    return parser.parse_args()


def bbox_to_xyxy(row: dict[str, Any]) -> tuple[float, float, float, float]:
    x1 = float(row["bbox_x"])
    y1 = float(row["bbox_y"])
    x2 = x1 + float(row["bbox_width"])
    y2 = y1 + float(row["bbox_height"])
    return (x1, y1, x2, y2)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def compute_video_metrics(gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, float | int | str]:
    gt_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pred_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in gt_rows:
        gt_by_frame[str(row.get("frame_index", row["frame_name"]))].append(row)
    for row in pred_rows:
        pred_by_frame[str(row.get("frame_index", row["frame_name"]))].append(row)

    gt_total = 0
    pred_total = 0
    tp = 0
    fp = 0
    fn = 0
    id_switches = 0
    mean_iou = 0.0
    iou_sum = 0.0

    last_pred_by_gt_track: dict[int, int] = {}

    for frame_name in sorted(set(gt_by_frame) | set(pred_by_frame), key=lambda s: s):
        gt_objs = gt_by_frame.get(frame_name, [])
        pred_objs = pred_by_frame.get(frame_name, [])
        gt_total += len(gt_objs)
        pred_total += len(pred_objs)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for gi, gt_obj in enumerate(gt_objs):
            gbox = bbox_to_xyxy(gt_obj)
            for pi, pred_obj in enumerate(pred_objs):
                if pi in matched_pred:
                    continue
                pbox = bbox_to_xyxy(pred_obj)
                score = iou(gbox, pbox)
                if score >= iou_threshold:
                    candidates.append((score, gi, pi))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            tp += 1
            iou_sum += score
            gt_track = int(gt_objs[gi]["track_id"])
            pred_track = int(pred_objs[pi]["track_id"])
            prev_pred = last_pred_by_gt_track.get(gt_track)
            if prev_pred is not None and prev_pred != pred_track:
                id_switches += 1
            last_pred_by_gt_track[gt_track] = pred_track

        fn += len(gt_objs) - len([1 for gi in range(len(gt_objs)) if gi in matched_gt])
        fp += len(pred_objs) - len([1 for pi in range(len(pred_objs)) if pi in matched_pred])

    if tp:
        mean_iou = iou_sum / tp
    else:
        mean_iou = 0.0

    if gt_total == 0:
        mota = 1.0
    else:
        mota = 1.0 - (fp + fn + id_switches) / gt_total
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "video": "",
        "gt_total_objects": gt_total,
        "pred_total_objects": pred_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mota": mota,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "id_switches": id_switches,
        "mean_iou": mean_iou,
    }


def load_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        for key in ["surgery_num", "frame_index", "track_id", "object_index"]:
            if key in row:
                row[key] = int(float(row[key]))
        for key in ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]:
            if key in row:
                row[key] = float(row[key])
    return rows


def file_stem_for_video(path: Path) -> str:
    name = path.name
    if name.endswith("_tracking_gt.csv"):
        return name.replace("_tracking_gt.csv", "")
    if name.endswith("_pred.csv"):
        return name.replace("_pred.csv", "")
    if name.endswith(".csv"):
        return name[:-4]
    return name


def normalize_video_id(video_value: str | None) -> str | None:
    if video_value is None:
        return None
    value = video_value.strip().lower()
    if value.startswith("video_"):
        return value
    if value.isdigit():
        return f"video_{int(value):02d}"
    if value.startswith("video"):
        return value
    return value


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n=== Video benchmark summary ===")
    print(f"Video: {metrics['video']}")
    print(f"GT objects: {metrics['gt_total_objects']}")
    print(f"Predicted objects: {metrics['pred_total_objects']}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"MOTA: {metrics['mota']:.4f}")
    print(f"ID switches: {metrics['id_switches']}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("==============================\n")


def main() -> None:
    args = parse_args()
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    target_video = normalize_video_id(args.video)

    if not gt_dir.exists() or not pred_dir.exists():
        raise FileNotFoundError("GT directory and prediction directory must exist.")

    gt_files = sorted(gt_dir.glob("*.csv"))
    pred_files = sorted(pred_dir.glob("*.csv"))
    if not gt_files:
        raise FileNotFoundError(f"No CSV files found in {gt_dir}")
    if not pred_files:
        raise FileNotFoundError(f"No CSV files found in {pred_dir}")

    pred_map = {file_stem_for_video(path): path for path in pred_files}

    summary_rows: list[dict[str, Any]] = []
    for gt_path in gt_files:
        video_id = file_stem_for_video(gt_path)
        if target_video is not None and normalize_video_id(video_id) != target_video:
            continue
        pred_path = pred_map.get(video_id)
        if pred_path is None:
            print(f"Skipping {video_id}: no prediction file found")
            continue

        gt_rows = load_csv_rows(gt_path)
        pred_rows = load_csv_rows(pred_path)
        metrics = compute_video_metrics(gt_rows, pred_rows, args.iou_threshold)
        metrics["video"] = video_id
        summary_rows.append(metrics)
        print_summary(metrics)

    if not summary_rows:
        raise FileNotFoundError(
            f"No benchmark rows found for video filter '{args.video}'. "
            "Check the video ID or ensure the prediction file exists."
        )

    output_path = args.output or (gt_dir / "tracking_metrics_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "gt_total_objects",
        "pred_total_objects",
        "tp",
        "fp",
        "fn",
        "mota",
        "precision",
        "recall",
        "f1",
        "id_switches",
        "mean_iou",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in summary_rows)

    print(f"Wrote {len(summary_rows)} video metrics rows to {output_path}")
    if len(summary_rows) == 1:
        row = summary_rows[0]
        print(
            f"Quick readout: {row['video']} | "
            f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | MOTA={row['mota']:.4f} | ID switches={row['id_switches']}"
        )


if __name__ == "__main__":
    main()
