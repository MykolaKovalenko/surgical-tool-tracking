#!/usr/bin/env python3
"""Reconstruct pseudo-ground-truth tracking IDs from per-frame detection labels.

This script reads the per-frame bounding box CSV exported by the annotation
workflow and rebuilds stable track IDs across consecutive frames. It is useful
when the dataset contains only detection labels (frame + bbox + class) and not a
pre-made MOT/TrackID file.

Important caveat:
    This is a practical reconstruction, not an official tracking ground truth.
    It is designed to create a consistent benchmark-friendly representation from
    existing annotations, so that the project can compare a model's track output
    with a stable reference.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct pseudo track IDs from per-frame surgical-tool detections."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to the ROI_Labels.csv file. Defaults to the first matching CSV in the project.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root folder containing Images and ROI_Labels.csv. If omitted, script auto-detects it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for per-video reconstructed tracking GT CSVs. Defaults to outputs/tracking_gt_by_video.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.10,
        help="Minimum IoU required to link an object across frames.",
    )
    return parser.parse_args()


def auto_detect_dataset_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd() / "vieos41-45",
        Path.cwd() / "videos41-45",
        Path.cwd() / "videos_41_45",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if (candidate / "ROI_Labels.csv").exists():
            return candidate
        if (candidate / "Images").exists() and (candidate / "ROI_Labels.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the annotation dataset. Provide --dataset-root or --labels."
    )


def auto_detect_labels_path(dataset_root: Path | None) -> Path:
    if dataset_root is None:
        dataset_root = auto_detect_dataset_root()
    labels = dataset_root / "ROI_Labels.csv"
    if not labels.exists():
        raise FileNotFoundError(f"No ROI_Labels.csv at {labels}")
    return labels


def frame_number_from_name(frame_name: str) -> int:
    stem = Path(frame_name).stem
    suffix = stem.split("_")[-1]
    if suffix.isdigit():
        return int(suffix)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def bbox_iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def load_rows(labels_path: Path) -> list[dict[str, Any]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
    for row in rows:
        row["Surgery_num"] = int(row["Surgery_num"])
        row["NumBBox_inFrame"] = int(row["NumBBox_inFrame"])
        row["BBox_X"] = float(row["BBox_X"])
        row["BBox_Y"] = float(row["BBox_Y"])
        row["BBox_Width"] = float(row["BBox_Width"])
        row["BBox_Height"] = float(row["BBox_Height"])
    return rows


def reconstruct_tracks(rows: list[dict[str, Any]], min_iou: float = 0.10) -> dict[int, list[dict[str, Any]]]:
    """Build one tracking-style CSV per surgery sequence by matching boxes across frames."""
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["Surgery_num"]].append(row)

    output_by_video: dict[int, list[dict[str, Any]]] = {}

    for surgery_num in sorted(by_video):
        video_rows = by_video[surgery_num]
        frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in video_rows:
            frames[row["FrameName"]].append(row)

        output_rows: list[dict[str, Any]] = []
        next_track_id = 1
        prev_boxes_by_track: dict[int, dict[str, float]] = {}
        prev_obj_index_by_track: dict[int, int] = {}

        for frame_name in sorted(frames.keys(), key=frame_number_from_name):
            frame_rows = frames[frame_name]
            current_items = []
            for row in frame_rows:
                current_items.append(
                    {
                        "bbox": {
                            "x1": row["BBox_X"],
                            "y1": row["BBox_Y"],
                            "x2": row["BBox_X"] + row["BBox_Width"],
                            "y2": row["BBox_Y"] + row["BBox_Height"],
                        },
                        "tool_name": row["ToolName"],
                        "row": row,
                    }
                )

            assigned_current_idx: set[int] = set()
            assigned_prev_ids: set[int] = set()
            matches: list[tuple[float, int, int]] = []

            for prev_track_id, prev_box in prev_boxes_by_track.items():
                for idx, item in enumerate(current_items):
                    if idx in assigned_current_idx:
                        continue
                    score = bbox_iou(prev_box, item["bbox"])
                    if score >= min_iou:
                        matches.append((score, prev_track_id, idx))

            matches.sort(key=lambda item: item[0], reverse=True)
            for _, prev_track_id, idx in matches:
                if prev_track_id in assigned_prev_ids or idx in assigned_current_idx:
                    continue
                assigned_prev_ids.add(prev_track_id)
                assigned_current_idx.add(idx)

                item = current_items[idx]
                row = item["row"]
                output_rows.append(
                    {
                        "surgery_num": surgery_num,
                        "frame_name": frame_name,
                        "frame_index": frame_number_from_name(frame_name),
                        "track_id": prev_track_id,
                        "object_index": prev_obj_index_by_track.get(prev_track_id, idx + 1),
                        "tool_name": row["ToolName"],
                        "bbox_x": row["BBox_X"],
                        "bbox_y": row["BBox_Y"],
                        "bbox_width": row["BBox_Width"],
                        "bbox_height": row["BBox_Height"],
                    }
                )
                prev_boxes_by_track[prev_track_id] = item["bbox"]
                prev_obj_index_by_track[prev_track_id] = idx + 1

            for idx, item in enumerate(current_items):
                if idx in assigned_current_idx:
                    continue
                track_id = next_track_id
                next_track_id += 1
                row = item["row"]
                output_rows.append(
                    {
                        "surgery_num": surgery_num,
                        "frame_name": frame_name,
                        "frame_index": frame_number_from_name(frame_name),
                        "track_id": track_id,
                        "object_index": idx + 1,
                        "tool_name": row["ToolName"],
                        "bbox_x": row["BBox_X"],
                        "bbox_y": row["BBox_Y"],
                        "bbox_width": row["BBox_Width"],
                        "bbox_height": row["BBox_Height"],
                    }
                )
                prev_boxes_by_track[track_id] = item["bbox"]
                prev_obj_index_by_track[track_id] = idx + 1

        output_by_video[surgery_num] = output_rows

    return output_by_video


def write_output_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surgery_num",
        "frame_name",
        "frame_index",
        "track_id",
        "object_index",
        "tool_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    labels_path = args.labels or auto_detect_labels_path(dataset_root)
    if dataset_root is None:
        dataset_root = labels_path.parent

    output_dir = args.output or dataset_root / "outputs" / "tracking_gt_by_video"
    if output_dir.suffix.lower() == ".csv":
        output_dir = output_dir.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(labels_path)
    gt_by_video = reconstruct_tracks(rows, min_iou=args.min_iou)

    total_rows = 0
    total_tracks = 0
    total_frames = 0
    for surgery_num in sorted(gt_by_video):
        video_rows = gt_by_video[surgery_num]
        output_path = output_dir / f"video_{surgery_num}_tracking_gt.csv"
        write_output_csv(video_rows, output_path)
        total_rows += len(video_rows)
        total_tracks += len({row["track_id"] for row in video_rows})
        total_frames += len({row["frame_name"] for row in video_rows})
        print(f"Video {surgery_num}: {len(video_rows)} rows -> {output_path}")

    print(f"Loaded labels: {len(rows)} detection rows")
    print(f"Reconstructed tracks across all videos: {total_tracks}")
    print(f"Frames covered across all videos: {total_frames}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
