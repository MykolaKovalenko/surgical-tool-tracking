"""Prepare the videos 41-45 annotations as a YOLO detection dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = ["Grasper", "Hook", "Irrigator", "Bag", "Bipolar", "Clipper", "Scissors"]
SPLIT_VIDEOS = {"train": [41, 42, 43], "val": [44], "test": [45]}
IMAGE_ROOT_NAME = "Images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/samples/vieos41-45"),
        help="Directory containing Images/ and ROI_Labels.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/kaggle_yolo_videos41_45"),
        help="Directory for the prepared YOLO dataset.",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=Path("kaggle_yolo_videos41_45.zip"),
        help="ZIP archive ready to upload to Kaggle.",
    )
    return parser.parse_args()


def load_annotations(labels_path: Path) -> dict[tuple[int, str], list[dict[str, str]]]:
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[(int(row["Surgery_num"]), row["FrameName"])].append(row)
    return grouped


def yolo_line(row: dict[str, str], image_width: int, image_height: int, class_ids: dict[str, int]) -> str:
    x = float(row["BBox_X"])
    y = float(row["BBox_Y"])
    width = float(row["BBox_Width"])
    height = float(row["BBox_Height"])
    center_x = x + width / 2
    center_y = y + height / 2
    return (
        f"{class_ids[row['ToolName']]} "
        f"{center_x / image_width:.6f} "
        f"{center_y / image_height:.6f} "
        f"{width / image_width:.6f} "
        f"{height / image_height:.6f}"
    )


def write_yaml(output_root: Path) -> None:
    content = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def write_readme(output_root: Path, stats: Counter[str]) -> None:
    lines = [
        "# YOLO11 dataset: surgical videos 41-45",
        "",
        "This archive is prepared from the frame-level annotations in ROI_Labels.csv.",
        "It contains detection labels in YOLO format, not tracking identities.",
        "",
        "## Split protocol",
        "",
        "The split is by complete video to prevent near-duplicate neighboring frames from leaking across sets:",
        "- train: videos 41, 42, 43",
        "- val: video 44",
        "- test: video 45",
        "",
        "Only frames with at least one annotation are included. Frames without an annotation were not treated as confirmed background images.",
        "The original source FPS is unknown; FPS is irrelevant for this frame-based detector training dataset.",
        "",
        "## Kaggle training",
        "",
        "```python",
        "from ultralytics import YOLO",
        "model = YOLO('yolo11s.pt')",
        "model.train(data='/kaggle/input/<dataset-name>/data.yaml', imgsz=640, epochs=100, batch=-1)",
        "```",
        "",
        "## Contents",
        "",
        *[f"- {key}: {value} labeled images" for key, value in sorted(stats.items())],
        "",
        "Classes: " + ", ".join(CLASS_NAMES),
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(source_root: Path, output_root: Path) -> Counter[str]:
    labels = load_annotations(source_root / "ROI_Labels.csv")
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    if output_root.exists():
        shutil.rmtree(output_root)
    stats: Counter[str] = Counter()
    for split, video_numbers in SPLIT_VIDEOS.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for video_number in video_numbers:
            source_dir = source_root / IMAGE_ROOT_NAME / f"video_{video_number}"
            frame_keys = sorted(
                key for key in labels if key[0] == video_number
            )
            for _, frame_name in frame_keys:
                source_image = source_dir / frame_name
                if not source_image.is_file():
                    raise FileNotFoundError(f"Missing image for annotation: {source_image}")
                image = __import__("cv2").imread(str(source_image))
                if image is None:
                    raise RuntimeError(f"Could not read image: {source_image}")
                image_height, image_width = image.shape[:2]
                destination_stem = f"video_{video_number}_{Path(frame_name).stem}"
                shutil.copy2(source_image, image_dir / f"{destination_stem}.png")
                lines = [
                    yolo_line(row, image_width, image_height, class_ids)
                    for row in labels[(video_number, frame_name)]
                ]
                (label_dir / f"{destination_stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                stats[split] += 1
    write_yaml(output_root)
    write_readme(output_root, stats)
    return stats


def create_zip(output_root: Path, zip_output: Path) -> None:
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    if zip_output.exists():
        zip_output.unlink()
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(output_root.name) / path.relative_to(output_root))


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(args.source_root, args.output_root)
    create_zip(args.output_root, args.zip_output)
    print(f"Prepared dataset: {args.output_root}")
    print(f"ZIP archive: {args.zip_output}")
    print(f"Split counts: {dict(stats)}")


if __name__ == "__main__":
    main()
