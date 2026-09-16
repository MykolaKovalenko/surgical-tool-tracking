"""Video inference and multi-object tracking for surgical instruments."""

from __future__ import annotations

import time
import csv
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


DEFAULT_TRACKER = "bytetrack.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def track_video(
	model_path: str | Path,
	source_path: str | Path,
	output_path: str | Path,
	*,
	prediction_path: str | Path | None = None,
	confidence: float = 0.25,
	imgsz: int = 640,
	device: str | None = None,
	tracker: str = DEFAULT_TRACKER,
	display: bool = False,
) -> Path:
	"""Track surgical instruments in a video and save an annotated copy.

	Parameters
	----------
	model_path:
		Path to Ultralytics YOLO weights.
	source_path:
		Path to the input video.
	output_path:
		Path where the annotated video will be written.
	confidence:
		Minimum detection confidence passed to the model.
		imgsz:
			Inference image size. Lower values can improve live throughput.
		device:
			Inference device, for example ``"cpu"`` or ``"0"`` for CUDA.
	tracker:
		Ultralytics tracker configuration, usually ``bytetrack.yaml``.
	display:
		Whether to show the annotated frames in an OpenCV window.

	Returns
	-------
	pathlib.Path
		The path of the generated video.

	Raises
	------
	FileNotFoundError
		If the model or input video does not exist.
	RuntimeError
		If OpenCV cannot open the input or output video.
	ValueError
		If the confidence value is outside the interval [0, 1].
	"""
	model_path = Path(model_path)
	source_path = Path(source_path)
	output_path = Path(output_path)
	prediction_path = Path(prediction_path) if prediction_path is not None else None

	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not source_path.is_file():
		raise FileNotFoundError(f"Input video not found: {source_path}")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be between 0 and 1")
	if imgsz < 32:
		raise ValueError("imgsz must be at least 32")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	if prediction_path is not None:
		prediction_path.parent.mkdir(parents=True, exist_ok=True)
	capture = cv2.VideoCapture(str(source_path))
	if not capture.isOpened():
		raise RuntimeError(f"Could not open input video: {source_path}")

	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = capture.get(cv2.CAP_PROP_FPS)
	fps = fps if fps > 0 else 30.0
	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)
	if not writer.isOpened():
		capture.release()
		raise RuntimeError(f"Could not create output video: {output_path}")

	model = YOLO(str(model_path))
	previous_time = time.perf_counter()
	frame_count = 0
	prediction_rows: list[dict[str, Any]] = []

	try:
		while True:
			success, frame = capture.read()
			if not success:
				break

			results = model.track(
				frame,
				persist=True,
				tracker=tracker,
				conf=confidence,
				imgsz=imgsz,
				device=device,
				verbose=False,
			)
			annotated = _annotate_frame(results[0], previous_time, frame_count + 1)
			previous_time = time.perf_counter()
			writer.write(annotated)
			if prediction_path is not None:
				prediction_rows.extend(
					_prediction_rows(
						results[0], f"frame_{frame_count + 1:06d}.png", frame_count + 1,
					)
				)
			frame_count += 1

			if display:
				cv2.imshow("Surgical Tool Tracking", annotated)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	finally:
		capture.release()
		writer.release()
		cv2.destroyAllWindows()

	if prediction_path is not None:
		_write_prediction_csv(prediction_path, prediction_rows)

	return output_path


def track_image_sequence(
	model_path: str | Path,
	source_dir: str | Path,
	output_path: str | Path,
	prediction_path: str | Path,
	*,
	confidence: float = 0.25,
	imgsz: int = 640,
	device: str | None = None,
	tracker: str = DEFAULT_TRACKER,
	display: bool = False,
	source_fps: float = 10.0,
) -> Path:
	"""Track an ordered image sequence and export video plus prediction CSV."""
	model_path = Path(model_path)
	source_dir = Path(source_dir)
	output_path = Path(output_path)
	prediction_path = Path(prediction_path)
	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not source_dir.is_dir():
		raise FileNotFoundError(f"Image sequence directory not found: {source_dir}")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be between 0 and 1")
	if imgsz < 32:
		raise ValueError("imgsz must be at least 32")

	image_paths = sorted(
		(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
		key=_frame_sort_key,
	)
	if not image_paths:
		raise FileNotFoundError(f"No image frames found in {source_dir}")

	first_frame = cv2.imread(str(image_paths[0]))
	if first_frame is None:
		raise RuntimeError(f"Could not read first frame: {image_paths[0]}")
	height, width = first_frame.shape[:2]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	prediction_path.parent.mkdir(parents=True, exist_ok=True)
	writer = cv2.VideoWriter(
		str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
	)
	if not writer.isOpened():
		raise RuntimeError(f"Could not create output video: {output_path}")

	fieldnames = [
		"surgery_num", "frame_name", "frame_index", "track_id", "object_index",
		"tool_name", "confidence", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
	]
	model = YOLO(str(model_path))
	prediction_rows: list[dict[str, Any]] = []
	previous_time = time.perf_counter()
	try:
		for frame_number, image_path in enumerate(image_paths, start=1):
			frame = cv2.imread(str(image_path))
			if frame is None:
				continue
			results = model.track(
				frame, persist=True, tracker=tracker, conf=confidence,
				imgsz=imgsz, device=device, verbose=False,
			)
			result = results[0]
			annotated = _annotate_frame(result, previous_time, frame_number)
			previous_time = time.perf_counter()
			writer.write(annotated)
			prediction_rows.extend(
				_prediction_rows(
					result, image_path.name, frame_number, source_dir=source_dir
				)
			)
			if display:
				cv2.imshow("Surgical Tool Tracking", annotated)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	finally:
		writer.release()
		cv2.destroyAllWindows()

	_write_prediction_csv(prediction_path, prediction_rows)

	return output_path


def _frame_sort_key(path: Path) -> tuple[int, str]:
	stem = path.stem
	digits = "".join(character for character in stem if character.isdigit())
	return (int(digits) if digits else 0, path.name)


def _prediction_rows(
	result,
	frame_name: str,
	frame_index: int,
	*,
	source_dir: Path | None = None,
	surgery_num: int | None = None,
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	boxes = result.boxes
	if boxes is None or len(boxes) == 0:
		return rows
	track_ids = boxes.id.int().tolist() if boxes.id is not None else []
	coordinates = boxes.xyxy.tolist()
	confidences = boxes.conf.tolist()
	classes = boxes.cls.int().tolist()
	if surgery_num is None:
		if source_dir is None:
			surgery_num = 0
		else:
			surgery_digits = "".join(character for character in source_dir.name if character.isdigit())
			surgery_num = int(surgery_digits) if surgery_digits else 0
	for object_index, (box, confidence, class_id) in enumerate(
		zip(coordinates, confidences, classes), start=1
	):
		track_id = track_ids[object_index - 1] if object_index <= len(track_ids) else object_index
		tool_name = result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else str(class_id)
		rows.append({
			"surgery_num": surgery_num,
			"frame_name": frame_name,
			"frame_index": frame_index,
			"track_id": int(track_id),
			"object_index": object_index,
			"tool_name": tool_name,
			"confidence": float(confidence),
			"bbox_x": float(box[0]),
			"bbox_y": float(box[1]),
			"bbox_width": float(box[2] - box[0]),
			"bbox_height": float(box[3] - box[1]),
		})
	return rows


def _write_prediction_csv(prediction_path: Path, rows: list[dict[str, Any]]) -> None:
	fieldnames = [
		"surgery_num", "frame_name", "frame_index", "track_id", "object_index",
		"tool_name", "confidence", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
	]
	with prediction_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _annotate_frame(result, previous_time: float, frame_count: int):
	"""Add tracking IDs, class labels, FPS, and active-tool count."""
	annotated = result.plot()
	now = time.perf_counter()
	elapsed = max(now - previous_time, 1e-6)
	fps = 1.0 / elapsed
	active_tools = len(result.boxes) if result.boxes is not None else 0
	text = f"FPS: {fps:5.1f} | Active tools: {active_tools} | Frame: {frame_count}"
	cv2.rectangle(annotated, (12, 12), (390, 48), (20, 28, 36), -1)
	cv2.putText(
		annotated,
		text,
		(22, 36),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(235, 245, 245),
		1,
		cv2.LINE_AA,
	)
	return annotated
