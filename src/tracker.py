"""Video inference and multi-object tracking for surgical instruments."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_TRACKER = "bytetrack.yaml"


def track_video(
	model_path: str | Path,
	source_path: str | Path,
	output_path: str | Path,
	*,
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

	if not model_path.is_file():
		raise FileNotFoundError(f"Model not found: {model_path}")
	if not source_path.is_file():
		raise FileNotFoundError(f"Input video not found: {source_path}")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be between 0 and 1")
	if imgsz < 32:
		raise ValueError("imgsz must be at least 32")

	output_path.parent.mkdir(parents=True, exist_ok=True)
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
			annotated = _annotate_frame(results[0], previous_time, frame_count)
			previous_time = time.perf_counter()
			writer.write(annotated)
			frame_count += 1

			if display:
				cv2.imshow("Surgical Tool Tracking", annotated)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	finally:
		capture.release()
		writer.release()
		cv2.destroyAllWindows()

	return output_path


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
