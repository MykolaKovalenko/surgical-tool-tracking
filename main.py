"""Command-line entry point for surgical tool tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tracker import track_image_sequence, track_video


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line argument parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
	parser.add_argument("--source", type=Path, required=True)
	parser.add_argument(
		"--output", type=Path, default=Path("outputs/tracked_video.mp4")
	)
	parser.add_argument(
		"--predictions", type=Path, default=None,
		help="CSV output for image-sequence predictions; generated automatically for folders.",
	)
	parser.add_argument(
		"--source-fps",
		type=float,
		default=10.0,
		help="Playback FPS for image sequences; use the extraction FPS, not inference FPS.",
	)
	parser.add_argument("--confidence", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", default=None, help="CPU, CUDA device (0), or auto")
	parser.add_argument(
		"--tracker",
		type=Path,
		default=Path("bytetrack.yaml"),
		help="Ultralytics tracker configuration, for example configs/bytetrack_video41_experiment.yaml",
	)
	parser.add_argument("--display", action="store_true")
	return parser


def main() -> None:
	"""Parse arguments and process the requested video."""
	args = build_parser().parse_args()
	if args.source.is_dir():
		prediction_path = args.predictions or (
			args.output.parent / "prediction_by_video" / f"{args.source.name}_pred.csv"
		)
		output_path = track_image_sequence(
			model_path=args.model,
			source_dir=args.source,
			output_path=args.output,
			prediction_path=prediction_path,
			confidence=args.confidence,
			imgsz=args.imgsz,
			device=args.device,
			tracker=str(args.tracker),
			source_fps=args.source_fps,
			display=args.display,
		)
		print(f"Image sequence tracked: {args.source}")
		print(f"Annotated video written to: {output_path}")
		print(f"Prediction CSV written to: {prediction_path}")
	else:
		output_path = track_video(
			model_path=args.model, source_path=args.source, output_path=args.output,
			prediction_path=args.predictions,
			confidence=args.confidence, imgsz=args.imgsz, device=args.device,
			tracker=str(args.tracker),
			display=args.display,
		)
		print(f"Tracking video written to: {output_path}")
		if args.predictions is not None:
			print(f"Prediction CSV written to: {args.predictions}")


if __name__ == "__main__":
	main()
