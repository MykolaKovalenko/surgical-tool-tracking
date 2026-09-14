"""Command-line entry point for surgical tool tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tracker import track_video


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line argument parser."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
	parser.add_argument("--source", type=Path, required=True)
	parser.add_argument(
		"--output", type=Path, default=Path("outputs/tracked_video.mp4")
	)
	parser.add_argument("--confidence", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", default=None, help="CPU, CUDA device (0), or auto")
	parser.add_argument("--display", action="store_true")
	return parser


def main() -> None:
	"""Parse arguments and process the requested video."""
	args = build_parser().parse_args()
	output_path = track_video(
		model_path=args.model,
		source_path=args.source,
		output_path=args.output,
		confidence=args.confidence,
		imgsz=args.imgsz,
		device=args.device,
		display=args.display,
	)
	print(f"Tracking video written to: {output_path}")


if __name__ == "__main__":
	main()
