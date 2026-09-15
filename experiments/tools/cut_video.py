"""Create a frame-accurate MP4 excerpt from a source video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def cut_video(
    input_path: str | Path,
    output_path: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Write the interval ``[start_seconds, end_seconds)`` to a new MP4.

    Parameters
    ----------
    input_path:
        Source video path.
    output_path:
        Destination video path. Existing files are replaced.
    start_seconds:
        Start time from the beginning of the source, in seconds.
    end_seconds:
        End time from the beginning of the source, in seconds.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Source video not found: {input_path}")
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported source format: {input_path.suffix}")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("The interval must satisfy 0 <= start < end")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Source video has invalid FPS or dimensions")
    if start_seconds >= duration:
        capture.release()
        raise ValueError(
            f"Start time {start_seconds:.2f}s is after the video duration "
            f"({duration:.2f}s)"
        )

    end_seconds = min(end_seconds, duration)
    start_frame = int(start_seconds * fps)
    end_frame = min(int(end_seconds * fps), frame_count)
    expected_frames = end_frame - start_frame
    if expected_frames <= 0:
        capture.release()
        raise ValueError("The requested interval contains no complete frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    written_frames = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while written_frames < expected_frames:
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"Could not decode frame {start_frame + written_frames}"
                )
            writer.write(frame)
            written_frames += 1
    finally:
        capture.release()
        writer.release()

    if written_frames != expected_frames:
        raise RuntimeError(
            f"Incomplete output: wrote {written_frames}/{expected_frames} frames"
        )

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="End time in seconds")
    return parser


def main() -> None:
    """Run the video cutter from the command line."""
    args = build_parser().parse_args()
    output_path = cut_video(args.input, args.output, args.start, args.end)
    print(f"Created {output_path} ({args.start:.2f}s to {args.end:.2f}s)")


if __name__ == "__main__":
    main()