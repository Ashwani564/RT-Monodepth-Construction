#!/usr/bin/env python3
"""
Headless batch processor for RT-MonoDepth + YOLOv11 pipeline.

Reuses the model wrappers, depth logger and distance utilities defined in
``realtime_depth_video.py`` but runs without any OpenCV GUI calls so it can be
executed in a non-interactive shell (e.g. for automated paper-validation runs).

Outputs:
    depth_logs/depth_log_<timestamp>.csv     - per-detection depth log
    depth_logs/distance_log_<timestamp>.csv  - pairwise 3D distance log
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import cv2
import numpy as np
import torch

from realtime_depth_video import (
    DEFAULT_CAMERA_PARAMS,
    DepthLogger,
    RTMonoDepthModel,
    YOLODetector,
    calculate_object_distance,
    load_camera_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless RT-MonoDepth + YOLO video processor for CSV log generation",
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument(
        "-w",
        "--weights",
        default="./weights/RTMonoDepth/s/m_640_192/",
        help="RT-MonoDepth weights folder",
    )
    parser.add_argument("--width", type=int, default=640, help="Processing width (frame is rescaled keeping aspect ratio)")
    parser.add_argument(
        "--camera",
        default="macbook_m1_pro",
        choices=list(DEFAULT_CAMERA_PARAMS.keys()),
        help="Camera intrinsics preset",
    )
    parser.add_argument("--depth-scale", type=float, default=5.0, help="Initial depth scale factor")
    parser.add_argument(
        "--auto-calib",
        action="store_true",
        help="Enable anthropometric auto-calibration using detected persons",
    )
    parser.add_argument("--person-height", type=float, default=1.70, help="Assumed person height in meters")
    parser.add_argument(
        "--auto-calib-smoothing",
        type=float,
        default=0.9,
        help="EMA smoothing factor for the auto-calibrated scale",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Minimum seconds between depth-log snapshots (0 = log on every frame with detections)",
    )
    parser.add_argument(
        "--no-distance",
        action="store_true",
        help="Disable pairwise distance logging",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames (0 = all frames)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: every frame)",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO detection")
    parser.add_argument("--no-mlx", action="store_true", help="Disable MLX acceleration")
    return parser.parse_args()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def auto_calibrate(processor_state: dict, detections, depth_map, fx: float, person_height: float, smoothing: float) -> None:
    """Anthropometric auto calibration mirroring the logic in realtime_depth_video.py."""
    if not detections:
        return
    valid_scales = []
    frame_h = depth_map.shape[0]
    for det in detections:
        if det["class"].lower() != "person":
            continue
        x1, y1, x2, y2 = det["bbox"]
        bbox_h = max(1, y2 - y1)
        if bbox_h < 80 or bbox_h > frame_h * 0.9:
            continue
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        samples = []
        for dy in (-10, 0, 10):
            for dx in (-10, 0, 10):
                sx = int(np.clip(cx + dx, 0, depth_map.shape[1] - 1))
                sy = int(np.clip(cy + dy, 0, depth_map.shape[0] - 1))
                samples.append(depth_map[sy, sx])
        net_depth = float(np.median(samples))
        if net_depth <= 0:
            continue
        geom_depth = (fx * person_height) / float(bbox_h)
        candidate = geom_depth / net_depth
        if 0.05 < candidate < 100:
            valid_scales.append(candidate)
    if not valid_scales:
        return
    median_scale = float(np.median(valid_scales))
    processor_state["auto_scale"] = (
        smoothing * processor_state["auto_scale"] + (1.0 - smoothing) * median_scale
    )


def main() -> None:
    args = parse_args()

    device = select_device()
    print(f"Using device: {device}")

    camera_params = load_camera_params(args.camera)

    use_mlx = not args.no_mlx
    depth_model = RTMonoDepthModel(args.weights, device=device, use_mlx=use_mlx)

    yolo = None
    if not args.no_yolo:
        yolo = YOLODetector(device=device)
        if yolo.model is None:
            yolo = None

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {args.input}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input: {args.input}  frames={total_frames}  fps={fps:.2f}")

    logger = DepthLogger(
        log_interval=args.log_interval,
        enabled=True,
        measure_distances=not args.no_distance,
    )

    processor_state = {"auto_scale": 1.0}
    user_scale = float(args.depth_scale)

    frame_count = 0
    processed = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break
        if (frame_count - 1) % max(1, args.frame_stride) != 0:
            continue

        if frame.shape[1] != args.width:
            aspect = frame.shape[0] / frame.shape[1]
            new_h = int(args.width * aspect)
            frame = cv2.resize(frame, (args.width, new_h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = []
        if yolo is not None:
            detections = yolo.detect_objects(rgb)

        effective_scale = user_scale * processor_state["auto_scale"]
        depth_map = depth_model.predict_depth(rgb, camera_params, effective_scale)
        depth_map_resized = cv2.resize(
            depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        if args.auto_calib:
            auto_calibrate(
                processor_state,
                detections,
                depth_map_resized,
                fx=float(camera_params.get("fx", 640.0)),
                person_height=args.person_height,
                smoothing=args.auto_calib_smoothing,
            )

        logger.log_detections(detections, depth_map_resized, frame_count, camera_params)
        processed += 1

        if processed % 50 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"  processed={processed}/{frame_count}  "
                f"rate={rate:.2f} fps  detections={len(detections)}  "
                f"effective_scale={effective_scale:.3f}"
            )

    cap.release()
    logger.close()

    elapsed = time.time() - t0
    print(
        f"Done. Processed {processed} frames from {frame_count} read in {elapsed:.1f}s "
        f"({processed / max(elapsed, 1e-6):.2f} eff. fps)"
    )
    if logger.log_file_path:
        print(f"Depth log: {logger.log_file_path}")
    if logger.measure_distances and getattr(logger, "distance_log_file_path", None):
        print(f"Distance log: {logger.distance_log_file_path}")


if __name__ == "__main__":
    main()
