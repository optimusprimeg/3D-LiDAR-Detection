#!/workspaces/3D-LiDAR-Detection/PointPillars/venv/bin/python
"""Run PointPillars inference on demo KITTI samples with GPU support.

This script is GPU-ready and will use CUDA if available, falling back to CPU.
It saves a text report with predicted and ground-truth overlays.
"""

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from pointpillars.model import PointPillars
from pointpillars.utils import (
    bbox3d2corners_camera,
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    points_camera2image,
    read_calib,
    read_label,
    read_points,
)


LINES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [2, 6],
    [7, 3],
    [1, 5],
    [4, 0],
]

COLORS_IMG = {
    "Pedestrian": (255, 0, 0),
    "Cyclist": (0, 255, 0),
    "Car": (0, 0, 255),
    "DontCare": (0, 255, 255),
}


def point_range_filter(pts: np.ndarray, point_range: Sequence[float] = (0, -39.68, -3, 69.12, 39.68, 1)) -> np.ndarray:
    point_range = np.asarray(point_range, dtype=np.float32)
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def summarize_output(output):
    lines = []
    lines.append(f"model output type: {type(output).__name__}")
    if isinstance(output, list):
        lines.append(f"batch size: {len(output)}")
        for idx, item in enumerate(output):
            lines.append(f"batch[{idx}] type: {type(item).__name__}")
            if isinstance(item, dict):
                lines.append(f"batch[{idx}] keys: {list(item.keys())}")
                for key in item:
                    val = item[key]
                    if isinstance(val, np.ndarray):
                        lines.append(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        lines.append(f"  {key}: {type(val).__name__}")
            elif isinstance(item, tuple):
                lines.append(f"batch[{idx}] tuple lengths: {[len(part) for part in item]}")
                lines.append(f"batch[{idx}] raw: {item}")
            else:
                lines.append(f"batch[{idx}] repr: {repr(item)}")
    else:
        lines.append(f"repr: {repr(output)}")
    return lines


def summarize_scene_objects(gt: dict) -> list[str]:
    names = gt["name"]
    location = gt["location"]
    rotation_y = gt["rotation_y"]
    dimensions = gt["dimensions"]

    lines = ["scene objects:"]
    preview_count = min(200, len(names))
    for idx in range(preview_count):
        name = names[idx]
        x, y, z = location[idx]
        h, w, l = dimensions[idx]
        yaw = rotation_y[idx]
        if z >= 0:
            depth_note = f"{z:.2f}m ahead"
        else:
            depth_note = f"{abs(z):.2f}m behind"
        side_note = "right side" if x > 0 else "left side"
        lines.append(
            f"- {name:<10} at x={x:.2f}, y={y:.2f}, z={z:.2f}  ({depth_note}, {side_note}, hwl={h:.2f},{w:.2f},{l:.2f}, yaw={yaw:.2f})"
        )

    remaining = len(names) - preview_count
    if remaining > 0:
        lines.append(f"- ... {remaining} more objects")
    return lines


def draw_camera_boxes(image: np.ndarray, image_points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for idx, bbox_points in enumerate(image_points):
        label_id = int(labels[idx])
        if label_id == 0:
            color = COLORS_IMG["Pedestrian"]
        elif label_id == 1:
            color = COLORS_IMG["Cyclist"]
        elif label_id == 2:
            color = COLORS_IMG["Car"]
        else:
            color = COLORS_IMG["DontCare"]
        for start, end in LINES:
            x1, y1 = bbox_points[start]
            x2, y2 = bbox_points[end]
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return overlay


def save_camera_overlay(
    image: np.ndarray | None,
    vis_dir: Path,
    sample_name: str,
    image_points: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    note: str | None = None,
    suffix: str = "image",
) -> Path:
    vis_dir.mkdir(parents=True, exist_ok=True)
    if image is None:
        overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        overlay = image.copy()
    if image_points is not None and labels is not None and len(image_points) > 0:
        overlay = draw_camera_boxes(overlay, image_points, labels)
    if note:
        cv2.putText(overlay, note, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    if image is None:
        cv2.putText(overlay, sample_name, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    vis_path = vis_dir / f"{sample_name.replace('/', '_')}_{suffix}.png"
    cv2.imwrite(str(vis_path), overlay)
    return vis_path


def run_sample(model, sample_name: str, pc_path: Path, calib_path: Path | None = None, gt_path: Path | None = None, img_path: Path | None = None, vis_dir: Path | None = None, device: torch.device = torch.device("cpu")):
    report = []
    report.append(f"SAMPLE: {sample_name}")
    report.append(f"point cloud file: {pc_path}")

    pc = read_points(str(pc_path))
    filtered_pc = point_range_filter(pc)
    report.append(f"raw points: {pc.shape[0]}")
    report.append(f"filtered points: {filtered_pc.shape[0]}")
    report.append(f"point format: [x, y, z, intensity]")
    report.append(f"xyz min: {np.round(pc[:, :3].min(axis=0), 3).tolist()}")
    report.append(f"xyz max: {np.round(pc[:, :3].max(axis=0), 3).tolist()}")
    report.append(f"intensity min/max: {float(pc[:, 3].min()):.3f} / {float(pc[:, 3].max()):.3f}")

    calib = None
    if calib_path is not None and calib_path.exists():
        calib = read_calib(str(calib_path))
        report.append(f"calib file: {calib_path}")
        report.append(f"calib keys: {sorted(calib.keys())}")

    image = None
    if img_path is not None and img_path.exists():
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if gt_path is not None and gt_path.exists():
        gt = read_label(str(gt_path))
        names = np.asarray(gt["name"])
        unique, counts = np.unique(names, return_counts=True)
        report.append(f"gt file: {gt_path}")
        report.append(f"gt objects: {len(names)}")
        report.append(f"gt class counts: {dict(zip(unique.tolist(), counts.tolist()))}")
        report.extend(summarize_scene_objects(gt))

        if image is not None and calib is not None and vis_dir is not None:
            camera_bboxes = np.concatenate([gt["location"], gt["dimensions"], gt["rotation_y"][:, None]], axis=-1)
            gt_labels = np.array([0 if name == "Pedestrian" else 1 if name == "Cyclist" else 2 if name == "Car" else -1 for name in gt["name"]], dtype=np.int32)
            valid = gt_labels >= 0
            if np.any(valid):
                camera_bboxes = camera_bboxes[valid]
                gt_labels = gt_labels[valid]
                image_points = points_camera2image(bbox3d2corners_camera(camera_bboxes), calib["P2"].astype(np.float32))
                overlay = draw_camera_boxes(image, image_points, gt_labels)
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{sample_name.replace('/', '_')}_gt.png"
                cv2.imwrite(str(vis_path), overlay)
                report.append(f"visual image saved: {vis_path}")
            else:
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{sample_name.replace('/', '_')}_image.png"
                cv2.imwrite(str(vis_path), image)
                report.append(f"visual image saved: {vis_path}")

    with torch.no_grad():
        pc_torch = torch.from_numpy(filtered_pc).float().to(device)
        output = model(batched_pts=[pc_torch], mode="test")
    report.extend(summarize_output(output))

    pred_result = None
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        pred_result = output[0]

    if pred_result is not None and calib is not None and image is not None:
        pred_result = keep_bbox_from_image_range(
            pred_result,
            calib["Tr_velo_to_cam"].astype(np.float32),
            calib["R0_rect"].astype(np.float32),
            calib["P2"].astype(np.float32),
            image.shape[:2],
        )
    if pred_result is not None:
        pred_result = keep_bbox_from_lidar_range(pred_result, np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32))

    if vis_dir is not None:
        pred_note = "No predicted detections (check model output above)"
        if image is not None:
            pred_count = int(len(pred_result["lidar_bboxes"])) if pred_result is not None else 0
            if pred_count > 0 and pred_result is not None:
                pred_data = pred_result
                camera_bboxes = pred_data["camera_bboxes"]
                image_points = points_camera2image(bbox3d2corners_camera(camera_bboxes), calib["P2"].astype(np.float32)) if calib is not None else None
                pred_vis_path = save_camera_overlay(
                    image,
                    vis_dir,
                    sample_name,
                    image_points=image_points,
                    labels=pred_data["labels"],
                    suffix="pred",
                )
            else:
                pred_vis_path = save_camera_overlay(image, vis_dir, sample_name, note=pred_note, suffix="pred")
            report.append(f"prediction overlay saved: {pred_vis_path}")
            report.append(f"predicted boxes after filtering: {pred_count}")
        else:
            pred_vis_path = save_camera_overlay(None, vis_dir, sample_name, note=pred_note, suffix="pred")
            report.append(f"prediction overlay saved: {pred_vis_path}")
            report.append("predicted boxes after filtering: 0")

    report.append("")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PointPillars demo inference with GPU support and save a text report.")
    parser.add_argument("--ckpt", default="/workspaces/3D-LiDAR-Detection/PointPillars/pretrained/epoch_160.pth")
    parser.add_argument("--output", default="/workspaces/3D-LiDAR-Detection/PointPillars/gpu_demo/demo_inference_output.txt")
    parser.add_argument("--visual-output-dir", default="/workspaces/3D-LiDAR-Detection/PointPillars/gpu_demo/demo_visuals")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU and run on CPU only")
    parser.add_argument("--score-thr", type=float, default=0.10, help="classification confidence threshold")
    parser.add_argument("--nms-thr", type=float, default=0.01, help="NMS IoU threshold")
    parser.add_argument("--nms-pre", type=int, default=100, help="top-k proposals before NMS")
    parser.add_argument("--max-num", type=int, default=50, help="max boxes after NMS")
    args = parser.parse_args()

    # Automatically detect GPU availability
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
        device_name = f"CUDA GPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    base = Path("/workspaces/3D-LiDAR-Detection/PointPillars/pointpillars/dataset/demo_data")
    samples = [
        (
            "val/000134",
            base / "val/000134.bin",
            base / "val/000134.txt",
            base / "val/000134_gt.txt",
        ),
        (
            "test/000002",
            base / "test/000002.bin",
            base / "test/000002.txt",
            None,
        ),
    ]

    model = PointPillars(nclasses=3)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    model.score_thr = float(args.score_thr)
    model.nms_thr = float(args.nms_thr)
    model.nms_pre = int(args.nms_pre)
    model.max_num = int(args.max_num)
    model.to(device)
    model.eval()

    lines = []
    lines.append("POINTPILLARS DEMO INFERENCE OUTPUT")
    lines.append("===================================")
    lines.append(f"checkpoint: {args.ckpt}")
    lines.append(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    lines.append(f"device: {device_name}")
    lines.append(f"inference thresholds: score_thr={model.score_thr}, nms_thr={model.nms_thr}, nms_pre={model.nms_pre}, max_num={model.max_num}")
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda device: {torch.cuda.get_device_name(0)}")
        lines.append(f"cuda capability: {torch.cuda.get_device_capability(0)}")
    lines.append(f"visual output dir: {args.visual_output_dir}")
    lines.append("color legend:")
    lines.append("- Pedestrian: blue")
    lines.append("- Cyclist: green")
    lines.append("- Car: red")
    lines.append("- DontCare: yellow")
    lines.append("overlay notes:")
    lines.append("- {sample}_gt.png is ground truth boxes drawn on the actual camera image (when GT file exists)")
    lines.append("- {sample}_pred.png is predicted boxes from the model drawn on the actual camera image")
    lines.append("")

    visual_output_dir = Path(args.visual_output_dir)
    for sample_name, pc_path, calib_path, gt_path in samples:
        img_path = pc_path.with_suffix(".png")
        lines.extend(run_sample(model, sample_name, pc_path, calib_path=calib_path, gt_path=gt_path, img_path=img_path, vis_dir=visual_output_dir, device=device))

    lines.append("INTERPRETATION")
    lines.append("---------------")
    lines.append("The checkpoint loads correctly and the demo data paths are valid.")
    lines.append("The forward pass completes on both demo samples.")
    if device.type == "cuda":
        lines.append(f"Running on {torch.cuda.get_device_name(0)}.")
        lines.append("Check the predicted overlay images to see model detections.")
    else:
        lines.append("Running on CPU. If GPU is available and predictions are poor, try with CUDA enabled.")

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
