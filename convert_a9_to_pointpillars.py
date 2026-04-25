#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np


CALIB_CONTENT = (
    "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "R0_rect: 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
)


CLASS_MAP = {
    "CAR": "Car",
    "TRUCK": "Truck",
    "VAN": "Truck",
    "TRAILER": "Truck",
    "PEDESTRIAN": "Pedestrian",
    "BICYCLE": "Bicycle",
    "CYCLIST": "Bicycle",
    "BICYCLIST": "Bicycle",
}


OCCLUSION_MAP = {
    "NOT_OCCLUDED": 0,
    "PARTIALLY_OCCLUDED": 1,
    "MOSTLY_OCCLUDED": 2,
}


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # OpenLABEL cuboid quaternion is stored as (qx, qy, qz, qw).
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))


def _normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return intensity
    max_intensity = float(np.max(intensity))
    if max_intensity > 1.0:
        intensity = intensity / 65535.0
    intensity = np.clip(intensity, 0.0, 1.0)
    return intensity


def read_pcd_xyzi(pcd_path: Path) -> np.ndarray:
    header_lines = 0
    fields = []
    data_kind = None
    with pcd_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            header_lines += 1
            line = line.strip()
            if not line:
                continue
            if line.startswith("FIELDS "):
                fields = line.split()[1:]
            elif line.startswith("DATA "):
                data_kind = line.split()[1].lower()
                break

    if data_kind != "ascii":
        # Fallback for binary or binary_compressed PCD.
        try:
            import open3d as o3d
        except ImportError as e:
            raise ValueError(
                f"Non-ASCII PCD detected ({data_kind}) and open3d is not available: {pcd_path}"
            ) from e

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            intensity = np.asarray(pcd.colors, dtype=np.float32)[:, 0:1]
        else:
            intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        intensity = _normalize_intensity(intensity)
        return np.hstack([xyz, intensity]).astype(np.float32)

    if not fields:
        raise ValueError(f"Could not parse PCD fields in: {pcd_path}")

    raw = np.loadtxt(pcd_path, dtype=np.float32, skiprows=header_lines)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    field_to_idx = {name: i for i, name in enumerate(fields)}
    for req in ("x", "y", "z"):
        if req not in field_to_idx:
            raise ValueError(f"Missing required field '{req}' in {pcd_path}")

    xyz = raw[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]]
    if "intensity" in field_to_idx:
        intensity = raw[:, field_to_idx["intensity"]].reshape(-1, 1)
        intensity = _normalize_intensity(intensity)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)

    points = np.hstack([xyz, intensity]).astype(np.float32)
    return points


def parse_openlabel_to_kitti_lines(label_path: Path):
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    skipped = 0
    frames = data.get("openlabel", {}).get("frames", {})
    for frame in frames.values():
        objects = frame.get("objects", {})
        for obj in objects.values():
            obj_data = obj.get("object_data", {})
            src_type = obj_data.get("type") or obj.get("type") or ""
            src_type = str(src_type).upper()
            class_name = CLASS_MAP.get(src_type)
            if class_name is None:
                skipped += 1
                continue

            cuboid = obj_data.get("cuboid", {}).get("val", None)
            if cuboid is None or len(cuboid) < 10:
                skipped += 1
                continue

            attrs_text = obj_data.get("cuboid", {}).get("attributes", {}).get("text", [])
            occluded = 3
            for item in attrs_text:
                if item.get("name") == "occlusion_level":
                    occluded = OCCLUSION_MAP.get(item.get("val"), 3)

            x_center = float(cuboid[0])
            y_center = float(cuboid[1])
            z_center = float(cuboid[2])
            qx, qy, qz, qw = map(float, cuboid[3:7])
            length = float(cuboid[7])
            width = float(cuboid[8])
            height = float(cuboid[9])
            yaw = quaternion_to_yaw(qx, qy, qz, qw)

            # Keep KITTI's 15-column text layout for PointPillars compatibility.
            line = (
                f"{class_name} 0.00 {occluded} 0.00 "
                f"0.00 0.00 0.00 0.00 "
                f"{height:.4f} {width:.4f} {length:.4f} "
                f"{x_center:.4f} {y_center:.4f} {z_center:.4f} {yaw:.4f}"
            )
            lines.append(line)
    return lines, skipped


def write_ids(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert A9 OpenLABEL + PCD to PointPillars KITTI layout")
    parser.add_argument("--dataset-root", required=True, help="Path to a9_dataset_r02_s01")
    parser.add_argument("--sensor", default="s110_lidar_ouster_south", help="Lidar sensor folder name")
    parser.add_argument("--out-root", required=True, help="Output KITTI root folder")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument(
        "--pointpillars-imagesets-dir",
        default="",
        help="Optional PointPillars/pointpillars/dataset/ImageSets dir to also write split files",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve()

    pcd_dir = dataset_root / "point_clouds" / args.sensor
    label_dir = dataset_root / "labels_point_clouds" / args.sensor
    if not pcd_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing source folders. Expected {pcd_dir} and {label_dir}"
        )

    train_velodyne = out_root / "training" / "velodyne"
    train_label_2 = out_root / "training" / "label_2"
    train_calib = out_root / "training" / "calib"
    test_velodyne = out_root / "testing" / "velodyne"
    test_calib = out_root / "testing" / "calib"
    imagesets_dir = out_root / "ImageSets"
    for d in (train_velodyne, train_label_2, train_calib, test_velodyne, test_calib, imagesets_dir):
        d.mkdir(parents=True, exist_ok=True)

    label_by_stem = {p.stem.replace(f"_{args.sensor}", ""): p for p in sorted(label_dir.glob("*.json"))}
    pcd_files = sorted(pcd_dir.glob("*.pcd"))
    if not pcd_files:
        raise RuntimeError(f"No .pcd files found in {pcd_dir}")

    ids = []
    total_objects = 0
    skipped_objects = 0
    matched_frames = 0

    for idx, pcd_path in enumerate(pcd_files):
        stem = pcd_path.stem.replace(f"_{args.sensor}", "")
        label_path = label_by_stem.get(stem)
        if label_path is None:
            continue

        frame_id = f"{idx:06d}"
        points = read_pcd_xyzi(pcd_path)
        points.tofile(train_velodyne / f"{frame_id}.bin")

        kitti_lines, skipped = parse_openlabel_to_kitti_lines(label_path)
        (train_label_2 / f"{frame_id}.txt").write_text("\n".join(kitti_lines) + ("\n" if kitti_lines else ""), encoding="utf-8")
        (train_calib / f"{frame_id}.txt").write_text(CALIB_CONTENT, encoding="utf-8")

        ids.append(frame_id)
        matched_frames += 1
        total_objects += len(kitti_lines)
        skipped_objects += skipped

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(pcd_files)} frames")

    if not ids:
        raise RuntimeError("No frames were converted. Check sensor names and source folders.")

    split_idx = max(1, min(len(ids) - 1, int(len(ids) * args.train_ratio)))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    trainval_ids = ids

    write_ids(train_ids, imagesets_dir / "train.txt")
    write_ids(val_ids, imagesets_dir / "val.txt")
    write_ids(trainval_ids, imagesets_dir / "trainval.txt")
    write_ids([], imagesets_dir / "test.txt")

    if args.pointpillars_imagesets_dir:
        pp_imagesets = Path(args.pointpillars_imagesets_dir).resolve()
        pp_imagesets.mkdir(parents=True, exist_ok=True)
        write_ids(train_ids, pp_imagesets / "train.txt")
        write_ids(val_ids, pp_imagesets / "val.txt")
        write_ids(trainval_ids, pp_imagesets / "trainval.txt")
        write_ids([], pp_imagesets / "test.txt")

    summary = {
        "dataset_root": str(dataset_root),
        "sensor": args.sensor,
        "output_root": str(out_root),
        "source_pcd_frames": len(pcd_files),
        "matched_frames": matched_frames,
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "objects_written": total_objects,
        "objects_skipped_unmapped_or_invalid": skipped_objects,
    }
    (out_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Conversion finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()