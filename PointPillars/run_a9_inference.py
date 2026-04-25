import argparse
from pathlib import Path

import numpy as np
import torch

from pointpillars.model import PointPillars
from pointpillars.utils import read_points


CLASS_MAP_3 = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
CLASS_MAP_4 = {0: "Car", 1: "Truck", 2: "Pedestrian", 3: "Bicycle"}


def filter_points(pc: np.ndarray, point_range: np.ndarray) -> np.ndarray:
    mask = (
        (pc[:, 0] > point_range[0])
        & (pc[:, 0] < point_range[3])
        & (pc[:, 1] > point_range[1])
        & (pc[:, 1] < point_range[4])
        & (pc[:, 2] > point_range[2])
        & (pc[:, 2] < point_range[5])
    )
    return pc[mask]


def infer_one(model: PointPillars, pc_path: Path, point_range: np.ndarray, topk: int = 20) -> list[str]:
    pc = read_points(str(pc_path))
    pc = filter_points(pc, point_range)

    with torch.no_grad():
        out = model(batched_pts=[torch.from_numpy(pc).float()], mode="test")[0]

    lines = []
    lines.append(f"pc_path: {pc_path}")
    lines.append(f"points_after_filter: {len(pc)}")

    if isinstance(out, tuple):
        lines.append("raw_output_type: tuple")
        lengths = [len(p) if hasattr(p, "__len__") else -1 for p in out]
        lines.append(f"tuple_part_lengths: {lengths}")
        return lines

    if not isinstance(out, dict):
        lines.append(f"raw_output_type: {type(out).__name__}")
        lines.append(repr(out))
        return lines

    lines.append("raw_output_type: dict")
    bboxes = out.get("lidar_bboxes", [])
    labels = np.asarray(out.get("labels", []))
    scores = np.asarray(out.get("scores", []))

    lines.append(f"num_boxes: {len(bboxes)}")
    if len(scores) == 0:
        return lines

    lines.append(f"score_min_max: {float(scores.min()):.4f}, {float(scores.max()):.4f}")
    order = np.argsort(-scores)
    order = order[: min(topk, len(order))]

    class_map = CLASS_MAP_4 if model.nclasses == 4 else CLASS_MAP_3
    lines.append("top_predictions:")
    for rank, idx in enumerate(order, start=1):
        cls_name = class_map.get(int(labels[idx]), str(int(labels[idx])))
        bb = np.asarray(bboxes[idx])
        lines.append(
            f"  {rank:02d}. class={cls_name}, score={float(scores[idx]):.4f}, "
            f"xyz=({bb[0]:.2f},{bb[1]:.2f},{bb[2]:.2f}), "
            f"lwh=({bb[3]:.2f},{bb[4]:.2f},{bb[5]:.2f}), yaw={bb[6]:.2f}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PointPillars inference on A9-converted KITTI .bin files")
    parser.add_argument("--ckpt", default="pretrained/epoch_160.pth")
    parser.add_argument("--pc-path", required=True, help="Path to one KITTI velodyne .bin file")
    parser.add_argument("--out", default="inference_report.txt")
    parser.add_argument("--nclasses", type=int, default=3, choices=[3, 4])
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--nms-pre", type=int, default=1000)
    parser.add_argument("--max-num", type=int, default=200)
    args = parser.parse_args()

    point_range = np.array([-50, -50, -8, 49.84, 49.84, 5], dtype=np.float32)

    model = PointPillars(nclasses=args.nclasses)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.score_thr = args.score_thr
    model.nms_pre = args.nms_pre
    model.max_num = args.max_num
    model.eval()

    lines = [
        "A9 INFERENCE REPORT",
        "===================",
        f"checkpoint: {args.ckpt}",
        f"nclasses: {args.nclasses}",
        f"score_thr: {args.score_thr}",
        f"nms_pre: {args.nms_pre}",
        f"max_num: {args.max_num}",
    ]
    lines.extend(infer_one(model, Path(args.pc_path), point_range))

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
