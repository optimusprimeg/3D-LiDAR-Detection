#!/workspaces/3D-LiDAR-Detection/PointPillars/venv/bin/python
"""Debug script: lower score threshold to check for any predicted boxes."""

from pathlib import Path
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
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [2, 6], [7, 3], [1, 5], [4, 0],
]

COLORS_IMG = {
    "Pedestrian": (255, 0, 0),
    "Cyclist": (0, 255, 0),
    "Car": (0, 0, 255),
    "DontCare": (0, 255, 255),
}


def point_range_filter(pts, point_range=(0, -39.68, -3, 69.12, 39.68, 1)):
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]


def draw_camera_boxes(image, image_points, labels):
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


def main():
    base = Path("/workspaces/3D-LiDAR-Detection/PointPillars/pointpillars/dataset/demo_data")
    pc_path = base / "val/000134.bin"
    calib_path = base / "val/000134.txt"
    img_path = base / "val/000134.png"
    
    ckpt_path = "/workspaces/3D-LiDAR-Detection/PointPillars/pretrained/epoch_160.pth"

    pc = read_points(str(pc_path))
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc).float()

    calib = read_calib(str(calib_path))
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    model = PointPillars(nclasses=3)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # sandbox: try multiple thresholds
    thresholds = [0.1, 0.05, 0.01, 0.005, 0.001]

    for score_thr in thresholds:
        print(f"\n=== Testing with score_thr={score_thr} ===")
        
        # Temporarily set the threshold
        model.score_thr = score_thr
        
        # Also patch the model to add debug output in get_predicted_bboxes_single
        original_get_pred = model.get_predicted_bboxes_single
        
        def debug_get_pred(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
            bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred.permute(1, 2, 0).reshape(-1, model.nclasses))
            scores = bbox_cls_pred_sigmoid.max(1)[0].detach().cpu().numpy()
            print(f"  Raw scores (before threshold): min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            above_thr = (scores > score_thr).sum()
            print(f"  Boxes above score_thr {score_thr}: {above_thr} / {len(scores)}")
            return original_get_pred(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors)
        
        model.get_predicted_bboxes_single = debug_get_pred
        
        with torch.no_grad():
            output = model(batched_pts=[pc_torch], mode="test")
        
        if isinstance(output, list) and len(output) > 0:
            result = output[0]
            if isinstance(result, dict):
                pred_count = len(result.get("lidar_bboxes", []))
                print(f"Predicted boxes: {pred_count}")
                
                if pred_count > 0:
                    print("SUCCESS! Found boxes with this threshold.")
                    print(f"Scores: {result['scores'][:5]}")  # show first 5 scores
                    print(f"Labels: {result['labels'][:5]}")
                    
                    # Try to draw it
                    result_filtered = keep_bbox_from_image_range(
                        result,
                        calib["Tr_velo_to_cam"].astype(np.float32),
                        calib["R0_rect"].astype(np.float32),
                        calib["P2"].astype(np.float32),
                        image.shape[:2],
                    )
                    result_filtered = keep_bbox_from_lidar_range(
                        result_filtered,
                        np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32),
                    )
                    
                    filtered_count = len(result_filtered.get("lidar_bboxes", []))
                    print(f"After image/lidar filtering: {filtered_count}")
                    
                    if filtered_count > 0:
                        camera_bboxes = result_filtered["camera_bboxes"]
                        image_points = points_camera2image(
                            bbox3d2corners_camera(camera_bboxes),
                            calib["P2"].astype(np.float32)
                        )
                        overlay = draw_camera_boxes(image, image_points, result_filtered["labels"])
                        vis_path = f"/workspaces/3D-LiDAR-Detection/PointPillars/debug_threshold_thr{score_thr}.png"
                        cv2.imwrite(vis_path, overlay)
                        print(f"Saved overlay to: {vis_path}")
                        break
            else:
                print(f"Result is not a dict: {type(result)}, raw: {result}")
        else:
            print(f"Output structure: {type(output)}, len={len(output) if isinstance(output, list) else 'N/A'}")


if __name__ == "__main__":
    main()
