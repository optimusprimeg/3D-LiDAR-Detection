# CPU fallback implementation for 3D IOU and NMS operations
import torch
import numpy as np


def boxes_overlap_bev_cpu(boxes_a, boxes_b):
    """Calculate 2D (BEV) overlap between boxes.
    
    Args:
        boxes_a: [N, 7] boxes in format [x, y, z, dx, dy, dz, yaw]
        boxes_b: [M, 7] boxes in format [x, y, z, dx, dy, dz, yaw]
        
    Returns:
        overlap: [N, M] overlap matrix
    """
    def get_corners_2d(boxes):
        """Get 2D corners of boxes (BEV view)."""
        N = boxes.shape[0]
        corners = np.zeros((N, 4, 2), dtype=np.float32)
        
        for i in range(N):
            # Support both [x, y, z, dx, dy, dz, yaw] and [x1, y1, x2, y2, yaw]
            if boxes.shape[1] >= 7:
                x, y, dx, dy, yaw = boxes[i, 0], boxes[i, 1], boxes[i, 3], boxes[i, 4], boxes[i, 6]
            elif boxes.shape[1] == 5:
                x1, y1, x2, y2, yaw = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]
                x = (x1 + x2) * 0.5
                y = (y1 + y2) * 0.5
                dx = max(x2 - x1, 1e-6)
                dy = max(y2 - y1, 1e-6)
            else:
                raise ValueError(f"Unsupported box format with shape {boxes.shape}")
            
            # Half dimensions
            hdx = dx / 2
            hdy = dy / 2
            
            # Corners in local coordinates
            local_corners = np.array([
                [-hdx, -hdy],
                [hdx, -hdy],
                [hdx, hdy],
                [-hdx, hdy]
            ], dtype=np.float32)
            
            # Rotation matrix
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rot_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ], dtype=np.float32)
            
            # Apply rotation and translation
            rotated_corners = local_corners @ rot_matrix.T
            corners[i] = rotated_corners + np.array([x, y], dtype=np.float32)
        
        return corners
    
    def polygon_area(corners):
        """Calculate polygon area using shoelace formula."""
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def polygon_intersection_area(corners1, corners2):
        """Simple approximation: use bounding box intersection."""
        x1_min, x1_max = corners1[:, 0].min(), corners1[:, 0].max()
        y1_min, y1_max = corners1[:, 1].min(), corners1[:, 1].max()
        x2_min, x2_max = corners2[:, 0].min(), corners2[:, 0].max()
        y2_min, y2_max = corners2[:, 1].min(), corners2[:, 1].max()
        
        x_inter = min(x1_max, x2_max) - max(x1_min, x2_min)
        y_inter = min(y1_max, y2_max) - max(y1_min, y2_min)
        
        if x_inter > 0 and y_inter > 0:
            return x_inter * y_inter
        return 0.0
    
    device = boxes_a.device
    boxes_a = boxes_a.detach().cpu().numpy() if boxes_a.is_cuda else boxes_a.detach().numpy()
    boxes_b = boxes_b.detach().cpu().numpy() if boxes_b.is_cuda else boxes_b.detach().numpy()
    
    corners_a = get_corners_2d(boxes_a)
    corners_b = get_corners_2d(boxes_b)
    
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    overlap = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        area_a = polygon_area(corners_a[i])
        for j in range(M):
            area_b = polygon_area(corners_b[j])
            inter_area = polygon_intersection_area(corners_a[i], corners_b[j])
            
            if area_a + area_b - inter_area > 0:
                overlap[i, j] = inter_area / (area_a + area_b - inter_area)
    
    return torch.from_numpy(overlap).to(device)


def boxes_iou_bev_cpu(boxes_a, boxes_b):
    """Calculate 3D IOU using BEV projection.
    
    Args:
        boxes_a: [N, 7] boxes
        boxes_b: [M, 7] boxes
        
    Returns:
        iou: [N, M] IOU matrix
    """
    return boxes_overlap_bev_cpu(boxes_a, boxes_b)


def nms_gpu(boxes, scores, thresh=0.5, pre_maxsize=None, post_maxsize=None):
    """CPU NMS implementation."""
    device = boxes.device
    
    if boxes.is_cuda:
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
    else:
        boxes_np = boxes.numpy()
        scores_np = scores.numpy()
    
    # Sort by score
    sorted_idx = np.argsort(-scores_np)
    
    if pre_maxsize is not None:
        sorted_idx = sorted_idx[:pre_maxsize]
    
    keep = []
    while len(sorted_idx) > 0:
        # Keep the box with highest score
        current_idx = sorted_idx[0]
        keep.append(int(current_idx))
        
        if len(sorted_idx) == 1:
            break
        
        sorted_idx = sorted_idx[1:]
        
        # Calculate IOU with remaining boxes
        current_box = boxes_np[current_idx:current_idx+1]
        remaining_boxes = boxes_np[sorted_idx]
        
        iou_matrix = boxes_overlap_bev_cpu(
            torch.from_numpy(current_box),
            torch.from_numpy(remaining_boxes)
        ).numpy()
        
        # Remove boxes with IOU > thresh
        keep_mask = iou_matrix[0] <= thresh
        sorted_idx = sorted_idx[keep_mask]
    
    if post_maxsize is not None:
        keep = keep[:post_maxsize]
    
    return torch.tensor(keep, device=device, dtype=torch.long)


def nms_normal_gpu(boxes, scores, thresh=0.5):
    """Normal NMS without pre/post max size."""
    return nms_gpu(boxes, scores, thresh)


# Export functions for compatibility
boxes_overlap_bev_gpu = boxes_overlap_bev_cpu
boxes_iou_bev_gpu = boxes_iou_bev_cpu
