# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/iou3d_utils.py

import torch
from pointpillars.ops.iou3d_op import (
    boxes_overlap_bev_gpu,
    boxes_iou_bev_gpu,
    nms_gpu,
    nms_normal_gpu as nms_normal_gpu_op,
)


def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate boxes Overlap in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_overlap (torch.Tensor): Overlap result with shape (M, N).
    """
    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)

    return ans_overlap


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    # CPU fallback uses a different signature and returns kept indices directly.
    try:
        keep = nms_gpu(
            boxes,
            scores,
            thresh,
            pre_maxsize=pre_maxsize,
            post_maxsize=post_max_size,
        )
        if isinstance(keep, torch.Tensor):
            return keep.to(scores.device).contiguous()
    except TypeError:
        pass

    # CUDA extension compatibility path.
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = nms_gpu(boxes, keep, thresh, boxes.device.index)
    if isinstance(num_out, torch.Tensor):
        num_out = int(num_out.item())
    keep = order[keep[:num_out].to(order.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    try:
        keep = nms_normal_gpu_op(boxes, scores, thresh)
        if isinstance(keep, torch.Tensor):
            return keep.to(scores.device).contiguous()
    except TypeError:
        pass

    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = nms_normal_gpu_op(boxes, keep, thresh, boxes.device.index)
    if isinstance(num_out, torch.Tensor):
        num_out = int(num_out.item())
    return order[keep[:num_out].to(order.device)].contiguous()
