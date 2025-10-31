# modules/assign.py
import numpy as np
from modules.iou import compute_iou


def assign_anchors_to_gt(anchors, gt_boxes, pos_iou_threshold=0.5):
    """
    为每个 anchor 分配最匹配的真实框。
    参数:
      - anchors: numpy 数组，[N, 4]
      - gt_boxes: numpy 数组，[M, 4]
      - pos_iou_threshold: 正样本 IoU 阈值
    返回:
      - assigned_gt_indices: numpy 数组，[N,]，每个 anchor 对应的真实框索引（未匹配为 -1）
      - ious: numpy 数组，[N,]，每个 anchor 的最大 IoU 值
    """
    num_anchors = anchors.shape[0]
    assigned_gt_indices = -np.ones((num_anchors,), dtype=np.int32)
    ious = np.zeros((num_anchors,), dtype=np.float32)

    for i, gt in enumerate(gt_boxes):
        iou = compute_iou(gt, anchors)  # 形状: [N,]
        pos_indices = np.where(iou >= pos_iou_threshold)[0]
        assigned_gt_indices[pos_indices] = i
        ious[pos_indices] = iou[pos_indices]

    # 确保每个真实框至少有一个 anchor 被匹配
    for i, gt in enumerate(gt_boxes):
        iou = compute_iou(gt, anchors)
        best_anchor = np.argmax(iou)
        assigned_gt_indices[best_anchor] = i
        ious[best_anchor] = iou[best_anchor]

    return assigned_gt_indices, ious
