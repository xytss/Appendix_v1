# modules/iou.py
import numpy as np

def compute_iou(box, anchors):
    """
    计算单个真实框与所有 anchor 框的 IoU。
    参数:
      - box: [xmin, ymin, xmax, ymax]
      - anchors: numpy 数组，形状 [N, 4]
    返回:
      - iou: numpy 数组，形状 [N,]，每个元素为对应 anchor 的 IoU
    """
    x_min = np.maximum(box[0], anchors[:, 0])
    y_min = np.maximum(box[1], anchors[:, 1])
    x_max = np.minimum(box[2], anchors[:, 2])
    y_max = np.minimum(box[3], anchors[:, 3])
    inter_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    union_area = box_area + anchors_area - inter_area
    iou = inter_area / (union_area + 1e-7)
    return iou
