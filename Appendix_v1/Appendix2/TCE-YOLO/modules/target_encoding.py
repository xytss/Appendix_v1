# modules/target_encoding.py
import numpy as np
from modules.assign import assign_anchors_to_gt

def encode_box(gt_box, anchor):
    """
    将真实框转换为相对于 anchor 的偏移量编码。
    参数:
      - gt_box: [xmin, ymin, xmax, ymax]（真实框）
      - anchor: [xmin, ymin, xmax, ymax]（对应 anchor）
    返回:
      - 编码后的偏移量：[t_x, t_y, t_w, t_h]
    """
    gt_x = (gt_box[0] + gt_box[2]) / 2.0
    gt_y = (gt_box[1] + gt_box[3]) / 2.0
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]

    anchor_x = (anchor[0] + anchor[2]) / 2.0
    anchor_y = (anchor[1] + anchor[3]) / 2.0
    anchor_w = anchor[2] - anchor[0]
    anchor_h = anchor[3] - anchor[1]

    t_x = (gt_x - anchor_x) / anchor_w
    t_y = (gt_y - anchor_y) / anchor_h
    t_w = np.log(gt_w / anchor_w + 1e-7)
    t_h = np.log(gt_h / anchor_h + 1e-7)
    return np.array([t_x, t_y, t_w, t_h])

def build_targets(anchors, gt_boxes, gt_labels, pos_iou_threshold=0.5, num_anchors_expected=None):
    """
    根据所有 anchor 与真实框的匹配情况构建训练目标。
    参数:
      - anchors: numpy 数组，形状为 [N, 4]
      - gt_boxes: numpy 数组，形状为 [M, 4]
      - gt_labels: numpy 数组，形状为 [M,]（真实框对应的类别）
      - pos_iou_threshold: 正样本 IoU 阈值
      - num_anchors_expected: （可选）预期的 anchors 数量，如果提供，则检查 anchors.shape[0] 是否符合预期，
                               例如对于多 anchor 模式应为 H×W×A。
    返回:
      - reg_targets: [N, 4]，回归目标
      - cls_targets: [N,]，类别标签（未匹配 anchor 设为 0 表示背景）
      - mask: [N,]，正样本 mask（1 表示正样本，0 表示负样本）
    """
    print("build_targets: anchors shape:", anchors.shape)
    if num_anchors_expected is not None:
        if anchors.shape[0] != num_anchors_expected:
            print(f"Warning: Expected {num_anchors_expected} anchors, but got {anchors.shape[0]}")
    assigned_indices, ious = assign_anchors_to_gt(anchors, gt_boxes, pos_iou_threshold)
    print("build_targets: assigned_indices shape:", assigned_indices.shape)
    num_anchors = anchors.shape[0]
    reg_targets = np.zeros((num_anchors, 4), dtype=np.float32)
    cls_targets = np.zeros((num_anchors,), dtype=np.int64)  # 0 表示背景，单目标任务会标记为1
    mask = np.zeros((num_anchors,), dtype=np.float32)

    for idx in range(num_anchors):
        gt_idx = assigned_indices[idx]
        if gt_idx >= 0:
            mask[idx] = 1.0
            reg_targets[idx] = encode_box(gt_boxes[gt_idx], anchors[idx])
            cls_targets[idx] = 1  # 这里直接将目标类别设为 1，因为这是单目标任务
        else:
            cls_targets[idx] = 0  # 背景

    print("build_targets: reg_targets shape:", reg_targets.shape)
    print("build_targets: cls_targets shape:", cls_targets.shape)
    print("build_targets: mask shape:", mask.shape)
    return reg_targets, cls_targets, mask

# ---------------------------
# 测试入口（仅用于调试目标编码部分）
# ---------------------------
if __name__ == "__main__":
    # 假设采用多 anchor 模式：
    # 例如，输入图像尺寸为 640x640，对应的某特征层尺寸为 40x40，
    # 每个 grid cell 生成 A=13 个 anchor，则总数 N = 40 * 40 * 13 = 20800。
    H, W, A = 40, 40, 13
    N = H * W * A
    # 构造 dummy anchors，形状为 [N, 4]
    dummy_anchors = np.random.uniform(0, 640, size=(N, 4)).astype(np.float32)
    # 构造 dummy ground truth（假设有2个真实框）
    dummy_gt_boxes = np.array([[100, 100, 200, 200],
                               [300, 300, 400, 400]], dtype=np.float32)
    dummy_gt_labels = np.array([1, 2], dtype=np.int64)

    # 调用 build_targets，并期望 anchors 数量为 20800
    reg_targets, cls_targets, mask = build_targets(dummy_anchors, dummy_gt_boxes, dummy_gt_labels,
                                                   num_anchors_expected=N)
    print("Dummy reg_targets shape:", reg_targets.shape)  # 预期应为 [20800, 4]
    print("Dummy cls_targets shape:", cls_targets.shape)  # 预期应为 [20800,]
    print("Dummy mask shape:", mask.shape)  # 预期应为 [20800,]
