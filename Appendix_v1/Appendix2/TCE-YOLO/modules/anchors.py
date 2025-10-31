# modules/anchors.py
import math
import numpy as np

def generate_anchors(feature_map_size, scales, ratios, stride):
    """
    根据特征图尺寸、尺度、长宽比和步长生成 Anchor 框。
    参数:
      - feature_map_size: (H, W)
      - scales: 尺度列表，例如 [32, 64, 128]
      - ratios: 长宽比列表，例如 [0.5, 1.0, 2.0]
      - stride: 在原图上的步长
    返回:
      - anchors: numpy 数组，形状为 [N, 4]，每个 anchor 表示为 [xmin, ymin, xmax, ymax]
    """
    anchors = []
    H, W = feature_map_size
    for i in range(H):
        for j in range(W):
            center_x = (j + 0.5) * stride
            center_y = (i + 0.5) * stride
            for scale in scales:
                for ratio in ratios:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)
                    anchor = [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2]
                    anchors.append(anchor)
    return np.array(anchors)
