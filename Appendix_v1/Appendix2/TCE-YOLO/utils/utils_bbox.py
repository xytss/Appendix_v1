import numpy as np
import torch
from torchvision.ops import nms
import pkg_resources as pkg


def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    根据特征图生成 anchor **中心点**（返回 (cx, cy)），以及对应的 stride 张量。
    feats: 一个包含特征图的列表，每个特征图形状为 [B, C, H, W]
    strides: 对应特征图在原图上的步长
    返回：anchor_points: [sum(H*W), 2]，stride_tensor: [sum(H*W), 1]
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance(ltrb) to box(xywh or xyxy).
    将预测的偏移量转换为边界框坐标。
    distance: [B, 4, N]  (ltrb)
    anchor_points: [1 or B, 2, N]  (cx, cy)
    """
    lt, rb = torch.split(distance, 2, dim)
    print("dist2bbox: lt shape:", lt.shape)  # 例如 [B, 2, N]
    print("dist2bbox: anchor_points shape:", anchor_points.shape)  # 例如 [1, 2, N]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh   = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # 返回 [B, 4, N]
    return torch.cat((x1y1, x2y2), dim)


class DecodeBox():
    def __init__(self, num_classes, input_shape, use_obj_score: bool = False, **kwargs):
        """
        DecodeBox 用于解码模型的预测输出。
        num_classes: 类别数（用于上游配置；本类会在必要时根据张量自适配）
        input_shape: 输入图像尺寸 (h, w)
        use_obj_score: 是否在推理时使用 obj×cls（默认 False）
        """
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.bbox_attrs  = 4 + num_classes
        self.input_shape = input_shape
        self.use_obj_score = use_obj_score

    @staticmethod
    def _flatten_to_b_c_n(x: torch.Tensor, B: int, num_classes: int, N_box: int = None) -> torch.Tensor:
        """
        统一展平成 [B, C, N]。当 x 是 [B, Cp, H, W] 且给定 N_box=H*W*A 时，
        优先用 A=N_box/(H*W) 推断 C_infer=Cp/A（若能整除），
        直接重排为 [B, C_infer, N_box]，避免依赖外部 num_classes。
        支持：
          - [nl,B,C,H,W] / [B,nl,C,H,W]
          - [B, A*C, H, W]（自动识别 A、C）
          - [B, C, H, W]
          - [B, C, N] / [B, N, C]
        """
        if x.dim() == 5:
            if x.size(0) == B:
                x = x.permute(0, 2, 1, 3, 4).contiguous().flatten(2)  # [B,nl,C,H,W] -> [B,C,nl*H*W]
            else:
                x = x.permute(1, 2, 0, 3, 4).contiguous().flatten(2)  # [nl,B,C,H,W] -> [B,C,nl*H*W]
            return x

        if x.dim() == 4:
            Bx, Cp, H, W = x.shape
            if N_box is not None and (H * W) > 0 and (N_box % (H * W) == 0):
                A = N_box // (H * W)
                if A > 0 and (Cp % A == 0):
                    C_infer = Cp // A
                    # [B, Cp, H, W] -> [B, C_infer, A, H, W] -> [B, C_infer, H, W, A] -> [B, C_infer, N_box]
                    x = x.view(B, C_infer, A, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, C_infer, N_box)
                    print(f"[decode] inferred (A,C)=({A},{C_infer}) from cls 4D shape {tuple((Bx,Cp,H,W))} and N_box={N_box}")
                    return x
            # 兜底：当识别不到 A 时，保持原逻辑（仅展平空间维）
            return x.flatten(2)  # [B,Cp,H*W]

        if x.dim() == 3:
            if x.size(1) == num_classes:
                return x              # [B,C,N]
            if x.size(2) == num_classes:
                return x.permute(0, 2, 1).contiguous()  # [B,N,C] -> [B,C,N]
            # 智能回退：按 [B,C,N] 重构
            total = x.numel()
            if num_classes > 0 and total % (B * num_classes) == 0:
                N = total // (B * num_classes)
                return x.view(B, num_classes, N)
            # 不可恢复则如实报错
            raise RuntimeError(f"Unsupported 3D cls shape {tuple(x.shape)} for num_classes={num_classes}")

        raise RuntimeError(f"Unsupported cls shape: {tuple(x.shape)}")

    def decode_box(self, inputs):
        """
        解码预测输出：
          inputs 可以是：
            - 5 元组: (dbox, cls, origin_cls, anchors, strides)
            - 6 元组: (dbox, cls, obj, origin_cls, anchors, strides)
          其中：
            dbox:    [B, 4, N]，ltrb 距离（相对 anchor_point）
            cls:     分类预测，5D/4D/3D，最终展平到 [B, C, N]
            obj:     （可选）objectness/centerness，展平到 [B, 1, N]
            anchors: [N, 2] 或 [B, N, 2]，中心坐标 (cx, cy)
            strides: 标量或可广播到 [B,1,N] / [B,4,N]
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 6:
            dbox, cls, obj, origin_cls, anchors, strides = inputs
        else:
            dbox, cls, origin_cls, anchors, strides = inputs
            obj = None

        print("DecodeBox.decode_box: initial dbox shape:", dbox.shape)        # [B, 4, N]
        print("DecodeBox.decode_box: initial anchors shape:", anchors.shape)  # [N,2] 或 [B,N,2]

        B = dbox.size(0)
        N = dbox.size(-1)

        # anchors -> [1 or B, 2, N]
        if anchors.dim() == 2 and anchors.size(-1) == 2:          # [N,2]
            anchors_processed = anchors.unsqueeze(0).permute(0, 2, 1)  # [1,2,N]
        elif anchors.dim() == 3 and anchors.size(-1) == 2:        # [B,N,2]
            anchors_processed = anchors.permute(0, 2, 1)               # [B,2,N]
        else:
            raise RuntimeError(f"Unsupported anchors shape: {tuple(anchors.shape)}")
        print("DecodeBox.decode_box: processed anchors shape:", anchors_processed.shape)

        # 解码边界框（center-xywh）
        dbox_decoded = dist2bbox(dbox, anchors_processed, xywh=True, dim=1)
        print("DecodeBox.decode_box: dbox_decoded shape (before scaling):", dbox_decoded.shape)

        # 处理 strides 广播
        if not torch.is_tensor(strides):
            strides = torch.tensor(strides, dtype=dbox_decoded.dtype, device=dbox_decoded.device)
        if strides.dim() == 0:
            pass  # 标量，自动广播
        elif strides.dim() == 1 and strides.numel() == N:
            strides = strides.view(1, 1, N)  # [1,1,N]，与 [B,4,N] 广播
        elif strides.dim() == 2 and strides.shape[0] == N:  # [N,1]
            strides = strides.view(1, 1, N)
        dbox_scaled = dbox_decoded * strides
        print("DecodeBox.decode_box: dbox shape after multiplying strides:", dbox_scaled.shape)

        # 分类分支：统一展平为 [B, C, N]，优先依据 N 推断 A 与 C
        cls_logits  = cls
        cls_sigmoid = torch.sigmoid(cls_logits)
        cls_sigmoid = self._flatten_to_b_c_n(cls_sigmoid, B=B, num_classes=self.num_classes, N_box=N)

        # （可选）obj/centerness × cls（同样带上 N 以对齐）
        if self.use_obj_score and (obj is not None):
            obj_sigmoid = torch.sigmoid(obj)
            obj_sigmoid = self._flatten_to_b_c_n(obj_sigmoid, B=B, num_classes=1, N_box=N)  # -> [B,1,N]
            if obj_sigmoid.size(1) != 1:
                raise RuntimeError(f"Objectness after flatten must be [B,1,N], got {tuple(obj_sigmoid.shape)}")
            cls_sigmoid = cls_sigmoid * obj_sigmoid  # [B,C,N] *= [B,1,N]（广播）

        # 与 boxes 的 N 对齐检查；必要时安全回退 reshape
        N_box = dbox_scaled.size(-1)
        N_cls = cls_sigmoid.size(-1)
        if N_box != N_cls:
            total  = cls_sigmoid.numel()
            # 优先尝试按 N_box 强行 view
            if total % (B * N_box) == 0:
                C_infer = total // (B * N_box)
                cls_sigmoid = cls_sigmoid.view(B, C_infer, N_box)
                print(f"[decode] fixed N mismatch by reshape -> C_infer={C_infer}, N={N_box}")
                N_cls = N_box
            else:
                raise RuntimeError(f"N mismatch between boxes ({N_box}) and classes ({N_cls}). "
                                   f"numel={total}, expected multiple of {B}*C*N. "
                                   f"Check feature-level flatten/concat order.")

        # 拼接 -> [B, 4 + C, N]
        y = torch.cat((dbox_scaled, cls_sigmoid), dim=1)
        print("DecodeBox.decode_box: shape after concat (before permute):", y.shape)

        # 调整到 [B, N, 4+C]
        y = y.permute(0, 2, 1).contiguous()
        print("DecodeBox.decode_box: final output shape after permute:", y.shape)

        # 归一化边界框坐标到 [0, 1]
        normalization = y.new_tensor([self.input_shape[1], self.input_shape[0],
                                      self.input_shape[1], self.input_shape[0]])
        y[:, :, :4] = y[:, :, :4] / normalization
        print("DecodeBox.decode_box: final output shape after normalization:", y.shape)

        return y

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins  = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1], box_mins[..., 1:2],
            box_maxes[..., 0:1], box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image,
                            conf_thres=0.2, nms_thres=0.1):
        """
        prediction: [B, N, 4 + C]，坐标为归一化 xywh
        这里以预测张量的第三维自动推断 C，忽略传入的 num_classes 参数，避免配置与模型不一致。
        """
        # 自动推断类别数
        num_classes_pred = prediction.size(2) - 4
        if num_classes_pred <= 0:
            raise RuntimeError(f"Bad prediction shape {tuple(prediction.shape)}: C<=0")
        if num_classes != num_classes_pred:
            print(f"[nms] WARNING: num_classes({num_classes}) != inferred({num_classes_pred}), "
                  f"using inferred value.")

        # xywh -> xyxy（在原始 image scale 上会在 yolo_correct_boxes 里处理）
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 取每个类别的最大分数及类别索引
            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes_pred], 1, keepdim=True)
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
