import torch
import torch.nn as nn

def eiou_loss(pred_bboxes, target_bboxes, eps=1e-7):
    """EIoU on xyxy; 若你用的是 ltrb/距离空间监督，请确保 target 与 pred 同编码。"""
    inter = (torch.min(pred_bboxes[..., 2], target_bboxes[..., 2]) -
             torch.max(pred_bboxes[..., 0], target_bboxes[..., 0])).clamp_(min=0) * \
            (torch.min(pred_bboxes[..., 3], target_bboxes[..., 3]) -
             torch.max(pred_bboxes[..., 1], target_bboxes[..., 1])).clamp_(min=0)

    union = ((pred_bboxes[..., 2] - pred_bboxes[..., 0]) *
             (pred_bboxes[..., 3] - pred_bboxes[..., 1]) +
             (target_bboxes[..., 2] - target_bboxes[..., 0]) *
             (target_bboxes[..., 3] - target_bboxes[..., 1]) - inter + eps)

    iou = inter / (union + eps)

    pcx = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) * 0.5
    pcy = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) * 0.5
    tcx = (target_bboxes[..., 0] + target_bboxes[..., 2]) * 0.5
    tcy = (target_bboxes[..., 1] + target_bboxes[..., 3]) * 0.5
    center_dist = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    pw = (pred_bboxes[..., 2] - pred_bboxes[..., 0]).clamp_min(eps)
    ph = (pred_bboxes[..., 3] - pred_bboxes[..., 1]).clamp_min(eps)
    tw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp_min(eps)
    th = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp_min(eps)
    aspect_ratio_loss = ((pw - tw).pow(2) + (ph - th).pow(2)) / (pw.pow(2) + ph.pow(2) + eps)

    ex1 = torch.min(pred_bboxes[..., 0], target_bboxes[..., 0])
    ey1 = torch.min(pred_bboxes[..., 1], target_bboxes[..., 1])
    ex2 = torch.max(pred_bboxes[..., 2], target_bboxes[..., 2])
    ey2 = torch.max(pred_bboxes[..., 3], target_bboxes[..., 3])
    enclose_diag = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    eiou = 1 - iou + center_dist / enclose_diag + aspect_ratio_loss
    return eiou.mean()


class Loss(nn.Module):
    def __init__(self, model=None):
        super(Loss, self).__init__()
        self.model = model

    # --------- 工具：把 4D 回归图 [B, A*5, H, W] -> [B, N, 4]，每 5 个取前 4 个 ---------
    @staticmethod
    def _reg_from_a5(bchw: torch.Tensor) -> torch.Tensor:
        # bchw: [B, C, H, W], C = A*5
        B, C, H, W = bchw.shape
        if C % 5 != 0:
            raise RuntimeError(f"expect C == A*5, but got C={C}")
        A = C // 5
        x = bchw.view(B, A, 5, H, W)[:, :, :4]  # [B, A, 4, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, 4]
        return x.view(B, A * H * W, 4)            # [B, N, 4]

    # --------- 工具：统一任意预测到 [B, N, 4] ---------
    @staticmethod
    def _to_b_n_4_any(pred) -> torch.Tensor:
        """
        接受以下形式并转为 [B,N,4]：
          - Tensor [B,4,N] / [B,N,4] / [B,4,H,W] / [B,A*5,H,W]
          - list/tuple: 含多个层级的回归图（4D且 C=A*5），会拼接
        """
        if torch.is_tensor(pred):
            # 直接是 Tensor
            if pred.dim() == 3:
                if pred.size(1) == 4:      # [B,4,N]
                    return pred.permute(0, 2, 1).contiguous()
                if pred.size(2) == 4:      # [B,N,4]
                    return pred.contiguous()
                raise RuntimeError(f"3D pred shape {tuple(pred.shape)} not one of [B,4,N]/[B,N,4]")
            if pred.dim() == 4:
                B, C, H, W = pred.shape
                if C == 4:
                    return pred.view(B, 4, H * W).permute(0, 2, 1).contiguous()  # [B,N,4]
                # 训练期 YOLOv8 风格：C = A*5
                if C % 5 == 0:
                    return Loss._reg_from_a5(pred)
                raise RuntimeError(f"4D pred C must be 4 or A*5, got C={C}, shape={tuple(pred.shape)}")
            raise RuntimeError(f"Unsupported pred dim={pred.dim()}, shape={tuple(pred.shape)}")

        # list/tuple：聚合各层
        if isinstance(pred, (list, tuple)):
            pieces = []
            for t in pred:
                if not torch.is_tensor(t):
                    # 忽略非张量项（例如分类图、anchors、strides 等）
                    continue
                if t.dim() == 4 and t.size(1) % 5 == 0:
                    # 只把 “回归图（A*5通道）” 收进来
                    pieces.append(Loss._reg_from_a5(t))
                elif t.dim() == 3 and t.size(1) == 4 or (t.dim() == 3 and t.size(2) == 4) or (t.dim() == 4 and t.size(1) == 4):
                    # 这几种也做一下兜底支持
                    pieces.append(Loss._to_b_n_4_any(t))
                else:
                    # 其它形状（比如分类头 [B, A*C, H, W]）直接跳过
                    pass
            if not pieces:
                raise RuntimeError("No regression features found in pred list/tuple.")
            # 按 N 维拼起来
            B = pieces[0].size(0)
            dtype = pieces[0].dtype
            device = pieces[0].device
            for p in pieces:
                if p.size(0) != B:
                    raise RuntimeError("Batch size mismatch among pyramid levels.")
                if p.dtype != dtype or p.device != device:
                    raise RuntimeError("dtype/device mismatch among pyramid levels.")
            return torch.cat(pieces, dim=1)
        raise RuntimeError(f"Unsupported pred type: {type(pred)}")

    # --------- 工具：把 reg_targets 统一成 [B,N,4] ---------
    @staticmethod
    def _target_to_b_n_4(tar, B, device, dtype) -> torch.Tensor:
        if not torch.is_tensor(tar):
            tar = torch.tensor(tar, device=device, dtype=dtype)
        else:
            tar = tar.to(device=device, dtype=dtype)

        if tar.dim() == 2 and tar.size(1) == 4:
            # [N,4] -> [1,N,4]（仅当 B==1）
            if B != 1:
                raise RuntimeError(f"reg_targets is [N,4] but batch size is {B}. "
                                   f"Please collate to [B,N,4].")
            return tar.unsqueeze(0).contiguous()

        if tar.dim() == 3:
            if tar.size(-1) == 4:   # [B,N,4]
                return tar.contiguous()
            if tar.size(1) == 4:    # [B,4,N]
                return tar.permute(0, 2, 1).contiguous()
        raise RuntimeError(f"Unsupported reg_targets shape: {tuple(tar.shape)}")

    def forward(self, pred, target):
        # 兜底：非 dict 的旧路径
        if not isinstance(target, dict):
            if torch.is_tensor(pred):
                return (pred - target).pow(2).mean()
            if isinstance(pred, (list, tuple)):
                losses = [(p - target).pow(2).mean() for p in pred if torch.is_tensor(p)]
                return sum(losses) / max(1, len(losses))
            raise RuntimeError("pred type unsupported for fallback MSE path.")

        # 取并规整预测到 [B,N,4]
        pred_reg = self._to_b_n_4_any(pred)          # <- 关键：支持 (B,65,H,W) / 多层列表
        B = pred_reg.size(0)

        # 目标也规整到 [B,N,4]
        reg_target = target.get('reg_targets')
        reg_target = self._target_to_b_n_4(reg_target, B, pred_reg.device, pred_reg.dtype)

        # 强校验 N 对齐（你的 build_targets 已经是密集 [N,4]，应一致）
        if pred_reg.size(1) != reg_target.size(1):
            raise RuntimeError(f"N mismatch: pred {tuple(pred_reg.shape)} vs target {tuple(reg_target.shape)}. "
                               f"Ensure build_targets returns dense [B,N,4] aligned with all pyramid anchors.")

        # 这里只做回归损失；分类分支（单类）忽略
        return eiou_loss(pred_reg, reg_target)
