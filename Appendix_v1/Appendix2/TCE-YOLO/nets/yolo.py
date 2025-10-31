import torch
import torch.nn as nn
from nets.backbone import MobileViTBackbone, Conv
from nets.modules import DFL

def list_to_tensor(data, device, dtype):
    """
    递归式地将 data（可能是多层嵌套 list/tuple/Tensor/数字）转换为 Tensor。
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (list, tuple)):
        if all(isinstance(x, torch.Tensor) for x in data):
            try:
                return torch.stack([x.to(device, dtype) for x in data])
            except RuntimeError:
                pass
        converted_list = [list_to_tensor(x, device, dtype) for x in data]
        if all(isinstance(x, torch.Tensor) for x in converted_list):
            try:
                return torch.stack(converted_list)
            except RuntimeError:
                pass
        return converted_list
    else:
        return torch.tensor(data, device=device, dtype=dtype)

class DepthwiseConv(nn.Module):
    """深度可分离卷积模块，自动适配 groups 避免通道数不匹配"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=self.groups, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class DC2(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, shortcut=False):
        super().__init__()
        self.shortcut = shortcut
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(*[
            DepthwiseConv(out_channels, out_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.adjust_channels(x)
        return x + self.blocks(x) if self.shortcut else self.blocks(x)

class YoloBody(nn.Module):
    """
    按方案A重构的头部：
      - 明确区分回归与分类分支
      - 回归输出：na * 5（ltrb×4 + obj×1）
      - 分类输出：na * num_classes
      - forward 返回 (pred_reg_P4, pred_cls_P4, origin_cls_P5)
    这样与你现在的 wrapped_net / callbacks 对齐。
    """
    def __init__(self, input_shape, num_classes, phi, pretrained=False, use_dfl=True, na: int = 13):
        super(YoloBody, self).__init__()
        self.backbone = MobileViTBackbone(pretrained=pretrained)

        # 保存关键配置
        self.num_classes = num_classes
        self.na = na
        self.use_dfl = use_dfl
        if self.use_dfl:
            self.dfl = DFL(16)  # 目前未启用 DFL 回归解码，这里保留占位

        # 调整层定义
        self.p4_adjust_first = nn.Conv2d(768, 1024, kernel_size=1, bias=False)
        self.p4_adjust_merge = nn.Conv2d(1280, 1024, kernel_size=1, bias=False)
        self.p5_adjust = nn.Conv2d(1536, 1024, kernel_size=1, bias=False)
        self.p3_adjust = nn.Conv2d(1152, 512, kernel_size=1, bias=False)
        self.p3_adjust_extra = Conv(512, 512, 1, 1)

        # 颈部（保持你的原始结构）
        self.conv3_for_upsample1 = DC2(1024, 1024, 3, shortcut=False)
        self.conv3_for_upsample2 = DC2(512, 256, 3, shortcut=False)
        self.down_sample1 = Conv(256, 256, 3, 2)
        self.conv3_for_downsample1 = DC2(1024, 512, 3, shortcut=False)
        self.down_sample2 = Conv(512, 512, 3, 2)
        self.conv3_for_downsample2 = DC2(1024, 1024, 3, shortcut=False)
        self.sppf_proj = Conv(1024, 1024, 1, 1)

        # ---------------- 预测头（回归与分类分开定义） ----------------
        # 回归分支：隐藏通道用 64，最终输出 na*5（方便你现有 wrapped_net 里的 A*5 逻辑）
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                Conv(x, 64, 3),
                Conv(64, 64, 3),
                nn.Conv2d(64, self.na * 5, 1)   # 4个距离 + 1个obj
            ) for x in [256, 512, 1024]
        ])
        # 分类分支：隐藏通道用 64，最终输出 na*num_classes
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                Conv(x, 64, 3),
                Conv(64, 64, 3),
                nn.Conv2d(64, self.na * self.num_classes, 1)
            ) for x in [256, 512, 1024]
        ])

    def forward(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = list_to_tensor(x, device=device, dtype=torch.float32)
        if isinstance(x, list):
            raise ValueError("Input x is still a list, cannot proceed with forward.")

        # backbone 输出
        backbone_outputs = self.backbone(x)
        if len(backbone_outputs) == 3:
            feat1, feat2, sppf_out = backbone_outputs
            feat3 = None
        elif len(backbone_outputs) == 4:
            feat1, feat2, feat3, sppf_out = backbone_outputs
        else:
            raise ValueError(f"Unexpected number of outputs from backbone: {len(backbone_outputs)}")

        # 对齐空间分辨率
        if feat3 is not None and (feat2.shape[2:] != feat3.shape[2:]):
            feat3 = nn.functional.interpolate(feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False)

        # 颈部整合
        sppf_out = self.sppf_proj(sppf_out)
        if feat3 is not None:
            # P4, P3, P5
            P4 = torch.cat([feat2, feat3], dim=1)
            P4 = self.p4_adjust_first(P4)
            P4 = self.conv3_for_upsample1(P4)

            if feat1.shape[2:] != P4.shape[2:]:
                feat1 = nn.functional.interpolate(feat1, size=P4.shape[2:], mode="bilinear", align_corners=False)
            P3 = torch.cat([feat1, P4], dim=1)
            P3 = self.p3_adjust(P3)
            P3 = self.p3_adjust_extra(P3)
            P3 = self.conv3_for_upsample2(P3)

            P3_downsample = self.down_sample1(P3)
            if P3_downsample.shape[2:] != P4.shape[2:]:
                P3_downsample = nn.functional.interpolate(P3_downsample, size=P4.shape[2:], mode="bilinear", align_corners=False)
            P4 = torch.cat([P3_downsample, P4], dim=1)
            P4 = self.p4_adjust_merge(P4)
            P4 = self.conv3_for_downsample1(P4)

            P4_downsample = self.down_sample2(P4)
            P5 = torch.cat([P4_downsample, sppf_out], dim=1)
            P5 = self.p5_adjust(P5)
            P5 = self.conv3_for_downsample2(P5)

            # ---------------- 头部输出（分支分开） ----------------
            # 常规设定：640 输入下 P3≈80×80, P4≈40×40, P5≈20×20
            # 你的评估/包装在 40×40（P4）上工作，因此这里挑 P4 作为主输出
            reg_p3 = self.reg_heads[0](P3)  # 备用
            cls_p3 = self.cls_heads[0](P3)  # 备用

            reg_p4 = self.reg_heads[1](P4)  # 主回归输出：B × (na*5) × 40 × 40
            cls_p4 = self.cls_heads[1](P4)  # 主分类输出：B × (na*C) × 40 × 40

            reg_p5 = self.reg_heads[2](P5)  # 备用
            cls_p5 = self.cls_heads[2](P5)  # 作为 origin_cls（可选）

            # 与你现有 wrapped_net / callbacks 对齐：
            # 只返回 (P4回归, P4分类, P5分类) 三个张量
            return (reg_p4, cls_p4, cls_p5)
        else:
            # 没有 feat3 的降级路径（不常见）
            # 直接在 feat2 上构造 P4 风格的输出
            P4 = self.conv3_for_upsample1(self.p4_adjust_first(feat2))
            reg_p4 = self.reg_heads[1](P4)
            cls_p4 = self.cls_heads[1](P4)
            # origin_cls 用一个同分辨率的占位
            origin_cls = cls_p4
            return (reg_p4, cls_p4, origin_cls)


if __name__ == "__main__":
    # 配置测试参数
    input_shape = [640, 640]
    num_classes = 1  # 如用于单类别检测
    phi = 's'
    model = YoloBody(input_shape, num_classes, phi, pretrained=False, use_dfl=True, na=13)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    with torch.no_grad():
        reg_p4, cls_p4, cls_p5 = model(dummy_input)
    print("reg_p4:", tuple(reg_p4.shape), "cls_p4:", tuple(cls_p4.shape), "cls_p5:", tuple(cls_p5.shape))
