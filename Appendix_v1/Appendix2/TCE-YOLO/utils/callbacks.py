import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        # 显式不乘 obj（若你的 DecodeBox 版本不支持该参数，自动回退）
        try:
            self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]), use_obj_score=False)
        except TypeError:
            self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    # -----------------------------
    # 工具：把 [B,4,H,W] 展平为 [B,4,N]
    # -----------------------------
    @staticmethod
    def _flatten_boxes_if_needed(dbox: torch.Tensor) -> torch.Tensor:
        if dbox.dim() == 4 and dbox.size(1) == 4:
            B, _, H, W = dbox.shape
            return dbox.view(B, 4, H * W)
        elif dbox.dim() == 3 and dbox.size(1) == 4:
            return dbox
        else:
            raise RuntimeError(f"Unexpected dbox shape {tuple(dbox.shape)}; expect [B,4,H,W] or [B,4,N].")

    # -----------------------------
    # 工具：识别 anchors/strides/cls/dbox 并组装为 decode_box 需要的 5 或 6 元组
    # 支持常见多种输出形态；缺失 cls 时用全 1 占位以先跑通评估
    # -----------------------------
    def _prepare_decode_inputs(self, raw_outputs):
        """
        Normalize model outputs into tuple expected by DecodeBox.decode_box.

        返回：
          - 5 元组: (dbox, cls, origin_cls, anchors, strides)
          - 6 元组: (dbox, cls, obj, origin_cls, anchors, strides)
        """
        if isinstance(raw_outputs, (list, tuple)):
            if len(raw_outputs) in (5, 6):
                return raw_outputs
            if len(raw_outputs) == 4:
                dbox, cls, anchors, strides = raw_outputs
                dbox = self._flatten_boxes_if_needed(dbox)
                return (dbox, cls, None, anchors, strides)
            if len(raw_outputs) == 3:
                dbox, anchors, strides = raw_outputs
                dbox = self._flatten_boxes_if_needed(dbox)
                B = dbox.size(0)
                N = dbox.size(-1)
                cls = dbox.new_ones((B, 1, N))
                return (dbox, cls, None, anchors, strides)

            tensors = [t for t in raw_outputs if torch.is_tensor(t)]
            dbox = None
            anchors = None
            strides = None
            cls = None
            for t in tensors:
                if t.dim() == 2 and t.size(-1) == 2 and anchors is None:
                    anchors = t
                elif t.dim() in (3, 4) and t.size(1) == 4 and dbox is None:
                    dbox = self._flatten_boxes_if_needed(t)
                elif t.dim() in (1, 2, 3):
                    if strides is None and (anchors is None or t.numel() == anchors.size(0) or t.shape[:1] == anchors.shape[:1]):
                        strides = t
                else:
                    if cls is None:
                        cls = t
            if dbox is None or anchors is None:
                raise RuntimeError("Cannot infer (dbox, anchors) from model outputs.")
            if cls is None:
                B, N = dbox.size(0), dbox.size(-1)
                cls = dbox.new_ones((B, 1, N))
            return (dbox, cls, None, anchors, strides)

        # 单张量：很可能是 [B,C,H,W] 的头部输出；这里无法可靠分解 -> 抛错提示你在 model.forward 就拆好
        raise RuntimeError(
            "Model outputs is a single tensor; please make the model return a tuple "
            "like (dbox, cls, origin_cls, anchors, strides)."
        )

    # ====== 打印原始输出的类型与形状，便于快速定位 ======
    def _summarize_raw_outputs(self, raw_outputs):
        if isinstance(raw_outputs, (list, tuple)):
            info = []
            for i, t in enumerate(raw_outputs):
                if torch.is_tensor(t):
                    info.append(f"[{i}] Tensor shape={tuple(t.shape)} dtype={t.dtype}")
                else:
                    info.append(f"[{i}] {type(t)}")
            return "\n".join(info)
        elif torch.is_tensor(raw_outputs):
            return f"Tensor shape={tuple(raw_outputs.shape)} dtype={raw_outputs.dtype}"
        else:
            return f"type={type(raw_outputs)}"

    # ====== 开训前快速自检（只跑一张验证图，forward→decode_box，不写文件/不做NMS） ======
    def sanity_check_once(self, idx: int = 0):
        assert len(self.val_lines) > 0, "val_lines 为空，无法做 sanity check。"
        line      = self.val_lines[idx].split()
        img_path  = line[0]

        image       = Image.open(img_path)
        image       = cvtColor(image)
        image_shape = np.array(np.shape(image)[0:2])
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # 只有当 net 是 nn.Module 才调用 eval()
        try:
            import torch.nn as nn
            if isinstance(self.net, nn.Module):
                self.net.eval()
        except Exception:
            pass  # 如果是函数或其它可调用体，直接跳过

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            raw_outputs = self.net(images)
            print(">>> [SanityCheck] raw outputs summary:\n" + self._summarize_raw_outputs(raw_outputs))

            outputs_for_decode = self._prepare_decode_inputs(raw_outputs)
            # 打印整理后的关键形状（若为 6 元组，obj 在 idx=2）
            try:
                dbox = outputs_for_decode[0]; cls = outputs_for_decode[1]
                anchors = outputs_for_decode[-2]; strides = outputs_for_decode[-1]
                print(f">>> [SanityCheck] prepared shapes: dbox={tuple(dbox.shape)}, "
                      f"cls={tuple(cls.shape) if torch.is_tensor(cls) else type(cls)}, "
                      f"anchors={tuple(anchors.shape)}, strides={tuple(getattr(strides,'shape',())) or type(strides)}")
            except Exception:
                pass

            _ = self.bbox_util.decode_box(outputs_for_decode)

        print(">>> [SanityCheck] decode_box OK（形状/展平/拼接正常）")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        image_shape = np.array(np.shape(image)[0:2])
        # 转 RGB
        image = cvtColor(image)
        # resize（可信箱）
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # +batch 维
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # 前向预测
            raw_outputs = self.net(images)
            # 将原始输出整理成 decode_box 期望的 5/6 元组
            outputs_for_decode = self._prepare_decode_inputs(raw_outputs)
            # 解码为 [B, N, 4+C]（内部已做 cls/obj 的稳健展平与对齐）
            outputs = self.bbox_util.decode_box(outputs_for_decode)
            # NMS
            results = self.bbox_util.non_max_suppression(
                outputs, self.num_classes, self.input_shape, image_shape,
                self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        if results[0] is None:
            return

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf  = results[0][:, 4]
        top_boxes = results[0][:, :4]

        # Top-K
        top_100   = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf  = top_conf[top_100]
        top_label = top_label[top_100]

        # 用上下文管理器保证文件总能正确关闭
        with open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8') as f:
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box   = top_boxes[i]
                score = str(top_conf[i])

                # FIX: 正确的 xyxy 解包顺序
                left, top, right, bottom = box

                # 这里的判断现在恒为 True（predicted_class 来自 class_names 下标）
                # 若你想更干净，可以删掉这个 if 块；我已移除
                f.write("%s %s %s %s %s %s\n" % (
                    predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))
                ))
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                # 读取图像并转换成RGB图像
                image = Image.open(line[0])
                # 获得预测txt
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                # 获得真实框txt
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
