from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input
from modules.anchors import generate_anchors
from modules.target_encoding import build_targets

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length,
                 mosaic=False, mixup=False, mosaic_prob=0.0, mixup_prob=0.0, train=True, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape  # 例如 [640, 640]
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.bbox_attrs = 5 + num_classes

        # -------------------------------
        # 生成 anchors
        feature_map_size = (self.input_shape[0] // 16, self.input_shape[1] // 16)
        stride = 16
        A = 13
        scales = np.linspace(stride, stride * 2, A).tolist()
        ratios = [1.0]
        self.anchors = generate_anchors(feature_map_size, scales, ratios, stride)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # 如果没有启用 mosaic 或 mixup，直接获取原始数据
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                line = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(line[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        # 图像预处理：转换为 float32、归一化，并调整为 (C, H, W)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))

        # 确保 box 转换为 numpy 数组并只保留一个目标框
        if len(box) > 0:
            box = box[0:1]  # 确保只有一个目标框
        box = np.array(box, dtype=np.float32) if not isinstance(box, np.ndarray) else box

        # ------------------------------#
        # 目标预处理：将原始标注转换为训练目标
        if len(box) > 0:
            gt_boxes = box[:, :4].copy()  # 保留绝对坐标
            gt_labels = box[:, 4].copy()
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int32)

        # 调用 build_targets 将真实标注转换为训练目标（回归目标、类别目标、mask）
        reg_targets, cls_targets, mask = build_targets(self.anchors, gt_boxes, gt_labels)
        targets = {'reg_targets': reg_targets, 'cls_targets': cls_targets, 'mask': mask}

        return image, targets

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = [list(map(int, b.split(','))) for b in line[1:]]

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            if len(box) > 0:
                np.random.shuffle(box)
                box = np.array(box, dtype=np.float32)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            return image_data, box

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(box) > 0:
                box = np.array(box, dtype=np.float32)
                box[:, [0, 2]] = w - box[:, [2, 0]]
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge(
            (cv2.LUT(hue_channel, lut_hue), cv2.LUT(sat_channel, lut_sat), cv2.LUT(val_channel, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        if len(box) > 0:
            np.random.shuffle(box)
            box = np.array(box, dtype=np.float32)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    # 新增：简单实现的 Mosaic 和 MixUp
    def get_random_data_with_Mosaic(self, annotation_lines, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        images = []
        boxes = []
        for line in annotation_lines:
            img, box = self.get_random_data(line, input_shape, random=self.train)
            images.append(img)
            boxes.append(box)
        mosaic_image = np.mean(np.stack(images, axis=0), axis=0).astype(np.uint8)
        mosaic_boxes = np.concatenate(boxes, axis=0)
        return mosaic_image, mosaic_boxes[:1]  # 确保只保留一个框

    def get_random_data_with_MixUp(self, image, box, image2, box2):
        new_image = ((np.array(image, dtype=np.float32) + np.array(image2, dtype=np.float32)) / 2).astype(np.uint8)
        if len(box) == 0:
            new_boxes = box2
        elif len(box2) == 0:
            new_boxes = box
        else:
            new_boxes = np.concatenate([box, box2], axis=0)
        return new_image, new_boxes[:1]  # 只保留一个目标框


# 更新后的 collate 函数：将每个样本返回的 (image, targets) 组合成 batch
def yolo_dataset_collate(batch):
    images = []
    reg_targets_list = []
    cls_targets_list = []
    masks_list = []
    for i, (img, targets) in enumerate(batch):
        images.append(img)
        reg_targets_list.append(targets['reg_targets'])
        cls_targets_list.append(targets['cls_targets'])
        masks_list.append(targets['mask'])
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    reg_targets = torch.from_numpy(np.array(reg_targets_list)).type(torch.FloatTensor)
    cls_targets = torch.from_numpy(np.array(cls_targets_list)).type(torch.FloatTensor)
    masks = torch.from_numpy(np.array(masks_list)).type(torch.FloatTensor)
    return images, reg_targets, cls_targets, masks
