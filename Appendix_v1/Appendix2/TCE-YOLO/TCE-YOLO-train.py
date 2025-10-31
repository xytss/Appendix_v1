import os
import sys
import datetime
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入自定义模块
from nets.yolo import YoloBody  # 更新后的网络
from nets.yolo_training import Loss  # 使用更新后的 Loss 类
import sys
from tqdm import tqdm

def new_fit_one_epoch(model_train, model, ema, loss_fn, optimizer, epoch, num_train, num_val,
                      train_loader, val_loader, total_epochs, Cuda, fp16, scaler, save_period, save_dir, print_interval):
    model_train.train()
    running_loss = 0.0
    # 指定 file=sys.stdout 确保输出到终端，并使用 ascii=True 避免特殊字符问题
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs}",
                postfix=dict(), mininterval=0.3, file=sys.stdout, dynamic_ncols=True, ascii=True)
    for iteration, batch in enumerate(train_loader):
        images, reg_targets, cls_targets, masks = batch
        targets = {'reg_targets': reg_targets, 'cls_targets': cls_targets, 'mask': masks}
        outputs = model_train(images)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
        optimizer.step()
        running_loss += loss.item()
        if iteration % print_interval == 0:
            # 使用 flush=True 确保输出及时刷新
            print(f"Epoch {epoch} Iteration {iteration}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)
        pbar.set_postfix({'loss': running_loss/(iteration+1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)
    pbar.close()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}", flush=True)
    return avg_loss

def get_lr_scheduler(lr_decay_type, initial_lr, min_lr, total_epochs, warmup_epochs=3):
    """获取学习率调整策略"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        if lr_decay_type == "cos":
            return min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
            )
        elif lr_decay_type == "step":
            decay_rate = 0.1
            step_size = total_epochs // 3
            return initial_lr * (decay_rate ** (epoch // step_size))
        else:
            raise ValueError(f"Unsupported lr_decay_type: {lr_decay_type}")
    return lr_lambda

def set_optimizer_lr(optimizer, lr_scheduler, epoch):
    """设置优化器的学习率"""
    lr = lr_scheduler(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 由于无法从 nets.yolo_training 导入 ModelEMA，这里直接定义 ModelEMA
import copy
class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        """Exponential Moving Average (EMA) for model parameters."""
        self.ema = copy.deepcopy(model).eval()  # 创建 EMA 模型
        self.decay = decay
        self.updates = updates  # 更新次数
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """更新 EMA 参数"""
        with torch.no_grad():
            self.updates += 1
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
            state_dict = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1.0 - decay) * state_dict[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """更新 EMA 属性"""
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            setattr(self.ema, k, v)


from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import download_weights, get_classes, seed_everything, show_config, worker_init_fn

# 更新后的 fit_one_epoch 函数，支持新 batch 格式（images, reg_targets, cls_targets, masks）
def new_fit_one_epoch(model_train, model, ema, loss_fn, optimizer, epoch, num_train, num_val,
                      train_loader, val_loader, total_epochs, Cuda, fp16, scaler, save_period, save_dir, print_interval):
    model_train.train()
    running_loss = 0.0
    for iteration, batch in enumerate(train_loader):
        # 新的 batch 格式：images, reg_targets, cls_targets, masks
        images, reg_targets, cls_targets, masks = batch
        # 构造目标字典，供 Loss 类使用
        targets = {'reg_targets': reg_targets, 'cls_targets': cls_targets, 'mask': masks}
        outputs = model_train(images)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
        optimizer.step()
        running_loss += loss.item()
        if iteration % print_interval == 0:
            print(f"Epoch {epoch} Iteration {iteration}/{len(train_loader)}, Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
    # 这里可以添加验证和保存逻辑
    return avg_loss

if __name__ == "__main__":
    # 配置参数
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False
    classes_path = 'model_data/voc_classes.txt'
    model_path = ''  # 更新后的模型权重路径，如有预训练权重则指定
    input_shape = [640, 640]
    phi = 's'  # 选择模型版本
    pretrained = False

    # 训练超参数
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8
    Freeze_Train = True

    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10
    num_workers = 4

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取类别信息
    class_names, num_classes = get_classes(classes_path)

    # 创建模型
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)
    if model_path:
        print(f'Load weights {model_path}.')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # 使用更新后的 Loss 类（内部使用 EIoU 进行边界框回归损失计算）
    yolo_loss = Loss(model)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    show_config(classes_path=classes_path, model_path=model_path, input_shape=input_shape,
                Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
                Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
                Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
                momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir,
                num_workers=num_workers, num_train=num_train, num_val=num_val)

    UnFreeze_flag = False
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = optim.SGD(model.parameters(), lr=Init_lr_fit, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=True,
                                mosaic=True, mixup=True, mosaic_prob=0.5, mixup_prob=0.5)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=False,
                              mosaic=False, mixup=False, mosaic_prob=0.0, mixup_prob=0.0)
    gen = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        new_fit_one_epoch(model_train, model, ema, yolo_loss, optimizer, epoch, num_train, num_val,
                          gen, gen_val, UnFreeze_Epoch, Cuda, fp16, None, save_period, save_dir, 10)

    print("Training Finished.")
