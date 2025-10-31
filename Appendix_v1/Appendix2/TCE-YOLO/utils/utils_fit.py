import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss_total = 0
    val_loss_total = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict(), mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # 更新后的 batch 格式为：images, reg_targets, cls_targets, masks
        images, reg_targets, cls_targets, masks = batch
        # 构造目标字典，供 Loss 类使用
        targets = {'reg_targets': reg_targets, 'cls_targets': cls_targets, 'mask': masks}

        if cuda:
            images = images.cuda(local_rank)
            targets['reg_targets'] = targets['reg_targets'].cuda(local_rank)
            targets['cls_targets'] = targets['cls_targets'].cuda(local_rank)
            targets['mask'] = targets['mask'].cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, targets)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, targets)
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss_total += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss_total / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict(), mininterval=0.3)

    if ema:
        model_eval = ema.ema
    else:
        model_eval = model_train.eval()

    model_eval.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, reg_targets, cls_targets, masks = batch
        targets = {'reg_targets': reg_targets, 'cls_targets': cls_targets, 'mask': masks}
        if cuda:
            images = images.cuda(local_rank)
            targets['reg_targets'] = targets['reg_targets'].cuda(local_rank)
            targets['cls_targets'] = targets['cls_targets'].cuda(local_rank)
            targets['mask'] = targets['mask'].cuda(local_rank)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model_eval(images)
            loss_value = yolo_loss(outputs, targets)
        val_loss_total += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss_total / (iteration + 1)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss_total / epoch_step, val_loss_total / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_eval)
        print('Epoch: {}/{}'.format(epoch + 1, Epoch))
        print('Total Loss: {:.3f} || Val Loss: {:.3f}'.format(loss_total / epoch_step, val_loss_total / epoch_step_val))

        # 保存权重
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss_total / epoch_step, val_loss_total / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss_total / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
