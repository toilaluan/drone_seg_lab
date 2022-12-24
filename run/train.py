import torch
from .metrics import *
from tqdm import tqdm
import numpy
import wandb
import os
wandb.init(project="Drone Semantic Segmentation")
def start_train(model, train_loader, val_loader, opt, criterion, scheduler, cfg):
    wandb.config = cfg
    device = torch.device(cfg['device'])
    model.train()
    model.to(device)
    epochs = cfg['epochs']
    best_miou = 0.
    for epoch in range(cfg['epochs']):
        print("START TRAINING EPOCH", epoch)
        epoch_loss = 0
        for i, (img, map) in enumerate(train_loader):
            # print(i)
            img = img.to(device)
            map = map.to(device)
            output = model(img)
            if cfg['use_transformers']:
                output = output.logits
            loss = criterion(output, map)
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_loss+=loss.item()
            if i % cfg['log_interval'] == 0:
                print(f'Epoch: [{epoch}/{epochs}], Lr: [{scheduler.get_last_lr()}], Step: [{i}/{len(train_loader)}], Epoch Loss : {epoch_loss/(i+1)}')
        mious, mloss = start_validate(model, val_loader, criterion, cfg, device)
        scheduler.step()
        print(f'METRICS: mIOU = {mious.mean()}, mLoss = {mloss}')
        print(f'IOU for each class: {mious}')
        wandb.log({'mIOU': mious.mean(), 'mLoss': mloss})
        if not os.path.exists('checkpoints/'+cfg['model_name']):
            os.makedirs('checkpoints/'+cfg['model_name'])
        if (mious.mean() > best_miou):
            torch.save(model.state_dict(), "checkpoints/{}/best_miou.pth".format(cfg['model_name']))
        torch.save(model.state_dict(), "checkpoints/{}/last.pth".format(cfg['model_name']))
def start_validate(model, val_loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.
    total_intersect = torch.zeros((cfg['n_classes']))
    total_union = torch.zeros((cfg['n_classes']))
    ious = torch.zeros((cfg['n_classes'],))
    i = 0
    for img, map in tqdm(val_loader):
        img = img.to(device)
        map = map.to(device)
        with torch.no_grad():
            out = model(img)
            if cfg['use_transformers']:
                out = out.logits
            pred = out.argmax(dim=1)
            loss = criterion(out, map)
            pred = pred.cpu()
            map = map.cpu() 
            total_loss += loss.item()
            # print(pred.shape, map.shape)
            intersect, union,_,_ = intersect_and_union(pred, map, num_classes=cfg['n_classes'], ignore_index=255)
            total_intersect += intersect
            total_union += union
    ious = total_intersect/total_union
    # ious = ious.mean()
    model.train()
    return ious, total_loss/len(val_loader)


