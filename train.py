from dataset import DroneDataset
import torch
import torch.nn as nn
import numpy as np
import cv2
from unet import UNet
from torch.utils.data import DataLoader
from run import *
from deeplabv3 import DeeplabV3plus
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, MaskFormerForInstanceSegmentation
import albumentations as A
cfg = {
    'data_path': '/home/kc/luantt/kaggle_data/dataset-medium',
    'batch_size' : 16,
    'epochs': 40,
    'device': 'cuda',
    'lr': 1e-4,
    'img_size': (512,512),
    'seg_ratio': 4,
    'n_classes': 6,
    'model_name': 'segformer_b0',
    'use_transformers': True,
    'log_interval': 10,
    'image_mean': (0.485,0.456,0.406),
    'image_std' : (0.229,0.224,0.225),
}
cfg['train_img_aug'] = A.Compose(
    (
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p = 0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(),
        A.GaussNoise(3),
    ) 
    )
cfg['train_map_aug'] = A.Compose(
    [
        # A.RandomScale((1, 1.5)),
        # A.RandomCrop(height=cfg['img_size'][0], width=cfg['img_size'][1], always_apply=True),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.CLAHE (clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ]
    )
# model = DeeplabV3plus(num_classes=6)
state_dict = torch.load('/home/kc/luantt/drone_segmentation/checkpoints/segformer/last.pth')
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=6, ignore_mismatched_sizes=True)
model.load_state_dict(state_dict)
model = nn.DataParallel(model)
train_data = DroneDataset(cfg['data_path'], img_size=cfg['img_size'], seg_ratio = cfg['seg_ratio'], mode='train', cfg=cfg)
val_data = DroneDataset(cfg['data_path'], mode='valid', img_size=cfg['img_size'], seg_ratio = cfg['seg_ratio'], cfg=cfg)
train_loader = DataLoader(train_data, cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, 1, num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss(ignore_index=255)
opt = torch.optim.Adam(model.parameters(), lr = cfg['lr'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,0.85)
start_train(model, train_loader, val_loader, opt, criterion, scheduler, cfg)
