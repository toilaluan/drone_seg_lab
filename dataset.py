import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import albumentations as A
import config as cfg
from PIL import Image 
import numpy as np
import torchvision.transforms as T
class DroneDataset(Dataset):
    def __init__(self, data_path, img_size, seg_ratio, cfg, mode = 'train'):
        self.mode = mode
        self.seg_ratio = seg_ratio
        self.img_size = img_size
        self.data_path = data_path
        self.img_list = self.load_txt(os.path.join(data_path, f'{mode}.txt'))
        self.img_aug = cfg['train_img_aug']
        self.map_aug = cfg['train_map_aug']
        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomCrop(img_size, pad_if_needed=True, fill=255),
            T.Normalize(cfg['image_mean'], cfg['image_std'])
        ])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index].rstrip()
        img_path = os.path.join(self.data_path, f'image-chips/{img_name}')
        seg_path = os.path.join(self.data_path, f'label-chips/{img_name}')
        # print(img_path)
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, self.img_size)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_path)[:,:,0]
        seg = cv2.resize(seg, (self.img_size[0]//self.seg_ratio, self.img_size[0]//self.seg_ratio))
        # seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            img = self.img_aug(image = img)['image']
            transformed = self.map_aug(image = img, mask=seg)
            img = transformed['image']
            seg = transformed['mask']
        # cv2.imwrite('debug.png', img)
        img = self.transform(img)
        # print(img)
        seg = torch.tensor(seg).long()
        return img , seg
    def load_txt(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        return data
    def visualize(self, img, seg):
        cv2.imwrite("visualize/debug.png", img)
if __name__ == "__main__":
    data_path = '/home/kc/luantt/kaggle_data/dataset-medium'
    dataset = DroneDataset(data_path=data_path, mode='train')