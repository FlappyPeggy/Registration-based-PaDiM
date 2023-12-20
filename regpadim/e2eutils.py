import os
import glob
from PIL import Image
import torch.utils.data as data
import random
import torch
import torchvision.transforms.functional as TVF
import torchvision.transforms as t
import numpy as np
from torch.nn import functional as F


def gkern(size):
    std = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    n = torch.arange(0, size) - (size - 1.0) / 2.0
    gker1d = torch.exp(-n ** 2 / 2 * std ** 2)
    gker2d = torch.outer(gker1d, gker1d)
    return (gker2d / gker2d.sum())[None, None], size // 2

def get_data_transforms(size, crop_h, rot=2, totensor=True):
    if totensor:
        data_transforms = t.Compose([
            t.Resize((size[1]*2, size[0]*2)),
            t.ToTensor(),
            t.RandomRotation(rot),
            RCrop(crop_h, rand_factor=0.01),
            t.Resize((size[1], size[0])),
            # t.CenterCrop(isize),
            t.ColorJitter(0.1, 0.1),
            t.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
    else:
        data_transforms = t.Compose([
            t.Resize((size[1]*2, size[0]*2)),
            t.RandomRotation(rot),
            RCrop(crop_h, rand_factor=0.01),
            t.Resize((size[1], size[0])),
            t.ColorJitter(0.1, 0.1),
            t.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
    return data_transforms

def fused_map(fs_list, out_size):
    a_map = F.interpolate(fs_list[0], size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
    b_map = F.interpolate(fs_list[1], size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)
    c_map = F.interpolate(fs_list[2], size=(out_size[1], out_size[0]), mode='bilinear', align_corners=True)

    return torch.cat([a_map, b_map, c_map], dim=1)

def fuse_param(m1, m2, c1, c2=0, w1=0.5):
    if w1 == 1:
        return m1, c1
    delta = m1 - m2
    if isinstance (delta, torch.Tensor):
        cov_ = c2 + w1 * torch.matmul(delta[:, :, None], delta[:, None])
    elif isinstance (delta, np.ndarray):
        cov_ = c2 + w1 * np.matmul(delta[:, :, None], delta[:, None])
    else:
        raise "input should be ndarray or tensor"
    cov = c1 * w1 + cov_ * (1 - w1)
    mean = m1 * w1 + m2 * (1 - w1)
    return mean, cov

class RCrop(torch.nn.Module):
    def __init__(self, roi, rand_factor=0.):
        super().__init__()
        self.top = roi[0]
        self.left = roi[2]
        self.height = roi[1]-roi[0]
        self.width = roi[3]-roi[2]
        self.rand_shift = rand_factor

    def forward(self, img):
        if self.height<1:
            self.height = int(img.size(1) * self.height)
            self.width = int(img.size(2) * self.width)
            self.top = int(img.size(1) * self.top)
            self.left = int(img.size(2) * self.left)

        if self.rand_shift and self.rand_shift<1:
            self.rand_shift = int(min(self.height, self.width) * self.rand_shift)

        return TVF.crop(img,
                        self.top+torch.randint(-self.rand_shift, self.rand_shift, (1,)),
                        self.left+torch.randint(-self.rand_shift, self.rand_shift, (1,)),
                        self.height, self.width)

REFERENCE_IDX = 500
class DataLoader(data.Dataset):
    def __init__(self, folder, transform, test=False):
        self.dir = folder
        self.transform = transform
        self.test = test
        self.samples1, self.samples2 = self.get_all_samples()
        self.ref_idx = min(len(self.samples1)//2, REFERENCE_IDX)
            
    def get_all_samples(self):
        shuffle1, shuffle2 = [], []
        if self.test:
            all_frame_from_one_dir = glob.glob(os.path.join(self.dir, "*"))
            return all_frame_from_one_dir.copy(), None

        root = os.path.join(self.dir, 'frames')
        for dir_name in os.listdir(root):
            dir_path = os.path.join(root, dir_name)
            all_frame_from_one_dir = glob.glob(os.path.join(dir_path, "*"))
            random.shuffle(all_frame_from_one_dir)
            shuffle1 += all_frame_from_one_dir.copy()
            random.shuffle(all_frame_from_one_dir)
            shuffle2 += all_frame_from_one_dir.copy()
            if dir_name in ['8', '9','11','15']:
                random.shuffle(all_frame_from_one_dir)
                shuffle1 += all_frame_from_one_dir.copy()
                random.shuffle(all_frame_from_one_dir)
                shuffle2 += all_frame_from_one_dir.copy()
            if dir_name in [ '9']:
                random.shuffle(all_frame_from_one_dir)
                shuffle1 += all_frame_from_one_dir.copy()
                random.shuffle(all_frame_from_one_dir)
                shuffle2 += all_frame_from_one_dir.copy()
                           
        return shuffle1, shuffle2
        
    def __getitem__(self, index):
        img, reg = Image.open(self.samples1[index]).convert('RGB'), Image.open(self.samples1[self.ref_idx]).convert('RGB')
        return self.transform(img), self.transform(reg)
        
    def __len__(self):
        return len(self.samples1)


class AnchorLoader(data.Dataset):
    def __init__(self, folder, transform, iter=10):
        self.dir = folder
        self.transform = transform
        self.samples1 = self.get_all_samples()
        self.anchors = self.get_all_anchors()
        self.iter = iter

    def get_all_samples(self):
        shuffle1 = []
        for dir_name in os.listdir(self.dir):
            dir_path = os.path.join(self.dir, dir_name)
            all_frame_from_one_dir = glob.glob(os.path.join(dir_path, "*"))
            all_frame_from_one_dir.reverse()
            shuffle1 += all_frame_from_one_dir.copy()

        return shuffle1

    def get_all_anchors(self):
        return glob.glob(os.path.join(self.dir, "*")).copy()

    def __getitem__(self, index):
        return self.transform(Image.open(self.anchors[index//self.iter]).convert('RGB'))

    def __len__(self):
        return len(self.anchors)*self.iter
