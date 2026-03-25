
import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from config import Config

class LOLDataset(Dataset):
    def __init__(self, mode='train'):
        base_dir = Config.DATASET_ROOT
        if mode == 'train':
            self.low_dir = os.path.join(base_dir, "train", "low")
        else:
            self.low_dir = os.path.join(base_dir, "eval15", "low")
        
        if not os.path.exists(self.low_dir):
            # Fallback to searching if standard structure not found
            all_low_folders = glob.glob(os.path.join(base_dir, "**/low"), recursive=True)
            if mode == 'train':
                self.low_dir = [x for x in all_low_folders if 'our485' in x or 'train' in x][0]
            else:
                self.low_dir = [x for x in all_low_folders if 'eval15' in x or 'test' in x][0]
                
        self.high_dir = self.low_dir.replace('low', 'high')
        self.names = sorted(os.listdir(self.low_dir))
        self.mode = mode
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        low = Image.open(os.path.join(self.low_dir, name)).convert('RGB')
        high = Image.open(os.path.join(self.high_dir, name)).convert('RGB')
        low = TF.resize(low, (Config.IMG_SIZE, Config.IMG_SIZE))
        high = TF.resize(high, (Config.IMG_SIZE, Config.IMG_SIZE))
        if self.mode == 'train':
            if random.random() > 0.5: low = TF.hflip(low); high = TF.hflip(high)
            if random.random() > 0.5: low = TF.vflip(low); high = TF.vflip(high)
        return (TF.to_tensor(low) - 0.5) * 2, (TF.to_tensor(high) - 0.5) * 2
