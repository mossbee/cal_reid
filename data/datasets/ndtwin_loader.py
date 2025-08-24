import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
from data.transforms import build_transforms

class NDTwinTrainDataset(Dataset):
    """Training dataset in Re-ID format"""
    def __init__(self, train_data, transform=None):
        self.train_data = train_data
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img_path, pid, camid = self.train_data[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid

class NDTwinTestDataset(Dataset):
    """Testing dataset in verification format"""
    def __init__(self, test_pairs, transform=None):
        self.test_pairs = test_pairs
        self.transform = transform

    def __len__(self):
        return len(self.test_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.test_pairs[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label