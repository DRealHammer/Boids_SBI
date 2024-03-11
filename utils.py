import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
import io
import pandas as pd
import numpy as np

def displayTensorImage(img, axis):
    axis.imshow(transforms.ToPILImage()(img), interpolation="nearest")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()])

valid_transform = transforms.Compose([
    transforms.ToTensor()
])

class BoidImagesDataset(Dataset):
    """Boid Images dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Simulation Directory with a csv 'params.csv' and a folder with all images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        params_name = os.path.join(root_dir, 'params.csv')
        self.params = pd.read_csv(params_name)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'img{idx}.png'
        image = io.imread(img_name)
        params = self.params.iloc[idx]
        params = np.array([params], dtype=float).reshape(-1)
        sample = {'image': image, 'params': params}

        if self.transform:
            sample = self.transform(sample)

        return sample