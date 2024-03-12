import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np

def displayTensorImage(img, axis):
    axis.imshow(transforms.ToPILImage()(img), interpolation="nearest")

class ZeroChannelTransform():
    def __init__(self, idx, p=0.5):
        self.idx = idx
        self.p = p

    def __call__(self, x):
        x[:, self.idx] = x[:, self.idx] * (torch.rand(len(x)) > self.p)[:, None]
        return x


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    ZeroChannelTransform(2)
    ])

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

        img_name = f'{self.root_dir}/images/img{idx}.png'
        image = Image.open(img_name)
        params = self.params.iloc[idx]
        params = np.array([params], dtype=np.float32).reshape(-1)

        if self.transform:
            image = self.transform(image)

        return [params, image]
        #return {'image': image, 'params': params}