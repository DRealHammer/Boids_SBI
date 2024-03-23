import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np

def displayTensorImage(img, axis):
    axis.imshow(transforms.ToPILImage()(img), interpolation="nearest")

class RandomZeroChannel():
    def __init__(self, idx, p=0.5):
        self.idx = idx
        self.p = p

    def __call__(self, x):
        x[:, self.idx] = x[:, self.idx] * (torch.rand(len(x)) > self.p)[:, None]
        return x


def buildDataTranform(random_flip=False, blurring=False, zero_channel=False, apply_to_tensor_fct=None):

    t = []

    if random_flip:
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.RandomVerticalFlip(p=0.5))

    if blurring:
        t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5))

    t.append(transforms.ToTensor())

    if zero_channel:
        t.append(RandomZeroChannel(2, p=0.5))

    if apply_to_tensor_fct is not None:
        t.append(apply_to_tensor_fct)

    return transforms.Compose(t)
    

simple_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    RandomZeroChannel(2, p=0.5)
    ])

valid_transform = transforms.Compose([
    transforms.ToTensor()
])

class BoidImagesDataset(Dataset):
    """Boid Images dataset."""

    def __init__(self, root_dir, transform=None, use_index=np.ones(8, dtype=bool), normalize=False):
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

        self.use_index = use_index
        self.normalize = normalize


        self.params_mean = np.array([275, 2.5, 2.5, 2, 4, 20, 32, 2, 4])
        self.params_scale = np.array([500, 5, 4, 4, 8, 4, 64, 64/8.0]) - self.params_mean

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'{self.root_dir}/images/img{idx}.png'
        image = Image.open(img_name)
        params = self.params.iloc[idx]
        params = np.array([params], dtype=np.float32).reshape(-1)[self.use_index]

        if self.normalize:
            params = (params - self.params_mean ) / self.params_scale

        if self.transform:
            image = self.transform(image)

        return [params, image]
        #return {'image': image, 'params': params}
    
import re
def get_folder_feature_index(foldername):
    indices = re.findall(r'\d+')[0]
    res = torch.zeros(len(indices), dtype=bool)
    for i, index in enumerate(indices):
        res[i] = index == '1'

    return res