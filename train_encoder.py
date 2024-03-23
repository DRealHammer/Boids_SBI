from autoencoders import LightningVAE, LightningSummaryFC, LightningSummaryConv
from utils import BoidImagesDataset, buildDataTranform

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse
import os

torch.set_float32_matmul_precision('medium')

'''
python train_encoder.py
--model ae200/ae23/fc4/fc8/conv4/conv8
--train_folder <folder>
--valid_folder <folder>
--max_minutes <minutes>
--max_hours <hours>

'''

models_dir = 'models'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a specified summary network to use as a summary network'
    )

    parser.add_argument('--model', required=True)
    parser.add_argument('--train_folder', required=True)
    parser.add_argument('--valid_folder')
    parser.add_argument('--max_minutes')
    parser.add_argument('--max_hours')

    args = parser.parse_args()

    if not os.path.exists(args.train_folder):
        print(f'Provided folder {args.train_folder} does not seem to exist!')
        exit()

    if args.valid_folder is not None and not os.path.exists(args.valid_folder):
        print(f'Provided folder {args.train_folder} does not seem to exist!')
        exit()

    training_time = 0

    if args.max_minutes is not None:
        training_time += int(args.max_minutes)

    if args.max_hours is not None:
        training_time += int(args.max_hours) * 60

    if training_time == 0:
        print(f'Provided no training time.')
        exit()

    
    # process arguments


    model:L.LightningModule = None
    model_name:str = ''

    # autoencoder
    if args.model.lower().startswith('ae'):
        if len(args.model) == 2:
            latent_size = 500
        else:
            latent_size = int(args.model[2:])
        
        model = LightningVAE(3, latent_size)
        model_name = f'AE{latent_size}'

    # fully connected
    elif args.model.lower().startswith('fc'):
        if len(args.model) == 2:
            predict_size = 8
        else:
            predict_size = int(args.model[2:])
        
        model = LightningSummaryFC(3*64*64, 2, 200, predict_size)
        model_name = f'FC{predict_size}'

    # convolutional
    elif args.model.lower().startswith('conv'):
        if len(args.model) == 2:
            predict_size = 8
        else:
            predict_size = int(args.model[2:])
        
        model = LightningSummaryConv(3, predict_size)
        model_name = f'CONV{predict_size}'


    # create folder structure
    if models_dir not in os.listdir():
        os.mkdir(models_dir)

    previous_versions = [name for name in os.listdir(models_dir) if name.startswith(model_name)]   

    if len(previous_versions) > 0:
        model_name = f'{model_name}_{len(previous_versions)}'

    model_folder = os.path.join(models_dir, model_name)
    os.mkdir(model_folder)

    os.mkdir(os.path.join(model_folder, 'encoder'))


    # load datasets

    train_dataset = BoidImagesDataset('boid_data', transform=buildDataTranform(random_flip=True, zero_channel=True, blurring=True))
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, persistent_workers=True)


    valid_dataset = None
    valid_loader = None

    if args.valid_folder is not None:
        valid_dataset = BoidImagesDataset(args.valid_folder, transform=buildDataTranform())
        valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False, num_workers=4, persistent_workers=True)


        validation_checkpoint = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.2f}')
        trainer = L.Trainer(max_time={'minutes': training_time % 60, 'hours': training_time // 60}, callbacks=[validation_checkpoint], default_root_dir=model_folder)
        trainer.fit(model, train_loader, valid_loader)

    else:

        trainer = L.Trainer(max_time={'minutes': training_time % 60, 'hours': training_time // 60}, default_root_dir=model_folder)
        trainer.fit(model, train_loader)