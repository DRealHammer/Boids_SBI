from flows import LightningRealNVP
from autoencoders import LightningVAE, LightningSummaryFC, LightningSummaryConv
from utils import BoidImagesDataset, train_transform, simple_train_transform, buildDataTranform
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse

import torch
from torch.utils.data import DataLoader
import lightning as L

import os
import re


'''
python train_flow.py
--model_folder <folder>
--train_folder <folder>
--valid_folder <folder>
--max_minutes <minutes>
--max_hours <hours>
--boids
--obstacles
--separation
--coherence
--alignment
--avoidance
--visual_range
--avoid_range

'''

def load_model(model_folder:str, model_class:L.LightningModule):

    encoder_dir = os.path.join(model_folder, 'encoder')

    ckpt_file = [file for file in os.listdir(encoder_dir) if file.endswith('ckpt')]
    if len(ckpt_file) != 1:
        print('Multiple or no ckpt model file found!')
        exit()
    ckpt_file = os.path.join(encoder_dir, ckpt_file[0])

    hparams_file = os.path.join(encoder_dir, 'hparams.yaml')

    model = model_class.load_from_checkpoint(checkpoint_path=ckpt_file, hparams_file=hparams_file)

    return model


models_dir = 'models'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a specified summary network to use as a summary network'
    )

    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--train_folder', required=True)
    parser.add_argument('--valid_folder')
    parser.add_argument('--max_minutes')
    parser.add_argument('--max_hours')

    parser.add_argument('--boids', action='store_true')
    parser.add_argument('--obstacles', action='store_true')
    parser.add_argument('--separation', action='store_true')
    parser.add_argument('--coherence', action='store_true')
    parser.add_argument('--alignment', action='store_true')
    parser.add_argument('--avoidance', action='store_true')
    parser.add_argument('--visual_range', action='store_true')
    parser.add_argument('--avoid_range', action='store_true')
    parser.add_argument('--hidden_size') # default 200
    parser.add_argument('--blocks') # default 5

    args = parser.parse_args()

    model_folder = os.path.join('models', args.model_folder)

    if not os.path.exists(model_folder):
        print(f'Provided folder {model_folder} does not seem to exist!')
        exit()

    if not os.path.exists(args.train_folder):
        print(f'Provided folder {args.train_folder} does not seem to exist!')
        exit()

    if args.valid_folder is not None and not os.path.exists(args.valid_folder):
        print(f'Provided folder {args.valid_folder} does not seem to exist!')
        exit()

    training_time = 0

    if args.max_minutes is not None:
        training_time += int(args.max_minutes)

    if args.max_hours is not None:
        training_time += int(args.max_hours) * 60

    if training_time == 0:
        print(f'Provided no training time.')
        exit()


    data_indices = torch.tensor([
        args.boids,                            
        args.obstacles,                        
        args.separation,                        
        args.coherence,                        
        args.alignment,                        
        args.avoidance,                        
        args.visual_range,                        
        args.avoid_range                                                
    ])

    if not torch.any(data_indices):
        data_indices = torch.logical_not(data_indices)
    
    data_dim = torch.count_nonzero(data_indices).item()
    feature_code = bin(torch.sum(data_indices * 2**torch.arange(7, -1, -1), dtype=int).item())[2:]

    feature_code = '0' * (8 - len(feature_code)) + feature_code

    # load the encoder

    cond_size = 0

    if args.model_folder.lower().startswith('ae'):
        encoder_class = LightningVAE
        cond_size = int(re.findall(r'\d+', args.model_folder)[0])

    elif args.model_folder.lower().startswith('fc'):
        encoder_class = LightningSummaryFC
        cond_size = 200

    elif args.model_folder.lower().startswith('conv'):
        encoder_class = LightningSummaryConv
        # TODO
        cond_size = 0

    else:
        print('No known model found in folder name')
        exit()

    encoder = load_model(model_folder=model_folder, model_class=encoder_class)

    # create folder structure
    flow_name = f'flow{feature_code}'
    flow_folder = os.path.join(model_folder, flow_name)
    previous_flows = [name for name in os.listdir(model_folder) if name == flow_name]   

    if len(previous_flows) != 0:
        print('There already exists a flow!')
        exit()

    os.mkdir(flow_folder)

    # create flow model

    hidden_dim = args.hidden_size
    if hidden_dim is None:
        hidden_dim = 200

    blocks = args.blocks
    if blocks is None:
        blocks = 5

    flow = LightningRealNVP(input_size=data_dim, hidden_size=hidden_dim, blocks=blocks, condition_size=cond_size, encoder=encoder)

    # load datasets

    train_dataset = BoidImagesDataset(args.train_folder, transform=buildDataTranform(random_flip=True, zero_channel=True, blurring=True), use_index=data_indices)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, persistent_workers=True)


    valid_dataset = None
    valid_loader = None

    if args.valid_folder is not None:
        valid_dataset = BoidImagesDataset(args.valid_folder, transform=buildDataTranform(), use_index=data_indices)
        valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False, num_workers=4, persistent_workers=True)


        validation_checkpoint = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.2f}')
        trainer = L.Trainer(max_time={'minutes': training_time % 60, 'hours': training_time // 60}, callbacks=[validation_checkpoint], default_root_dir=flow_folder)
        trainer.fit(flow, train_loader, valid_loader)

    else:

        trainer = L.Trainer(max_time={'minutes': training_time % 60, 'hours': training_time // 60}, default_root_dir=flow_folder)
        trainer.fit(flow, train_loader)

