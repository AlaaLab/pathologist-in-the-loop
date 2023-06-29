from __future__ import print_function

import argparse
# importing the libraries
import os
import random
import sys
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import  glob
import time
import albumentations
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
### Internal Imports
from models.models import Myresnext50, Myresnext50_algin
from train.train_scoring_function import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import Img_DataLoader_pair as Img_DataLoader

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob

def main(args):




    X_train = glob.glob(os.path.join(args.train_dir,'*/*'))

    labels = [x.split('/')[-2] for x in X_train]
    categories = set(labels)

    categories = list(categories)
    categories.sort()

    X_train_anno = glob.glob(os.path.join(args.train_anno_dir, '*/*/*.png'))

    X_val = glob.glob(os.path.join(args.val_dir,'*/*'))
    X_val_anno = glob.glob(os.path.join(args.val_anno_dir, '*/*/*.png'))
    df = pd.DataFrame({
        'dirs': X_val_anno,
        'categories': [x.split('/')[-3] for x in X_val_anno]
    })
    grouped = df.groupby('categories')
    df_sampled = pd.concat([d.loc[random.sample(list(d.index), int(len(list(d.index))*args.ratio))] for _, d in grouped]).reset_index(drop=True)
    X_val_anno = df_sampled['dirs'].tolist()
    X_train_anno = df_sampled['dirs'].tolist()

    df = pd.DataFrame(categories, columns=['categories'])# converting type of columns to 'category'
    df['categories'] = df['categories'].astype('category')# Assigning numerical values and storing in another column
    df['categories_Cat'] = df['categories'].cat.codes



    enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(df[['categories_Cat']]).toarray())# merge with main df bridge_df on key values
    df = df.join(enc_df)



    # load model

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # Interesting! This worked for no reason haha
    if args.input_model == 'ResNeXt50':
        resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
        My_model = Myresnext50(my_pretrained_model=resnext50_pretrained, num_classes=len(categories))

        if checkpoint_PATH is not None:
            checkpoint = torch.load(checkpoint_PATH)

            from collections import OrderedDict
            def remove_data_parallel(old_state_dict):
                new_state_dict = OrderedDict()

                for k, v in old_state_dict.items():
                    name = k[7:]  # remove `module.`

                    new_state_dict[name] = v

                return new_state_dict

            checkpoint = remove_data_parallel(checkpoint['model_state_dict'])

            #My_model.load_state_dict(checkpoint, strict=True)
            my_extended_model = Myresnext50_algin(my_pretrained_model= resnext50_pretrained)
            my_extended_model.load_state_dict(checkpoint, strict=True)
        else:
            my_extended_model = Myresnext50_algin(my_pretrained_model= resnext50_pretrained)


    #### train_iteration_1

    #

    transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                     std=(0.229, 0.224, 0.225)),

        ]
    )




    trainer = trainer_classification(train_image_files=X_train, train_image_files_anno = X_train_anno,
                                     validation_image_files = X_val,
                                     validation_image_files_anno=X_val_anno,
                                     model=my_extended_model,
                                     img_transform=transform_pipeline, init_lr=args.init_lr,
                                     lr_decay_every_x_epochs=args.lr_decay_every_x_epochs,
                                     weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma,
                                     df=df, graph_loss = args.graph_loss,
                                     save_checkpoints_dir=args.save_checkpoints_dir,
                                     ratio = args.ratio,
                                     weakfeedback = False)

    My_model = trainer.train(my_extended_model)


# Training settings
parser = argparse.ArgumentParser(description='Configurations for Model training')
parser.add_argument('--train_dir', type=str,
               default='.',
                    help='train data directory')

parser.add_argument('--train_anno_dir', type=str,
               default='.',
                    help='')

parser.add_argument('--val_dir', type=str,
               default='.',
                    help='val data directory')

parser.add_argument('--val_anno_dir', type=str,
               default='.',
                    help='val data directory')

parser.add_argument('--ratio', type=float,
               default=1.0,
                    help='val data directory')
parser.add_argument('--input_model', type=str,
                    default='ResNeXt50',
                    help='input model, the defulat is the pretrained model')


parser.add_argument('--pretrained', type=bool,
                    default=True,
                    help='the defulat is the pretrained model')

parser.add_argument('--init_lr', type=float,
                    default=0.0001,
                    help='learning rate')

parser.add_argument('--weight_decay', type=float,
                    default=0.0005,
                    help='weight decay')

parser.add_argument('--gamma', type=float,
                    default=0.1,
                    help='gamma')

parser.add_argument('--epochs', type=float,
                    default=30,
                    help='epoch number')

parser.add_argument('--batch_size', type=int,
                    default=64,
                    help='epoch number')

parser.add_argument('--lr_decay_every_x_epochs', type = int,
                    default=5,
                    help='learning rate decay per X step')

parser.add_argument('--save_checkpoints_dir', type = str,
                    default='.',
                    help='save dir')



args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    print('Done')
