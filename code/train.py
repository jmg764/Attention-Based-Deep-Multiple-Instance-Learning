# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import print_function

import os
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
import PIL
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Network definition
from model_def import Attention

# Import SMDataParallel PyTorch Modules
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist

dist.init_process_group()


cuda = False

class TileDataset(data_utils.Dataset):

    def __init__(self, img_path, dataframe, num_tiles, transform=None):
        """
        img_zip: Where the images are stored
        dataframe: The train.csv dataframe
        num_tiles: How many tiles should the dataset return per sample
        transform: The function to apply to the image. Usually dataaugmentation. DO NOT DO NORMALIZATION here.
        """
        self.img_path = img_path
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = list(self.df['image_id'])
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_list[idx]

        tiles = ['/'+img_id + '_' + str(i) + '.png' for i in range(0, self.num_tiles)]
        image_tiles = []

        for tile in tiles:
            image = PIL.Image.open(self.img_path+tile)

            if self.transform is not None:
                image = self.transform(image)

            image = 1 - image
            image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304], [0.1279171 , 0.24528177, 0.16098117])(image)
            image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)

        return torch.tensor(image_tiles), torch.tensor(self.df.iloc[idx]['isup_grade'])

    def __len__(self):
        return len(self.img_list)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    predictions = []
    labels = []
    for batch_idx, (data, label) in enumerate(train_loader):
        print('epoch = ', epoch)
        print('batch_idx = ', batch_idx)
        bag_label = label
        data = torch.squeeze(data)
#         if cuda:
#             data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        data, bag_label = data.to(device), bag_label.to(device)

        # reset gradients
        optimizer.zero_grad()
        # calculate error
        bag_label = bag_label.float()
        Y_prob, Y_hat, _ = model(data)
        error = 1. - Y_hat.eq(bag_label).cpu().float().mean().data
        train_error += error
        # calculate loss
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = -1. * (bag_label * torch.log(Y_prob) + (1. - bag_label) * torch.log(1. - Y_prob))
        train_loss += loss.data[0]
        # Keep track of predictions and labels to calculate accuracy after each epoch
        predictions.append(int(Y_hat)) 
        labels.append(int(bag_label))
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Train Set, Epoch: {}, Loss: {:.4f}, Error: {:.4f}, Accuracy: {:.2f}%'.format(epoch, train_loss.cpu().numpy()[0], train_error, accuracy_score(labels, predictions)*100))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label
            data = torch.squeeze(data)

            data, bag_label = Variable(data), Variable(bag_label)
            data, bag_label = data.to(device), bag_label.to(device)

            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.pt'), 'wb') as f:
        torch.save(model.module.state_dict(), f)      

def get_csv(directory, df, num):
    # Getting tiles that are in S3
    tiles_list = []
    for image in os.listdir(directory):
        tiles_list.append(image.split('_')[0])

    # Creating dataframe containing labels for each tile in S3
    tiles_df = pd.DataFrame(columns=['image_id', 'data_provider', 'isup_grade', 'gleason_score'])
    for i in range(len(tiles_list)):
        tiles_df = tiles_df.append(df.loc[df['image_id'] == tiles_list[i]])
    
    # Drop duplicates first
    tiles_df = tiles_df.drop_duplicates()

    # Select the first 312 benign and 312 malignant slides 
    benign = tiles_df[tiles_df.isup_grade == 0][:int(num/2)]
    malignant = tiles_df[tiles_df.isup_grade == 1][:int(num/2)]

    tiles_df = pd.concat([benign, malignant])
    
    return tiles_df

def main():
    # Training settings
    
    parser = argparse.ArgumentParser(description='Histopathology MIL')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='For displaying SMDataParallel-specific logs')
    parser.add_argument('--data-path', type=str, default='/tmp/data', help='Path for downloading '
                                                                           'the MNIST dataset')
    # Model checkpoint location
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
   
    args = parser.parse_args()
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    args.lr = 1.0
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    data_path = args.data_path

                        
    if args.verbose:
        print('Hello from rank', rank, 'of local_rank',
                local_rank, 'in world size of', args.world_size)

    if not torch.cuda.is_available():
        raise Exception("Must run SMDataParallel on CUDA-capable devices.")

    torch.manual_seed(args.seed)

    device = torch.device("cuda")


    import boto3
    import sagemaker
    from sagemaker import get_execution_role

    bucket = 'sagemaker-us-east-1-318322629142'

    train_dir = '/opt/ml/input/data/training'
#     test_dir = '/opt/ml/input/data/testing'

    dataset_csv_key = 'panda_dataset.csv'
    dataset_csv_dir = 's3://{}/{}'.format(bucket, dataset_csv_key)
    
    df = pd.read_csv(dataset_csv_dir)
#     df['isup_grade'] = df['isup_grade'].replace([1,2], 0)
    df['isup_grade'] = df['isup_grade'].replace([1,2,3,4,5], 1)
    
    tiles_df = get_csv(train_dir, df, 624)
#     test_df = get_csv(test_dir, df)
    train_df, test_df = train_test_split(tiles_df)
    
    # Save dataframes to s3 bucket
    train_df.to_csv('s3://{}/{}'.format(bucket, 'train_df'))
    test_df.to_csv('s3://{}/{}'.format(bucket, 'test_df'))
    
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.ToTensor()])

    train_set = TileDataset(train_dir, train_df, 16, transform=transform_train)
   
    train_loader = data_utils.DataLoader(train_set, 1, shuffle=True, num_workers=0)
    

    device = torch.device("cuda")
    model = DDP(Attention().to(device))
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)
#     scheduler = StepLR(optimizer, step_size=1)

    print('Start Training')
    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
        # if rank == 0:
        #    test(model, device, test_loader)
#         scheduler.step()
    # print('Start Testing')
    # test()

    if rank == 0:
        print("Saving the model...")
#         save_model(model, args.model_dir)
        torch.save(model.state_dict(), "/opt/ml/model/model.pt")


if __name__ == '__main__':
    main()


