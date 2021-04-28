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
        # Here I am using an already existing kernel output with a zipfile. 
        # I suggest extracting files as it can lead to issue with multiprocessing.
        # self.zip_img = zipfile.ZipFile(img_zip_path) 
        self.img_path = img_path
        self.df = dataframe
        self.num_tiles = num_tiles
        self.img_list = self.df['image_id'].values

        
        self.transform = transform

    def __getitem__(self, idx):
        img_id = list(self.img_list[idx].values())[0]

        tiles = ['/'+img_id + '_' + str(i) + '.png' for i in range(0, self.num_tiles)]
        metadata = self.df.iloc[idx]
        image_tiles = []

        for tile in tiles:
            image = PIL.Image.open(self.img_path+tile)

            if self.transform is not None:
                image = self.transform(image)

            image = 1 - image
            image = transforms.Normalize([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304], [0.1279171 , 0.24528177, 0.16098117])(image)
            image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)

        return torch.tensor(image_tiles), torch.tensor(list(metadata['isup_grade'].values())[0])

    def __len__(self):
        return len(self.img_list)

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label
        if cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label
        # instance_labels = label[1]
        if cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            # instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                #  np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n')
                  # 'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.module.state_dict(), f)

def main():
    # Training settings
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
    '''

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    # args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    # args.lr = 1.0
    # args.batch_size //= args.world_size // 8
    # args.batch_size = max(args.batch_size, 1)
    # data_path = args.data_path

                        
    # if args.verbose:
    #     print('Hello from rank', rank, 'of local_rank',
    #             local_rank, 'in world size of', args.world_size)

    if not torch.cuda.is_available():
        raise Exception("Must run SMDataParallel on CUDA-capable devices.")

    torch.manual_seed(args.seed)

    device = torch.device("cuda")


    import boto3
    import pandas as pd
    from sagemaker import get_execution_role

    role = get_execution_role()
    bucket='sagemaker-us-east-2-318322629142'

    train_tiles_key = 'train_tiles'
    train_tiles_dir = 's3://{}/{}'.format(bucket, train_tiles_key)
    
    test_tiles_key = 'test_tiles'
    test_tiles_dir = 's3://{}/{}'.format(bucket, test_tiles_key)

    dataset_csv_key = ‘panda_dataset.csv’
    dataset_csv_dir = 's3://{}/{}'.format(bucket, dataset_csv_key)

    model_key = ‘model’
    model_dir = 's3://{}/{}'.format(bucket, model_key)

    df = pd.read_csv(dataset_csv_dir)

    train_dict = {}
    for image in os.listdir(train_tiles_dir):
        train_dict[image.split('_')[0]] = train_dict.get(image.split('_')[0], 0) + 1

    test_dict = {}
    for image in os.listdir(test_tiles_dir):
        test_dict[image.split('_')[0]] = test_dict.get(image.split('_')[0], 0) + 1

    train_df = list(train_dict.keys())
    test_df = list(test_dict.keys())

    new_train_df = []
    for i in range(len(train_df)):
        row = df.loc[df['image_id'] == train_df[i]]
        new_train_df.append(row.to_dict())

    new_test_df = []
    for i in range(len(test_df)):
        row = df.loc[df['image_id'] == test_df[i]]
        new_test_df.append(row.to_dict())

    train_df = pd.DataFrame(new_train_df)
    test_df = pd.DataFrame(new_test_df)

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.ToTensor()])

    train_loader = TileDataset(train_dir, train_df, 12, transform=transform_train)
    test_loader = TileDataset(test_dir, test_df, 12, transform=transform_train)

    
    '''
    if local_rank == 0:
        train_dataset = datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    else:
        # TODO: Reduce time to half when upgrade to torchvision==0.9.1
        time.sleep(16)
        train_dataset = datasets.MNIST(data_path, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)
    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True)
    '''

    # Use SMDataParallel PyTorch DDP for efficient distributed training

    
    model = DDP(Attention().to(device))
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=1)

    print('Start Training')
    for epoch in range(1, 100 + 1):
        train(epoch)
        # if rank == 0:
        #    test(model, device, test_loader)
        scheduler.step()
    # print('Start Testing')
    # test()

    if rank == 0:
        print("Saving the model...")
        save_model(model, model_dir)   


if __name__ == '__main__':
    main()
