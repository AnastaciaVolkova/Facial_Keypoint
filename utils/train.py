import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from func_unitls import net_sample_output
from func_unitls import visualize_output
from func_unitls import train_net
import torch.optim as optim


def main():
    # Define network
    net = Net()
    print(net)

    data_transform = transforms.Compose([
        Rescale(250),
        Normalize(),
        RandomCrop(224)]
    )

    train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    # iterate through the transformed dataset and print some stats about the first few samples
    for i in range(4):
        sample = train_dataset[i]
        print(i, sample['image'].size, sample['keypoints'].size)

    train_loader = DataLoader(train_dataset,
                              batch_size=10,
                              shuffle=True,
                              num_workers=4)

    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                          root_dir='data/test/',
                                          transform=data_transform)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)

    test_images, test_outputs, gt_pts = net_sample_output(net, test_loader)

    # print out the dimensions of the data to see if they make sense
    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    # call it
    visualize_output(test_images, test_outputs, gt_pts, 1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    n_epochs = 2

    train_net(net, criterion, optimizer, train_loader,  n_epochs)

    # get a sample of test data again
    test_images, test_outputs, gt_pts = net_sample_output(net, test_loader)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())

    model_dir = 'saved_models/'
    model_name = 'keypoints_model_1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)

    weights1 = net.conv1.weight.data

    w = weights1.numpy()

    filter_index = 0

    print(w[filter_index][0])
    print(w[filter_index][0].shape)

    # display the filter weights
    plt.imshow(w[filter_index][0], cmap='gray')


def debug_me():
    # Define network
    net = Net()
    print(net)

    data_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        Normalize()
    ]
    )

    aww_dataset = FacialKeypointsDataset(csv_file='data/aww_frames_keypoints.csv',
                                           root_dir='data/aww/',
                                           transform=data_transform)

    sample = aww_dataset[0]
    print(sample['image'].shape, sample['keypoints'].shape)
    print(np.max(sample['keypoints']))

    aww_loader = DataLoader(aww_dataset,
                              batch_size=10,
                              shuffle=True,
                              num_workers=4)

    aww_images, aww_outputs, gt_pts = net_sample_output(net, aww_loader)

    visualize_output(aww_images, aww_outputs, gt_pts, 1)

    '''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    n_epochs = 2

    train_net(net, criterion, optimizer, aww_loader, n_epochs)

    # get a sample of test data again
    aww_images, aww_outputs, gt_pts = net_sample_output(net, aww_loader)

    print(aww_images.data.size())
    print(aww_outputs.data.size())
    print(gt_pts.size())

    model_dir = 'saved_models/'
    model_name = 'keypoints_model_1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir + model_name)

    weights1 = net.conv1.weight.data

    w = weights1.numpy()

    filter_index = 0

    print(w[filter_index][0])
    print(w[filter_index][0].shape)

    # display the filter weights
    plt.imshow(w[filter_index][0], cmap='gray')
    '''


if __name__ == '__main__':
    debug_me()
