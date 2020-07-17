## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
       
        linear_size_in = 17*17*256
        linear_size_out = 17*256 
        linear_size_out_2 = 136 
 
        self.conv = []
        self.conv.append(nn.Conv2d(1, 32, 4, padding=1))
        self.conv.append(nn.Conv2d(32, 64, 3, padding=1))
        self.conv.append(nn.Conv2d(64, 128, 2, padding=1))
        self.conv.append(nn.Conv2d(128, 256, 1, padding=1))
        
        self.act = []
        self.act.append(nn.ELU())
        self.act.append(nn.ELU())
        self.act.append(nn.ELU())
        self.act.append(nn.ELU())
        self.act.append(nn.ELU())
        self.act.append(nn.Linear(linear_size_out, linear_size_out))
        
        self.pool = []
        self.pool.append(nn.MaxPool2d(2, padding=1, stride=2))
        self.pool.append(nn.MaxPool2d(2, padding=1, stride=2))
        self.pool.append(nn.MaxPool2d(2, padding=1, stride=2))
        self.pool.append(nn.MaxPool2d(2, padding=1, stride=2))

        self.drop = []
        self.drop.append(nn.Dropout(0.1))
        self.drop.append(nn.Dropout(0.2))
        self.drop.append(nn.Dropout(0.3))
        self.drop.append(nn.Dropout(0.4))
        self.drop.append(nn.Dropout(0.5))
        self.drop.append(nn.Dropout(0.6))

        self.dense = []
        self.dense.append(nn.Linear(linear_size_in, linear_size_out))
        self.dense.append(nn.Linear(linear_size_out, linear_size_out))
        self.dense.append(nn.Linear(linear_size_out, linear_size_out_2))
        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.xavier_uniform_(self.dense[1].weight)
        nn.init.xavier_uniform_(self.dense[2].weight)
       
 
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        flat = nn.Flatten()
        x = self.conv[0](x)
        x = self.act[0](x)
        x = self.pool[0](x)
        x = self.drop[0](x)
        x = self.conv[1](x)
        x = self.act[1](x)
        x = self.pool[1](x)
        x = self.drop[1](x)
        x = self.conv[2](x)
        x = self.act[2](x)
        x = self.pool[2](x)
        x = self.drop[2](x)
        x = self.conv[3](x)
        x = self.act[3](x)
        x = self.pool[3](x)
        x = self.drop[3](x)
        x = flat(x)
        x = self.dense[0](x)
        x = self.act[4](x)
        x = self.drop[4](x)
        x = self.dense[1](x)
        x = self.act[5](x)
        x = self.drop[5](x)
        x = self.dense[2](x)
        return x
