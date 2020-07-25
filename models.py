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
        w2 = 5 
       
        linear_size_in = w2*w2*256
        linear_size_out = w2*256 
        linear_size_out_2 = 136 
 
        self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 1)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        
        self.act = nn.Linear(linear_size_out, linear_size_out)
        
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.6)

        self.dense1 = nn.Linear(linear_size_in, linear_size_out)
        self.dense2 = nn.Linear(linear_size_out, linear_size_out)
        self.dense3 = nn.Linear(linear_size_out, linear_size_out_2)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.dense3.weight)
       
 
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        flat = nn.Flatten()
        x = F.dropout(F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2), p=0.1)
        x = F.dropout(F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, stride=2), p=0.2)
        x = F.dropout(F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2), p=0.3)
        x = F.dropout(F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2, stride=2), p=0.4)

        x = flat(x) 
        
        x = F.dropout(F.relu(self.dense1(x)), p=0.5)
        x = F.dropout(self.act(self.dense2(x)), p=0.6)

        x = self.dense3(x)

        return x
