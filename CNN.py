import torch.nn as nn
import torch.nn.functional as F

"""
_summary_ Create CNN for CIFAR-10 inheriting attributes from pytorch nn.Module
"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input: 3 x 32 x 32, as defined by CIFAR-10 dataset
        # uses convolution to extract features from the input image, and creates 16 feature maps using 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        # max pooling layer to reduce the spatial dimensions of the feature maps
        # reduces the size of the feature maps by a factor of 2, effectively halving the spatial dimensions
        # reduces the number of parameters in the network and helps prevent overfitting
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # second convolutional layer to extract features from the feature maps
        # we use a second convolutional layer to extract features from the feature maps because the first layer may not capture all the features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # max pooling again to reduce the spatial dimensions of the feature maps
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # flatten 16x16x16 feature maps into a 1D vector
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        # fully connected layer to output 128 neurons
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

        # dropout layer to prevent overfitting & reliance on specific features
        self.dropout = nn.Dropout(p=0.15)  

    def forward(self, x):
        # apply ReLU activation function to the output of the first convolutional layer
        x = self.pool(F.relu(self.conv1(x)))
        # same to second CL
        x = self.pool(F.relu(self.conv2(x)))
        # flatten the feature maps into a 1D vector
        x = x.view(-1, 32 * 8 * 8)
        # apply ReLU activation function to the output of the fully connected layer
        x = F.relu(self.fc1(x))
        # apply dropout to the output of the fully connected layer, preventing overfitting
        x = self.dropout(x)
        # apply softmax activation function to the output of the fully connected layer
        x = self.fc2(x)
        return x
