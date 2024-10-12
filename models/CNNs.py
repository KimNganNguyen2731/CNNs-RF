
import torch.nn as nn 
import torch.nn.functional as F 


class ConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, stride: int = 1,
                 out_channels: int = 128, kernel_sz: int = 3, 
                 padding: int = 0, img_size: int = 224, 
                ):
        super(ConvNet, self).__init__()
        # The first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=int(out_channels/2),
                               kernel_size=kernel_sz,
                               stride=stride,
                               padding=padding)
        # The max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size 1
        out1 = (img_size - kernel_sz + 2*padding)/stride + 1
        out1 = out1//2
        # The second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=int(out_channels/2),
                               out_channels=out_channels,
                               kernel_size=kernel_sz,
                               padding=padding,
                               stride=stride)
        # Output size 2
        out2 = (out1 - kernel_sz + 2*padding)/stride + 1
        out2 = out2//2
        input_size = out2*out2*out_channels
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(input_size), 128)
        self.fc2 = nn.Linear(128, 32)
        
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x, 1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        return out
        
        