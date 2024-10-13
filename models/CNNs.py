
import torch.nn as nn 
import torch.nn.functional as F 


class ConvNet(nn.Module):
  def __init__(self, img_size: int = 224, in_channels: int = 3, out_channels:int = 64, output_classes: int = 10, kernel_size: int = 5, padding: int = 0, stride: int = 1):
    super(ConvNet,self).__init__()

    self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = int(out_channels/2), kernel_size = kernel_size, padding = padding, stride = stride)
    # output of self.conv1
    # output_size x output_size x out_channels (with output_channels of self.conv1)
    # 28x28x32
    output_size = (img_size - kernel_size + 2*padding)/stride + 1

    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    # output_size after using Max Pooling
    # 14x14x32
    output_size = output_size/2 # stride of self.pool = 2

    self.conv2 = nn.Conv2d(in_channels = int(out_channels/2), out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride = stride)
    # output_size after using self.conv2 and max pooling
    output_size = ((output_size - kernel_size + 2*padding)/stride + 1)/2
    self.in_features = int(output_size*output_size*out_channels)

    self.fc1 = nn.Linear(self.in_features, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, output_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, self.in_features)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
        
        
