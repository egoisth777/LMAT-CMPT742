import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line

class twoConvBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(twoConvBlock, self).__init__()
    #todo
    #initialize the block
    # first 3x3 convolution with ReLU
    self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=0)
    self.relu1 = nn.ReLU()
    
    # second 3x3 convolution with batch normalization and ReLU
    self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=0)
    self.bn2 = nn.BatchNorm2d(output_channel)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    #todo
    #implement the forward path
    # first convolution and ReLU
    x = self.relu1(self.conv1(x))
    
    # second convolution, batch norm, and ReLU
    x = self.relu2(self.bn2(self.conv2(x)))
    
    return x

class downStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(downStep, self).__init__()
    #todo
    #initialize the down path
    self.conv_block = twoConvBlock(input_channel, output_channel)
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self):
    #todo
    #implement the forward path
    x = self.conv_block(x)
    x = self.max_pool(x)
        
    return x

class upStep(nn.Module):
  def __init__(self, input_channel, output_channel):
    super(upStep, self).__init__()
    #todo
    #initialize the up path
    self.up_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
    
    # Convolution block to process concatenated output
    self.conv_block = twoConvBlock(input_channel, output_channel)

  def forward(self):
    #todo
    #implement the forward path
    # x1 is the input from the previous layer
    # x2 is the feature map from the contracting path (skip connection)
    
    # Upsample the input feature map
    x1 = self.up_conv(x1)
    
    # Concatenate the upsampled feature map with the corresponding contracting path feature map
    # Make sure the feature map sizes match for concatenation (use cropping if necessary)
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    
    # Crop x2 to match the size of x1
    x2 = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2, diffX // 2 : x2.size()[3] - diffX // 2]
    
    # Concatenate along the channel dimension
    x = torch.cat([x2, x1], dim=1)
    
    # Apply the convolutional block
    x = self.conv_block(x)
    
    return x

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    # Contracting Path
    self.down1 = downStep(1, 64)
    self.down2 = downStep(64, 128)
    self.down3 = downStep(128, 256)
    self.down4 = downStep(256, 512)

    # Bottom layer of the U-Net without pooling
    self.bottom = twoConvBlock(512, 1024)

    # Expansive Path
    self.up4 = upStep(1024, 512)
    self.up3 = upStep(512, 256)
    self.up2 = upStep(256, 128)
    self.up1 = upStep(128, 64)

    # Final output layer to get the segmentation map
    self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # Contracting Path
    x, x1 = self.down1(x)
    x, x2 = self.down2(x)
    x, x3 = self.down3(x)
    x, x4 = self.down4(x)

    # Bottom layer
    x = self.bottom(x)

    # Expansive Path with skip connections
    x = self.up4(x, x4)
    x = self.up3(x, x3)
    x = self.up2(x, x2)
    x = self.up1(x, x1)

    # Final output
    x = self.sigmoid(self.final_conv(x))
    return x
