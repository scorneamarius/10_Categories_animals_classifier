import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
	def __init__(self,in_channel, intermediate_channel, expansion, stride):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channel, intermediate_channel, kernel_size = 1, stride = stride,padding = 0, bias = False)
		self.bn1 = nn.BatchNorm2d(intermediate_channel)
		
		self.conv2 = nn.Conv2d(intermediate_channel, intermediate_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(intermediate_channel)
		
		self.conv3 = nn.Conv2d(intermediate_channel, intermediate_channel * expansion, kernel_size = 1, stride = 1, padding = 0, bias = False)
		self.bn3 = nn.BatchNorm2d(intermediate_channel * expansion)
		
		self.relu = nn.ReLU()
		self.identityConv = nn.Conv2d(in_channel, intermediate_channel * expansion, kernel_size = 1, stride = stride, bias = False)
		
		
	def forward(self, x):
		
		identity = self.identityConv(x.clone())
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		
		x = self.conv3(x)
		x = self.bn3(x)

		x = x + identity
		x = self.relu(x)
		
		return x

class ResNet(nn.Module):
    def __init__(self, factors, num_classes):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(64, factors[0], 64, False)
        self.layer2 = self._make_layer(256, factors[1], 128)
        self.layer3 = self._make_layer(512, factors[2], 256)
        self.layer4 = self._make_layer(1024, factors[3], 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fcl = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax()
        
        
    def forward(self, x):
    	
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, start_dim = 1)

        x = self.fcl(x)

        x = self.softmax(x)

        return x
         
    def _make_layer(self, in_channels, num_residual_blocks, intermediate_channels, downsample = True):
        layers = []
        residualBlock = ResidualBlock(in_channels, intermediate_channels, 4, 1)
        layers.append(residualBlock)
        if downsample == True:
            stride = 2
        else:
            stride = 1
        for i in range(1, num_residual_blocks):
            if i == (num_residual_blocks - 1):
                residualBlock = ResidualBlock(intermediate_channels * 4, intermediate_channels, 4, stride)
            else:
                residualBlock = ResidualBlock(intermediate_channels * 4, intermediate_channels, 4, 1)
            layers.append(residualBlock)            
        return nn.Sequential(*layers) 

def ResNet50(num_classes):
	return ResNet([3, 4, 6, 3], num_classes)

def ResNet101(num_classes):
	return ResNet([3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
	return ResNet([3, 8, 36, 3], num_classes)