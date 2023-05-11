import torch
from torch import nn
import torch.nn.functional as F

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], nc=3, zero_init_residual=False):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 224x224x3 > 112x112x64
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 # 112x112x64 > 56x56x64
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)      # 56x56x64 > 56x56x64
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)     # 56x56x64 > 28x28x128
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)     # 28x28x128 > 14x14x256
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)     # 14x14x256 > 7x7x512

        # Kaiming weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch (BarlowTwins does that)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockEnc):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))) # 112x112x64
        x = self.maxpool(x) # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128                    
        x = self.layer3(x)  # 14x14x256                    
        x = self.layer4(x)  # 7x7x512                   
        return x

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], nc=3, zero_init_residual=False):
        super().__init__()
        self.in_planes = 512
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2) # 7x7x512 > 14x14x256
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2) # 14x14x256 > 28x28x128
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)  # 28x28x128 > 56x56x64
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)  # 56x56x64 > 56x56x64
        self.conv2 = ResizeConv2d(64, 64, kernel_size=3, scale_factor=2)  # 56x56x64 > 112x112x64
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)  # 112x112x64 > 224x224x3
        self.act = nn.Tanh()

        # Kaiming weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch (BarlowTwins does that)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockDec):
                    nn.init.constant_(m.bn1.weight, 0)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):   # 7x7x512
        x = self.layer4(x)  # 14x14x256
        x = self.layer3(x)  # 28x28x128
        x = self.layer2(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = torch.relu(self.conv2(x))  # 112x112x64
        x = self.act(self.conv1(x))    # 224x224x3
        return x