
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class TinyGoNet(nn.Module):
    def __init__(self, num_blocks=10, channels=128, num_classes=361):
        super().__init__()
        
        # Initial Conv
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        
        # Residual Tower
        self.blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 19 * 19, num_classes)
        
    def forward(self, x):
        # x: (B, 3, 19, 19)
        out = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.blocks:
            out = block(out)
            
        # Policy Head
        out = F.relu(self.policy_bn(self.policy_conv(out)))
        out = out.view(out.size(0), -1) # Flatten
        out = self.policy_fc(out)
        
        return out # Return logits
