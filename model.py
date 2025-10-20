import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 编码器 (下采样路径)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # 尺寸减半
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # 瓶颈层 (U形的底部)
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器 (上采样路径)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 512(from up1) + 512(from skip) = 1024
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256) # 256 + 256 = 512
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128) # 128 + 128 = 256
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64) # 64 + 64 = 128

        # 最终输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        # 保存每一层池化前的输出，用于跳跃连接
        skip1 = self.down1(x)
        d1 = self.pool1(skip1)
        
        skip2 = self.down2(d1)
        d2 = self.pool2(skip2)
        
        skip3 = self.down3(d2)
        d3 = self.pool3(skip3)
        
        skip4 = self.down4(d3)
        d4 = self.pool4(skip4)
        
        # 瓶颈
        b = self.bottleneck(d4)
        
        # 解码 + 跳跃连接
        u1 = self.up1(b)
        # 如果上采样后尺寸和跳跃连接的特征图有1个像素的差异，需要进行裁剪或填充
        # 这里使用 F.interpolate 是更现代、更稳妥的做法
        if u1.shape != skip4.shape:
            u1 = F.interpolate(u1, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        cat1 = torch.cat([skip4, u1], dim=1) # 在通道维度上拼接
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        if u2.shape != skip3.shape:
            u2 = F.interpolate(u2, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        cat2 = torch.cat([skip3, u2], dim=1)
        uc2 = self.up_conv2(cat2)
        
        u3 = self.up3(uc2)
        if u3.shape != skip2.shape:
            u3 = F.interpolate(u3, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        cat3 = torch.cat([skip2, u3], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        if u4.shape != skip1.shape:
            u4 = F.interpolate(u4, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        cat4 = torch.cat([skip1, u4], dim=1)
        uc4 = self.up_conv4(cat4)
        
        return self.out_conv(uc4)

