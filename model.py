import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SAREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = DoubleConv(1, 32)
        self.pool0  = nn.MaxPool2d(2)
        self.stage1 = DoubleConv(32, 64)
        self.pool1  = nn.MaxPool2d(2)
        self.stage2 = DoubleConv(64, 128)
        self.pool2  = nn.MaxPool2d(2)
        self.stage3 = DoubleConv(128, 256)
        self.pool3  = nn.MaxPool2d(2)
        self.stage4 = DoubleConv(256, 512)
    def forward(self, x):
        s0 = self.stage0(x)
        s1 = self.stage1(self.pool0(s0))
        s2 = self.stage2(self.pool1(s1))
        s3 = self.stage3(self.pool2(s2))
        s4 = self.stage4(self.pool3(s3))
        return s0, s1, s2, s3, s4

class EOEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool   = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s0, s1, s2, s3, s4

class FusionBlock(nn.Module):
    def __init__(self, eo_ch, sar_ch, out_ch):
        super().__init__()
        self.sar_proj = nn.Conv2d(sar_ch, eo_ch, 1, bias=False) if sar_ch != eo_ch else nn.Identity()
        self.fuse = ConvBnRelu(eo_ch * 3, out_ch)
    def forward(self, eo_feat, sar_feat):
        sar_feat = self.sar_proj(sar_feat)
        if eo_feat.shape[2:] != sar_feat.shape[2:]:
            sar_feat = F.interpolate(sar_feat, size=eo_feat.shape[2:], mode="bilinear", align_corners=False)
        diff = torch.abs(eo_feat - sar_feat)
        return self.fuse(torch.cat([eo_feat, sar_feat, diff], dim=1))

class SiameseChangeNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.eo_enc  = EOEncoder(pretrained=pretrained)
        self.sar_enc = SAREncoder()
        self.fuse0 = FusionBlock(64,  32,  64)
        self.fuse1 = FusionBlock(64,  64,  64)
        self.fuse2 = FusionBlock(128, 128, 128)
        self.fuse3 = FusionBlock(256, 256, 256)
        self.fuse4 = FusionBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128,  64,  64)
        self.dec0 = DecoderBlock( 64,  64,  32)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
    def forward(self, eo, sar):
        eo0,  eo1,  eo2,  eo3,  eo4  = self.eo_enc(eo)
        sar0, sar1, sar2, sar3, sar4 = self.sar_enc(sar)
        f0 = self.fuse0(eo0, sar0)
        f1 = self.fuse1(eo1, sar1)
        f2 = self.fuse2(eo2, sar2)
        f3 = self.fuse3(eo3, sar3)
        f4 = self.fuse4(eo4, sar4)
        x = self.dec3(f4, f3)
        x = self.dec2(x,  f2)
        x = self.dec1(x,  f1)
        x = self.dec0(x,  f0)
        return self.head(x)
