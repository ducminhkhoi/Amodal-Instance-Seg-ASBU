# full assembly of the sub-parts to form the complete net
import torch.nn as nn
import torch.nn.functional as F
import torch

from .unet_parts import inconv, down, up, outconv

from mmdet.ops.dcn.deform_conv import ModulatedDeformConvPack as Deform

class UNetD2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetD2, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 32 * w)
        self.up1 = up(64 * w, 16 * w)
        self.up2 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x) # 16
        x2 = self.down1(x1) # 32
        x3 = self.down2(x2) # 32
        x = self.up1(x3, x2) # 16
        x = self.up2(x, x1) # 16

        if self.use_deform:
            x = self.deform(x)

        x = self.outc(x)
        return x


class UNetD3(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetD3, self).__init__()
        self.inc = inconv(in_channels, 16 * w)
        self.down1 = down(16 * w, 32 * w)
        self.down2 = down(32 * w, 64 * w)
        self.down3 = down(64 * w, 64 * w)
        self.up2 = up(128 * w, 32 * w)
        self.up3 = up(64 * w, 16 * w)
        self.up4 = up(32 * w, 16 * w)
        self.outc = outconv(16 * w, n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_deform:
            x = self.deform(x)

        x = self.outc(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)

    def forward(self, x, return_feat=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_deform:
            x = self.deform(x)
            
        x_ = x
        x = self.outc(x)

        if return_feat:
            return x, x_
        else:
            return x


class UNetPredictOrder(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetPredictOrder, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w)+n_classes, n_classes)

        self.up1_b = up(int(256 * w), int(64 * w))
        self.up2_b = up(int(128 * w), int(32 * w))
        self.up3_b = up(int(64 * w), int(16 * w))
        self.up4_b = up(int(32 * w), int(16 * w))
        self.outc_b = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_deform:
            x = self.deform(x)

        x_ = self.up1_b(x5, x4)
        x_ = self.up2_b(x_, x3)
        x_ = self.up3_b(x_, x2)
        x_ = self.up4_b(x_, x1)
        order_predict = self.outc_b(x_)

        x = self.outc(torch.cat([x, F.softmax(order_predict.detach(), 1)], 1))

        return x, order_predict


class UNetPredictOrder2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetPredictOrder2, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.up1_b = up(int(256 * w), int(64 * w))
        self.up2_b = up(int(128 * w), int(32 * w))
        self.up3_b = up(int(64 * w), int(16 * w))
        self.up4_b = up(int(32 * w), int(16 * w))
        self.outc_b = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        if self.use_deform:
            x = self.deform(x)

        x_ = self.up1_b(x5, x4)
        x_ = self.up2_b(x_, x3)
        x_ = self.up3_b(x_, x2)
        x_ = self.up4_b(x_, x1)
        order_predict = self.outc_b(x_)

        return x, order_predict


class UNetNMF(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetNMF, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))

        self.coeff = nn.Sequential(
            down(int(128 * w), int(64 * w)),
            down(int(64 * w), int(32 * w)),
            down(int(32 * w), int(16 * w)),
            down(int(16 * w), n_classes),
        )

        self.parts = nn.Parameter(torch.randn(1, n_classes, 32, 32))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        coeff = self.coeff(x5)

        parts = F.interpolate(self.parts, x.shape[-2:], mode='bilinear', align_corners=True)

        return parts, coeff


def unet05(in_channels, **kwargs):
    return UNet(in_channels, w=0.5, **kwargs)

def unet025(in_channels, **kwargs):
    return UNet(in_channels, w=0.25, **kwargs)

def unet1(in_channels, **kwargs):
    return UNet(in_channels, w=1, **kwargs)

def unet2(in_channels, **kwargs):
    return UNet(in_channels, w=2, **kwargs)

def unet2NMF(in_channels, **kwargs):
    return UNetNMF(in_channels, w=2, **kwargs)

def unet2PredictOrder(in_channels, **kwargs):
    return UNetPredictOrder(in_channels, w=2, **kwargs)

def unet2PredictOrder2(in_channels, **kwargs):
    return UNetPredictOrder2(in_channels, w=2, **kwargs)

def unet4(in_channels, **kwargs):
    return UNet(in_channels, w=4, **kwargs)

def unet1d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=1, **kwargs)

def unet2d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=2, **kwargs)

def unet4d2(in_channels, **kwargs):
    return UNetD2(in_channels, w=4, **kwargs)

def unet1d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=1, **kwargs)

def unet2d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=2, **kwargs)

def unet4d3(in_channels, **kwargs):
    return UNetD3(in_channels, w=4, **kwargs)
