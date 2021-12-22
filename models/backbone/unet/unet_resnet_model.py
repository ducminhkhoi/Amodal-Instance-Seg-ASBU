# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
from .. import resnet
from mmdet.ops.dcn.deform_conv import ModulatedDeformConvPack as Deform


class UNetResNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNet, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.image_encoder = resnet.resnet18(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(self.image_encoder.out_dim, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.up1 = up(int(384 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)


    def forward(self, x, rgb, return_feat=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        img_feat = self.image_encoder(rgb)
        img_feat = self.reduce_dim(img_feat)
        img_feat = F.interpolate(
            img_feat, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=True)
        cat = torch.cat((x5, img_feat), dim=1) # 256 + 128 * w
        x = self.up1(cat, x4)
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

    
class UNetResNetPredictOrder(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNetPredictOrder, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.image_encoder = resnet.resnet18(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(self.image_encoder.out_dim, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))

        self.up1 = up(int(384 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w)+n_classes, n_classes)

        self.up1_b = up(int(384 * w), int(64 * w))
        self.up2_b = up(int(128 * w), int(32 * w))
        self.up3_b = up(int(64 * w), int(16 * w))
        self.up4_b = up(int(32 * w), int(16 * w))
        self.outc_b = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)


    def forward(self, x, rgb):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        img_feat = self.image_encoder(rgb)
        img_feat = self.reduce_dim(img_feat)
        img_feat = F.interpolate(
            img_feat, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=True)
        cat = torch.cat((x5, img_feat), dim=1) # 256 + 128 * w

        x = self.up1(cat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_deform:
            x = self.deform(x)

        x_ = self.up1_b(cat, x4)
        x_ = self.up2_b(x_, x3)
        x_ = self.up3_b(x_, x2)
        x_ = self.up4_b(x_, x1)
        order_predict = self.outc_b(x_)
            
        x = self.outc(torch.cat([x, F.softmax(order_predict.detach(), 1)], 1))

        return x, order_predict


class UNetResNetPredictOrder2(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNetPredictOrder2, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.image_encoder = resnet.resnet18(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(self.image_encoder.out_dim, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))

        self.up1 = up(int(384 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.up1_b = up(int(384 * w), int(64 * w))
        self.up2_b = up(int(128 * w), int(32 * w))
        self.up3_b = up(int(64 * w), int(16 * w))
        self.up4_b = up(int(32 * w), int(16 * w))
        self.outc_b = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform
        if self.use_deform:
            self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)


    def forward(self, x, rgb):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        img_feat = self.image_encoder(rgb)
        img_feat = self.reduce_dim(img_feat)
        img_feat = F.interpolate(
            img_feat, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=True)
        cat = torch.cat((x5, img_feat), dim=1) # 256 + 128 * w

        x = self.up1(cat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.use_deform:
            x = self.deform(x)

        x = self.outc(x)

        x_ = self.up1_b(cat, x4)
        x_ = self.up2_b(x_, x3)
        x_ = self.up3_b(x_, x2)
        x_ = self.up4_b(x_, x1)
        order_predict = self.outc_b(x_)

        return x, order_predict



class UNetResNetNMF(nn.Module):
    """ predict coeff from mask and predict parts from image """
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNetNMF, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.image_encoder = resnet.resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True])
        
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        self.reduce_dim =nn.Conv2d(self.image_encoder.out_dim, n_classes, kernel_size=1)

        self.coeff = nn.Sequential(
            down(int(128 * w), int(64 * w)),
            down(int(64 * w), int(32 * w)),
            down(int(32 * w), int(16 * w)),
            down(int(16 * w), n_classes),
        )

    def forward(self, x, rgb):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        coeff = self.coeff(x5)

        # with torch.no_grad():
        img_feat = self.image_encoder(rgb)

        part = self.reduce_dim(img_feat)
        part = F.interpolate(part, rgb.shape[-2:], mode='bilinear', align_corners=True)

        return part, coeff


class UNetResNetNMF2(nn.Module):
    """from image predict both coeff and parts """
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNetNMF2, self).__init__()
        self.image_encoder = resnet.resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True])

        self.reduce_dim =nn.Conv2d(self.image_encoder.out_dim, n_classes, kernel_size=1)

        self.coeff = nn.Sequential(
            down(int(256 * w), int(128 * w)),
            down(int(128 * w), int(64 * w)),
            down(int(64 * w), int(32 * w)),
            down(int(32 * w), int(16 * w)),
            down(int(16 * w), n_classes),
        )

    def forward(self, x, rgb):
        img_feat = self.image_encoder(rgb)

        coeff = self.coeff(img_feat)

        part = self.reduce_dim(img_feat)
        part = F.interpolate(part, x.shape[-2:], mode='bilinear', align_corners=True)

        return part, coeff


def unet05res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.5, **kwargs)

def unet025res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.25, **kwargs)

def unet1res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=1, **kwargs)

def unet2res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=2, **kwargs)

def unet2resPredictOrder(in_channels, **kwargs):
    return UNetResNetPredictOrder(in_channels, w=2, **kwargs)

def unet2resPredictOrder2(in_channels, **kwargs):
    return UNetResNetPredictOrder2(in_channels, w=2, **kwargs)

def unet2resNMF(in_channels, **kwargs):
    return UNetResNetNMF(in_channels, w=2, **kwargs) # change here to UNetResNetNMF

def unet4res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=4, **kwargs)
