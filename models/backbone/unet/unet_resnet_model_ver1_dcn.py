# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
from .. import resnet
from mmdet.ops.dcn.deform_conv import ModulatedDeformConvPack as Deform
from mmdet.ops.context_block import ContextBlock


class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
        self,
        dim,
        dim_inner=None,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner if dim_inner is not None else dim // 2
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv2d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output.
        self.conv_out = nn.Conv2d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        # TODO: change the name to `norm`
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
        # Zero initializing the final bn.
        self.bn.transform_final_bn = zero_init_final_norm

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = nn.MaxPool2d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        x_identity = x
        N, C, H, W = x.size()

        g = self.conv_g(x)

        # Perform temporal-spatial pooling to reduce the computation.
        # if self.use_pool:
            # x = self.pool(x)
        
        x = F.adaptive_avg_pool2d(x, H // 4)

        phi = self.conv_phi(x)
        theta = self.conv_theta(x)
        
        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, TxHxW) * (N, C, TxHxW) => (N, C, C).
        theta_phi = torch.einsum("nct,ndp->ncd", (theta, phi))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, C, C) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ncg,ngt->nct", (theta_phi, g))

        # (N, C, TxHxW) => (N, C, T, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p


class UNetResNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
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

        # self.deform = Deform(int(16 * w), int(16 * w), 3, padding=1)
        # self.context_block = ContextBlock(int(16 * w), 1)
        # self.non_local = Nonlocal(int(16 * w))

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

        # x = self.deform(x)
        # x = self.context_block(x)
        # x = self.non_local(x)

        x = self.outc(x)
        return x

def unet05res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.5, **kwargs)

def unet025res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.25, **kwargs)

def unet1res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=1, **kwargs)

def unet2res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=2, **kwargs)

def unet4res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=4, **kwargs)
