# Modified from:
#
# Zhang, K., Li, Y., Liang, J., Cao, J., Zhang, Y., Tang, H., Fan, D.P., Timofte, R. and Gool, L.V., 2023. Practical blind image denoising via Swin-Conv-UNet and data synthesis. Machine Intelligence Research, 20(6), pp.822-836.
# https://arxiv.org/abs/2203.13278
# Copyright 2022 Kai Zhang (cskaizhang@gmail.com, https://cszn.github.io/). All rights reserved.
# licenses/SCUNet_LICENSE
# https://github.com/cszn/SCUNet/blob/main/models/network_scunet.py
# 
# Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L. and Wei, F., 2022. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12009-12019).
# https://arxiv.org/abs/2111.09883
# Copyright (c) Microsoft Corporation
# licenses/SwinTransformer_LICENSE
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
import torch.utils
import torch.utils.checkpoint

def make_sequential(list):
    return nn.Sequential(*list)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = int(windows.shape[-1])
    x = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.embedding_layer = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.linear = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, device=x.device), self.v_bias))
        qkv = F.linear(input=x, weight=self.embedding_layer.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=4.605170185988092).exp() # max = log(1/0.01)
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.linear(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.msa = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size))

        self.ln1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            act_layer(),
            nn.Linear(4 * dim, dim),
        )
        self.noise = GaussianNoise()

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # attn_mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1), device=x.device, dtype=x.dtype).to(memory_format=torch.channels_last)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # W-MSA/SW-MSA
        attn_windows = self.msa(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.ln1(x))
        x = x + self.drop_path(self.ln2(self.mlp(x)))

        return self.noise(x)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.noise = GaussianNoise()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = self.noise(input + self.drop_path(x))
        return x

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False, octaves=3):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))
        self.octaves = max(1, octaves)

    def forward(self, x):
        if self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            #sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            #x = x + sampled_noise

            b, c, h, w = x.size()
            for i in range(self.octaves):
                size_scale = 1/(2**i)
                z = self.noise.repeat((b, c, int(h*size_scale), int(w*size_scale))).normal_()
                z = F.interpolate(z, size=(h, w), mode='bilinear')
                x = x + z * scale# * size_scale
        return x 

class ConvTransBlock(nn.Module):
    def __init__(self, dim, n_heads, window_size, drop_path=0.0, dropout=0.0, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.dim = dim
        self.conv_dim = dim // 2
        self.trans_dim = dim // 2
        self.n_heads = n_heads
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        shift_size = 0 if self.type == 'W' else self.window_size // 2

        self.trans_block = SwinTransformerBlock(self.trans_dim, self.n_heads, self.window_size, shift_size, 4, True, dropout, dropout, drop_path)
        self.conv1_1 = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=True)

        self.conv_block = ConvNeXtBlock(self.conv_dim, self.drop_path)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)

        conv_x = torch.utils.checkpoint.checkpoint(self.conv_block, conv_x, use_reentrant=False)
        trans_x = torch.utils.checkpoint.checkpoint(self.trans_block, trans_x.transpose(1, 3), use_reentrant=False).transpose(1, 3)
        
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))

        x = x + res
        return x

class Upconv(nn.Module):
    def __init__(self, dim, out_dim, scale=2, bias=False):
        super(Upconv, self).__init__()
        self.scale = scale

        self.up = []
        for _ in range(int(math.log2(self.scale))):
            self.up += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, dim, 3, 1, 1, bias=bias), nn.LeakyReLU(0.2, True)]
        self.up += [nn.Conv2d(dim, out_dim, 3, 1, 1, bias=bias), nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*self.up)
    
    def forward(self, x):
        #x = self.up(x)
        x = torch.utils.checkpoint.checkpoint(self.up, x, use_reentrant=False)
        return x
    
class SCUNetEncoder(nn.Module):
    def __init__(self, dim=64, n_heads=4, window_size=8, num_blocks=2, drop_path_rate=0.1, dropout=0.1, input_resolution=256):
        super(SCUNetEncoder, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * 3)]

        begin = 0
        self.m_down1 = make_sequential([ConvTransBlock(dim, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(num_blocks)] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)])

        begin += num_blocks
        self.m_down2 = make_sequential([ConvTransBlock(dim * 2, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(num_blocks)] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)])

        begin += num_blocks
        self.m_down3 = make_sequential([ConvTransBlock(dim * 4, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(num_blocks)] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)])
    
    def forward(self, x0):
        x1 = self.m_down1(x0)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        return x1, x2, x3

class SCUNetDecoder(nn.Module):
    def __init__(self, dim=64, n_heads=4, window_size=8, num_blocks=2, drop_path_rate=0.1, dropout=0.1, input_resolution=256):
        super(SCUNetDecoder, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks * 3)]

        begin = 0
        self.m_up3 = make_sequential([Upconv(8*dim, 4*dim, 2, bias=False),] + \
                      [ConvTransBlock(dim * 4, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(num_blocks)])
                      
        begin += num_blocks
        self.m_up2 = make_sequential([Upconv(4*dim, 2*dim, 2, bias=False),] + \
                      [ConvTransBlock(dim * 2, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(num_blocks)])
                      
        begin += num_blocks
        self.m_up1 = make_sequential([Upconv(2*dim, dim, 2, bias=False),] + \
                    [ConvTransBlock(dim, self.n_heads, self.window_size, dpr[i+begin], dropout, 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(num_blocks)])
    
    def forward(self, x, x1, x2, x3):
        x_dec3 = self.m_up3(x3 + x)
        x_dec2 = self.m_up2(x2 + x_dec3)
        x_dec1 = self.m_up1(x1 + x_dec2)
        return x_dec1, x_dec2, x_dec3

class SCUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=[3], dim=64, n_heads=4, num_enc_blocks=2, num_dec_blocks=2, num_fuse_blocks=2, drop_path_rate=0.1, dropout=0.1, input_resolution=256, scale=1):
        super(SCUNet, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = 8
        self.scale = scale
        self.in_nc = in_nc
        self.out_nc = out_nc

        self.m_head = nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)

        self.m_enc = SCUNetEncoder(dim, self.n_heads, self.window_size, num_enc_blocks, drop_path_rate, dropout, input_resolution)
        
        self.m_body = make_sequential([ConvTransBlock(dim * 8, self.n_heads, self.window_size, drop_path_rate, dropout, 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(num_enc_blocks)])
        
        for i in range(len(out_nc)):
            m_dec = SCUNetDecoder(dim, self.n_heads, self.window_size, num_dec_blocks, drop_path_rate, dropout, input_resolution)

            if scale > 1:
                m_tail = make_sequential([Upconv(dim, dim, scale), nn.Conv2d(dim, out_nc[i], 3, 1, 1, bias=False)])
            else:
                m_tail = make_sequential([nn.Conv2d(dim, out_nc[i], 3, 1, 1, bias=False)])

            if scale > 1:
                m_aux = make_sequential([Upconv(dim, dim, scale), nn.Conv2d(dim, out_nc[i], 3, 1, 1, bias=False)])
            else:
                m_aux = make_sequential([nn.Conv2d(dim, out_nc[i], 3, 1, 1, bias=False)])

            setattr(self, f'm_dec_{i}', m_dec)
            setattr(self, f'm_aux_{i}', m_aux)
            setattr(self, f'm_tail_{i}', m_tail)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_fuse_blocks)]

        self.m_fuse = make_sequential(
            [nn.Conv2d(dim * len(out_nc), dim, 3, 1, 1)] + \
            [ConvTransBlock(dim, self.n_heads, self.window_size, dpr[i], dropout, 'W' if not i%2 else 'SW', input_resolution)
             for i in range(num_fuse_blocks)] + \
            [nn.Conv2d(dim, dim * len(out_nc), 3, 1, 1)]
        )

    def forward(self, x0, use_aux=False):
        input = x0
        h, w = x0.size()[-2:]

        paddingH = int(np.ceil(h/64)*64-h) + 64
        paddingW = int(np.ceil(w/64)*64-w) + 64

        paddingLeft = math.ceil(paddingW / 2)
        paddingRight = math.floor(paddingW / 2)
        paddingTop = math.ceil(paddingH / 2)
        paddingBottom = math.floor(paddingH / 2)

        x0 = F.pad(x0, (paddingLeft, paddingRight, paddingTop, paddingBottom), mode='reflect')

        x0 = self.m_head(x0)
        x1, x2, x3 = self.m_enc(x0)
        
        x = self.m_body(x3)

        dec = [None] * len(self.out_nc)
        aux = [None] * len(self.out_nc)
        for i in range(len(self.out_nc)):
            m_dec = getattr(self, f'm_dec_{i}')
            x_dec1, _, _ = m_dec(x, x1, x2, x3)
            dec[i] = x_dec1

            if self.training and use_aux:
                m_aux = getattr(self, f'm_aux_{i}')
                aux[i] = m_aux(x_dec1)
            
        dec = torch.cat(dec, dim=1)
        dec = self.m_fuse(dec) + dec
        dec = torch.split(dec, self.dim, dim=1)

        outs = [None] * len(self.out_nc)
        for i in range(len(self.out_nc)):
            m_tail = getattr(self, f'm_tail_{i}')
            outs[i] = m_tail(dec[i] + x0)
            
        out = torch.cat(outs, dim=1)[..., paddingTop*self.scale:paddingTop*self.scale+h*self.scale, paddingLeft*self.scale:paddingLeft*self.scale+w*self.scale]

        if self.training:
            if use_aux:
                aux = torch.cat(aux, dim=1)[..., paddingTop*self.scale:paddingTop*self.scale+h*self.scale, paddingLeft*self.scale:paddingLeft*self.scale+w*self.scale]
            return out, aux

        return out

import time

def test_eval():
    print( "SCUNet(3, [3, 3, 1, 1], 96, 1, 2, 2, 4, 0.1, 0.0, 64, 1)")
    model = SCUNet(3, [3, 3, 1, 1], 96, 1, 2, 2, 4, 0.1, 0.0, 64, 1).cuda()
    model.eval()
    x_cf = torch.randn(1, 3, 64, 64).cuda()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y_cf = model(x_cf)
    print(f"channels_first {x_cf.shape} -> {y_cf.shape}")
    
    if True:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        ) as profiler:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                y_cf = model(x_cf)

        for s in str(profiler.key_averages()).split('\n'):
            print(s)

    batch_size = 8
    size = 64
    iters = 8

    total_time = 0
    for _ in range(iters):
        x = torch.randn(batch_size, 3, size, size).cuda()
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            y = model(x)
        total_time += time.time() - start
    print(f"channels_first 8b 64x64: Average time: {total_time / iters * 1000:.3f}ms")

    model = model.to(memory_format=torch.channels_last)
    model.eval()
    x_cl = x_cf.to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y_cl = model(x_cl)
    print(f"channels_last {x_cl.shape} -> {y_cl.shape}")

    total_time = 0
    for _ in range(iters):
        x = torch.randn(batch_size, 3, size, size).cuda().to(memory_format=torch.channels_last)
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            y = model(x)
        total_time += time.time() - start
    print(f"channels_last 8b 64x64: Average time: {total_time / iters * 1000:.3f}ms")

    model = model.jit()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y_jit = model(x_cl)
    print(f"channels_last {x_cl.shape} -> {y_jit.shape}")

    print(f"channels_first: sum: {y_cf.sum()}, mean: {y_cf.mean()}, std: {y_cf.std()}")
    print(f"channels_last: sum: {y_cl.sum()}, mean: {y_cl.mean()}, std: {y_cl.std()}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")


def test_train():
    print( "SCUNet(3, [3, 3, 1, 1], 96, 1, 2, 2, 4, 0.1, 0.0, 64, 1)")
    model = SCUNet(3, [3, 3, 1, 1], 96, 1, 2, 2, 4, 0.1, 0.0, 64, 1).cuda().to(memory_format=torch.channels_last)
    model.train()
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.randn(1, 3, 64, 64).cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y, _ = model(x, use_aux=True, pad=False)
    print(f"{x.shape} -> {y.shape}")
    
    batch_size = 1
    size = 64
    iters = 8

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_time = 0
    for _ in range(iters):
        x = torch.randn(batch_size, 3, size, size).cuda().to(memory_format=torch.channels_last)
        start = time.time()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            y, _ = model(x, use_aux=True, pad=False)
            loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_time += time.time() - start
    print(f"training {batch_size}b 64x64: Average time: {total_time / iters * 1000:.3f}ms")
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)/1024/1024:.2f}MB memory")

if __name__ == "__main__":
    #test_eval()
    test_train()
