# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


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

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W) # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        x = img2windows(x, self.H_sp, self.W_sp) # [N*56*1 56 32] [N*14*1 56 64] [N*2*1 98 128] [N*1*1 49 512]
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() # [N*56*1 1 56 32] [N*14*1 2 56 32] [N*2*1 4 98 32] [N*1*1 16 49 32]
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape # [N 3136 32] [N 784 64] [N 196 128] [N 49 512]
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W) # [N 32 56 56] [N 64 28 28] [N 128 14 14] [N 512 7 7]

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp) # [N 32 1 56 56 1] [N 32 56 1 1 56] / [N 64 1 28 14 2] [N 64 14 2 1 28] / [N 128 1 14 2 7] [N 128 2 7 1 14] / [N 512 1 7 1 7]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W' # [N*56*1 32 56 1][N*56*1 32 1 56] / [N*14*1 64 28 2][N*14*1 64 2 28] / [N*2*1 128 14 7][N*2*1 128 7 14] / [N*1*1 512 7 7]

        lepe = func(x) ### B', C, H', W' # [N*56*1 32 56 1] [N*56*1 32 1 56] / [N*14*1 64 28 2][N*14*1 64 2 28] / [N*2*1 128 14 7][N*2*1 128 7 14]  / [N*1*1 512 7 7]
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous() # [N*56*1 1 56 32] [N*14*1 2 56 32] [N*2*1 4 98 32] [N*1*1 16 49 32]

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous() # [N*56*1 1 56 32] [N*14*1 2 56 32] [N*2*1 4 98 32] [N*1*1 16 49 32]
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2] # [N 3136 32] [N 784 64] [N 196 128] [N 49 512]

        ### Img2Window
        H = W = self.resolution # 56 28 14 7
        B, L, C = q.shape # [N 3136 32] [N 784 64] [N 196 128] [N 49 512]
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q) # [N*56*1 1 56 32] [N*14*1 2 56 32] [N*2*1 4 98 32] [N*1*1 16 49 32]
        k = self.im2cswin(k) # [N*56*1 1 56 32] [N*14*1 2 56 32] [N*2*1 4 98 32] [N*1*1 16 49 32]
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C # [N*56*1 56 32] [N*14*1 56 64] [N*2*1 98 128] [N*1*1 49 512]

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x # [N 3136 32] [N 784 64] [N 196 128] [N 49 512]


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution # 56
        B, L, C = x.shape # [N 3136 64] [N 784 128] [N 196 256] [N 49 512]
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # [3 N 3136 64] [3 N 784 128] [3 N 196 256] [3 N 49 512]
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2]) # qkv[3 N 3136 32]->x1[N 3136 32] qkv[3 N 784 128]->x1[N 784 64] qkv[3 N 196 256]->x1[N 196 128]
            x2 = self.attns[1](qkv[:,:,:,C//2:]) # qkv[3 N 3136 32]->x2[N 3136 32] qkv[3 N 784 128]->x1[N 784 64] qkv[3 N 196 256]->x1[N 196 128]
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x # [N 3136 64] [N 784 128] [N 196 256] [N 49 512]

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp) # [N 32 1 56 56 1] [N 32 56 1 1 56] / [N 64 1 28 14 2] [N 64 14 2 1 28] / [N 128 1 14 2 7] [N 128 2 7 1 14] / [N 512 1 7 1 7]
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C) # [N*56*1 56 32] [N*56*1 56 32] / [N*14*1 56 64] [N*14*1 56 64] / [N*2*1 98 128] [N*2*1 98 128] / [N*1*1 49 512]
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1) # [N*56*1 56 32]->[N 1 56 56 1 32] [N*56*1 56 32]->[N 56 1 1 56 32] / [N*14*1 56 64]->[N 1 14 28 2 64] [N*14*1 56 64]->[N 14 1 2 28 64] / [N*2*1 98 128]->[N 1 2 14 7 128] [N*2*1 98 128]->[N 2 1 7 14 128] / [N*1*1 49 512]->[N 1 1 7 7 512]
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1) # [N 56 56 32] [N 28 28 64] [N 14 14 128] [N 7 7 512]
    return img

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models  64
        heads=num_heads # [2 4 8 16]

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h = img_size//4, w = img_size//4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])
       
        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed(x) # [N 3 224 224] -> [N 3136 64]
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3], 
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x) # [N 784 128] [N 196 256] [N 49 512]
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        x = self.norm(x)
        return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

### 224 models

@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1,2,21,1],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

### 384 models

@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

