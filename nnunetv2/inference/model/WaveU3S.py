import numpy as np
import sys
import torch
from torch import nn
from typing import Tuple, Union, Sequence
from .neural_network import SegmentationNetwork
from .dynunet_block import UnetOutBlock, UnetResBlock
from timm.models.layers import trunc_normal_
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from .layers import LayerNorm
from .dynunet_block import get_conv_layer, UnetResBlock
import math
import pywt
from torch.autograd import Function
einops, _ = optional_import("einops")
import math


class IDWT_Function3D(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, D, H, W = x.shape
        x = x.view(B, 8, -1, D, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, D, H, W)
        filters = filters.repeat(C, 1, 1, 1, 1)
        x = torch.nn.functional.conv_transpose3d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, D, H, W = ctx.shape
            C = C // 8
            dx = dx.contiguous().float()

            w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh = torch.unbind(filters, dim=0)
            x_lll = torch.nn.functional.conv3d(dx, w_lll.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_llh = torch.nn.functional.conv3d(dx, w_llh.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_lhl = torch.nn.functional.conv3d(dx, w_lhl.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_lhh = torch.nn.functional.conv3d(dx, w_lhh.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_hll = torch.nn.functional.conv3d(dx, w_hll.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_hlh = torch.nn.functional.conv3d(dx, w_hlh.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_hhl = torch.nn.functional.conv3d(dx, w_hhl.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            x_hhh = torch.nn.functional.conv3d(dx, w_hhh.unsqueeze(1).expand(C, -1, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_lll, x_llh, x_lhl, x_lhh, x_hll, x_hlh, x_hhl, x_hhh], dim=1)
        return dx, None

class IDWT_3D(nn.Module):
    def __init__(self, wavename):
        super(IDWT_3D, self).__init__()
        w = pywt.Wavelet(wavename)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        LL = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        LH = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        HL = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        HH = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        
        
# 从2维滤波器到3维滤波器
        w_lll = LL.unsqueeze(2) * rec_lo.unsqueeze(0).unsqueeze(1)
        w_llh = LL.unsqueeze(2) * rec_hi.unsqueeze(0).unsqueeze(1)
        w_lhl = LH.unsqueeze(2) * rec_lo.unsqueeze(0).unsqueeze(1)
        w_lhh = LH.unsqueeze(2) * rec_hi.unsqueeze(0).unsqueeze(1)
        w_hll = HL.unsqueeze(2) * rec_lo.unsqueeze(0).unsqueeze(1)
        w_hlh = HL.unsqueeze(2) * rec_hi.unsqueeze(0).unsqueeze(1)
        w_hhl = HH.unsqueeze(2) * rec_lo.unsqueeze(0).unsqueeze(1)
        w_hhh = HH.unsqueeze(2) * rec_hi.unsqueeze(0).unsqueeze(1)


        w_lll = w_lll.unsqueeze(0).unsqueeze(1)
        w_llh = w_llh.unsqueeze(0).unsqueeze(1)
        w_lhl = w_lhl.unsqueeze(0).unsqueeze(1)
        w_lhh = w_lhh.unsqueeze(0).unsqueeze(1)
        w_hll = w_hll.unsqueeze(0).unsqueeze(1)
        w_hlh = w_hlh.unsqueeze(0).unsqueeze(1)
        w_hhl = w_hhl.unsqueeze(0).unsqueeze(1)
        w_hhh = w_hhh.unsqueeze(0).unsqueeze(1)

        filters = torch.cat([w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function3D.apply(x, self.filters)


class DWT_3D(nn.Module):
    def __init__(self, wavename):
        super(DWT_3D, self).__init__()
        w = pywt.Wavelet(wavename)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])


# 从1维滤波器到2维滤波器
        LL = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        LH = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        HL = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        HH = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        
        
# 从2维滤波器到3维滤波器
        w_lll = LL.unsqueeze(2) * dec_lo.unsqueeze(0).unsqueeze(1)
        w_llh = LL.unsqueeze(2) * dec_hi.unsqueeze(0).unsqueeze(1)
        w_lhl = LH.unsqueeze(2) * dec_lo.unsqueeze(0).unsqueeze(1)
        w_lhh = LH.unsqueeze(2) * dec_hi.unsqueeze(0).unsqueeze(1)
        w_hll = HL.unsqueeze(2) * dec_lo.unsqueeze(0).unsqueeze(1)
        w_hlh = HL.unsqueeze(2) * dec_hi.unsqueeze(0).unsqueeze(1)
        w_hhl = HH.unsqueeze(2) * dec_lo.unsqueeze(0).unsqueeze(1)
        w_hhh = HH.unsqueeze(2) * dec_hi.unsqueeze(0).unsqueeze(1)
        
        self.register_buffer('w_lll', w_lll.unsqueeze(0).unsqueeze(0)) 
        self.register_buffer('w_llh', w_llh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lhl', w_lhl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lhh', w_lhh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hll', w_hll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hlh', w_hlh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hhl', w_hhl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hhh', w_hhh.unsqueeze(0).unsqueeze(0))
        
        self.w_lll = self.w_lll.to(dtype=torch.float32)
        self.w_llh = self.w_llh.to(dtype=torch.float32)
        self.w_lhl = self.w_lhl.to(dtype=torch.float32)
        self.w_lhh = self.w_lhh.to(dtype=torch.float32)
        self.w_hll = self.w_hll.to(dtype=torch.float32)
        self.w_hlh = self.w_hlh.to(dtype=torch.float32)
        self.w_hhl = self.w_hhl.to(dtype=torch.float32)
        self.w_hhh = self.w_hhh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_lll, self.w_llh, self.w_lhl, self.w_lhh, self.w_hll, self.w_hlh, self.w_hhl, self.w_hhh)

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh):
        x = x.contiguous()
        ctx.save_for_backward(w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_lll = torch.nn.functional.conv3d(x, w_lll.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_llh = torch.nn.functional.conv3d(x, w_llh.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_lhl = torch.nn.functional.conv3d(x, w_lhl.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_lhh = torch.nn.functional.conv3d(x, w_lhh.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_hll = torch.nn.functional.conv3d(x, w_hll.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_hlh = torch.nn.functional.conv3d(x, w_hlh.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_hhl = torch.nn.functional.conv3d(x, w_hhl.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        x_hhh = torch.nn.functional.conv3d(x, w_hhh.expand(dim, -1, -1, -1, -1), stride = 2, groups = dim)
        
        x = torch.cat([x_lll, x_llh, x_lhl, x_lhh, x_hll, x_hlh, x_hhl, x_hhh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh = ctx.saved_tensors
            B, C, D, H, W = ctx.shape
            dx = dx.view(B, 8, -1, D//2, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, D//2, H//2, W//2).float()
            filters = torch.cat([w_lll, w_llh, w_lhl, w_lhh, w_hll, w_hlh, w_hhl, w_hhh], dim=0).repeat(C, 1, 1, 1, 1).float()
            #dx = torch.nn.functional.conv_transpose3d(dx, filters, stride=2,output_padding=(1, 0, 0) , groups=C)
            dx = torch.nn.functional.conv_transpose3d(dx, filters, stride=2, groups=C)
            #print('-----dx-----',dx.shape)
        return dx, None, None, None, None, None, None, None, None, None

class TransformerBlock(nn.Module):


    def __init__(
            self,
        
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.wcda_block = WCDA(input_size=input_size ,hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        y = self.pos_embed

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.wcda_block(self.norm(x),H,W,D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
    
    
class TransformerBlock_O(nn.Module):


    def __init__(
            self,
        
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = WCDA_O(input_size=input_size ,hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        y = self.pos_embed

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x),H,W,D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x

class WaveU3SEncoder(nn.Module):
    def __init__(self, input_size=[24 * 48 * 48, 12 * 24 * 24, 6 * 12 * 12, 3 * 6 * 6],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.15 ,**kwargs):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            DWT_3D(wavename = 'haar'),
            DWT_3D(wavename = 'haar'),
            nn.Conv3d(64, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm3d(32, eps=1e-5, momentum=0.1),

        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
            DWT_3D(wavename = 'haar'),
            nn.Conv3d( dims[i]*8,  dims[i + 1], kernel_size=1, padding=0, stride=1),
            nn.BatchNorm3d(dims[i + 1], eps=1e-5, momentum=0.1),
            
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        for i in range(4):
            stage_blocks = []
            if i == 3:
                for j in range(depths[i]):
                    stage_blocks.append(TransformerBlock_O(input_size=input_size[i],hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                         dropout_rate=transformer_dropout_rate, pos_embed=True))
            else:
                for j in range(depths[i]):
                    stage_blocks.append(TransformerBlock(input_size=input_size[i],hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                         dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        #print(x.shape)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
    
class WCDA_O(nn.Module):

    
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x,H,W,D):
        B, N, C = x.shape
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        x = x.view(B, H, W, D, C).permute(0, 4, 1, 2, 3)  #[B,C,H,W,D]
        #([4, 32, 24, 24, 24])
        #print(x.shape)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)
        
        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x


class WCDA(nn.Module):

    def __init__(self, input_size,hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)

        self.E  = nn.Linear(input_size//8, proj_size) 
        self.F  = nn.Linear(input_size//8, proj_size)
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

        
        self.reduce = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 8, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm3d(hidden_size // 8, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        
        self.reduce2 = nn.Sequential(
            nn.Conv3d(hidden_size*(8**(int((math.log((input_size/4)**(1/3), 2)-2)))), hidden_size, kernel_size=1, padding=0, stride=1,groups=1),
            nn.BatchNorm3d(hidden_size, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
                
        self.filter = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm3d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ) 
        
        self.dwt =  DWT_3D(wavename = 'haar')
        self.idwt = IDWT_3D(wavename = 'haar')
        
        self.qk_embed = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, stride=1)
        
        self.qk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2)
        )
        self.proj = nn.Linear(hidden_size+hidden_size//8, hidden_size)
        
    def forward(self, x, H, W, D):
        
        B, N, C = x.shape  
        qkvv = self.qkvv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        v_CA, q_S = qkvv[0], qkvv[1]
        x = x.view(B, H, W, D, C).permute(0, 4, 1, 2, 3) 
        x_dwt = self.dwt(self.reduce(x))
        x_dwt_f = self.filter(x_dwt)
        qk = self.qk_embed(x_dwt_f).reshape(B, C, -1).permute(0, 2, 1)
        qk = self.qk(qk).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_c, k_CA= qk[0],qk[1]
        x_idwt = self.idwt(x_dwt)
        for i in range(int(math.log(D/3, 2))-2):
            x_dwt_f = self.dwt(x_dwt_f)
        x_dwt_f = self.reduce2(x_dwt_f)
        qk = self.qk_embed(x_dwt_f).reshape(B, C, -1).permute(0, 2, 1)#[1,36,64]
        qk = self.qk(qk).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_SA, v_SA= qk[0],qk[1]
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-3)*x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)#[B,H*W,c/4]([1, 576, 16])
        q_c = q_c.transpose(-2, -1)
        k_SA = k_SA.transpose(-2, -1)
        k_CA = k_CA.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)
        q_c = torch.nn.functional.normalize(q_c, dim=-1)
        q_S = torch.nn.functional.normalize(q_S, dim=-1)
        k_SA = torch.nn.functional.normalize(k_SA, dim=-1)
        k_CA = torch.nn.functional.normalize(k_CA, dim=-1)
        attn_CA = (q_c @ k_CA.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        attn_SA = (q_S@ k_SA) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}



class WaveU3SUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out

class WaveU3S(SegmentationNetwork):


    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,
    ) -> None:

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)#(2,4,4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages #[96,96,96]->[3,3,3]
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.WaveU3S_encoder = WaveU3SEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = WaveU3SUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=6 * 12 * 12,
        )
        self.decoder4 = WaveU3SUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=12 * 24 * 24,
        )
        self.decoder3 = WaveU3SUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=24 * 48 * 48,
        )
        self.decoder2 = WaveU3SUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=96 * 192 * 192,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.WaveU3S_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)
        return logits[0]
