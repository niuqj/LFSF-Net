from functools import partial
import math
import pywt
import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch
from torch import nn
import typing as t
from einops import rearrange

    
class SpatialAttention(nn.Module):
    def __init__(self,dim):
        super(SpatialAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h_, w_ = x.size()
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        # 使用sigmoid进行归一化
        x_h_attn = self.sigmoid(x_h).view(b, c, h_, 1)  # [b, c, h_, 1]
        x_w_attn = self.sigmoid(x_w).view(b, c, 1, w_)  # [b, c, 1, w_]
        x = x * x_h_attn * x_w_attn
        return x    
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn 
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class SAFB(nn.Module):
    def __init__(self, dim, reduction=8):
        super(SAFB, self).__init__()
        self.sa = SpatialAttention(dim)
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        if x.shape != y.shape:
            min_h = min(x.shape[2], y.shape[2])
            min_w = min(x.shape[3], y.shape[3])
            x = x[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WT(nn.Module):
    def __init__(self, dim, wave='db1', wt_type='dec', bias=True):
        super(WT, self).__init__()
        self.dim = dim
        self.wave = wave
        self.wt_type = wt_type

        self.dec_filters, self.rec_filters = create_wavelet_filter(wave, dim, dim, torch.float)
        self.dec_filters = nn.Parameter(self.dec_filters, requires_grad=False)
        self.rec_filters = nn.Parameter(self.rec_filters, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.dec_filters)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.rec_filters)

        self.conv_cd = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1)  
        self.conv_hd = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1) 
        self.conv_vd = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1)  
        self.conv_ad = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1)

        self.conv_final = nn.Conv2d(dim * 4, dim * 4, 3, padding=1, bias=True)


    def forward(self, x):
        x_wave = self.wt_function(x)  
        LL = x_wave[:, :, 0, :, :]
        HL = x_wave[:, :, 1, :, :]
        LH = x_wave[:, :, 2, :, :]
        HH = x_wave[:, :, 3, :, :]

        res_cd = self.conv_cd(LL)
        res_hd = self.conv_hd(HL)
        res_vd = self.conv_vd(LH)
        res_ad = self.conv_ad(HH)

        res = torch.cat((res_cd, res_hd, res_vd, res_ad), dim=1) 

        output = self.conv_final(res)
        return output
    

class IWT(nn.Module):
    def __init__(self, dim, wave='db1', wt_type='dec', bias=True):
        super(IWT, self).__init__()
        self.dim = dim
        self.wave = wave
        self.wt_type = wt_type

        self.dec_filters, self.rec_filters = create_wavelet_filter(wave, dim, dim, torch.float)
        self.dec_filters = nn.Parameter(self.dec_filters, requires_grad=False)
        self.rec_filters = nn.Parameter(self.rec_filters, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.dec_filters)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.rec_filters)

        self.conv_final = nn.Conv2d(dim, dim, 3, padding=1, bias=True)


    def forward(self, x):
        res_wave = x.view(x.shape[0], self.dim, 4, x.shape[2], x.shape[3])
        x_recon = self.iwt_function(res_wave)  
        output = self.conv_final(x_recon)
        return output

def pad_if_needed(x):


    pad_h = (x.shape[2] % 2 != 0)  
    pad_w = (x.shape[3] % 2 != 0)  
    if pad_h or pad_w:

        padding = (0, int(pad_w), 0, int(pad_h))  
        x = nn.functional.pad(x, padding, mode='replicate')
    return x