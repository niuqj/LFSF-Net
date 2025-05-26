import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules.wt import WT,IWT,SAFB


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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

class FRB(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(FRB, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat1 = SAFB(num_filters,reduction=4)
        self.cat2 = SAFB(num_filters,reduction=4)
        self.cat3 = SAFB(num_filters,reduction=4)
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x)
        if x.shape[3] != x3.shape[3]:
            x = x[:, :, :, :x3.shape[3]]
        cat1 = self.cat1(x,x3)
        x = self.up3(cat1)
        cat2 = self.cat2(x,x2)
        x = self.up2(cat2)
        cat3 = self.cat3(x,x1)
        x = self.res_layer(cat3)
        return torch.tanh(self.output_layer(x))

class FRB_RGB(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(FRB_RGB, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_layer = nn.Conv2d(12, 12, kernel_size=kernel_size, padding=1)
        self.cat1 = SAFB(num_filters,reduction=4)
        self.cat2 = SAFB(num_filters,reduction=4)
        self.res_layer = nn.Conv2d(num_filters, 12, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x + x4)
        if x.shape[3] != x3.shape[3]:
            x = x[:, :, :, :x3.shape[3]]
        cat1 = self.cat1(x,x3)
        x = self.up3(cat1)
        cat2 = self.cat2(x,x2)
        x = self.up2(cat2)
        x = x + x1
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x))

class LFSFNet(nn.Module):
    def __init__(self, filters=32):
        super(LFSFNet, self).__init__()
        self.mag_y = FRB(filters // 2)
        self.yuv_init = nn.Conv2d(3,filters//2,kernel_size=3,stride=1,padding=1)
        self.rgb_init = nn.Conv2d(3,filters//2,kernel_size=3,stride=1,padding=1)
        self.cat = SAFB(filters//2,reduction=4)
        self.wt = WT(filters//2)
        self.process = FRB_RGB(filters*2)
        self.iwt = IWT(3)
        self.out1 = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)
        self.out2 = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv
    def ycbcr_to_rgb(self, image):
        y, u, v = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        r = y + 1.13983 * (v - 0.5)
        g = y - 0.39465 * (u - 0.5) - 0.58060 * (v - 0.5)
        b = y + 2.03211 * (u - 0.5)

        rgb = torch.stack((r, g, b), dim=1)
        return torch.clamp(rgb, 0.0, 1.0)
    def forward(self, inputs):
        _, _, H, W = inputs.shape
        ycbcr = self._rgb_to_ycbcr(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        image_fft = torch.fft.fft2(y, norm='forward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        mag_image = (mag_image - mag_image.min()) / (mag_image.max() - mag_image.min())
        mag_image1 = self.mag_y(mag_image)
        mag_image = mag_image * mag_image1
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='forward').real
        recombined = torch.cat([img_amp_enhanced,cb, cr], dim=1)
        out = self.ycbcr_to_rgb(recombined)     

        out_init = self.yuv_init(out)
        rgb_init = self.rgb_init(inputs)
        rgb = self.cat(out_init,rgb_init)
        wt = self.wt(rgb)
        process = self.process(wt)
        iwt = self.iwt(process)
        out = self.out1(iwt)
        out = self.out2(out)
        return out

