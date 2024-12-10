import torch
from torch import nn
import torch.nn.functional as F
from monai.networks import blocks
from monai.networks.blocks.mlp import MLPBlock


def CONV(s, i, o, k, n, a, d):
    return blocks.Convolution(spatial_dims=s, in_channels=i, out_channels=o, kernel_size=k, norm=n, act=a, dropout=d)

def UBB(s, i, o, k, n, a, d):
	return blocks.UnetBasicBlock(spatial_dims=s, in_channels=i, out_channels=o, kernel_size=k, stride=1, norm_name=n, act_name=a, dropout=d)

def RB(s, i, o, k, n, a, d):
	return blocks.ResBlock(spatial_dims=s, in_channels=i, norm=n, kernel_size=k, act=a)

def SERB(s, i, o, k, n, a, d):
	return blocks.SEResNetBottleneck(spatial_dims=s, inplanes=i, planes=4, groups=4, reduction=2, stride=1, downsample=None)

def P3AB(s, i, o, k, n, a, d):
	return blocks.P3DActiConvNormBlock(in_channel=i, out_channel=o, kernel_size=k, padding=1, act_name=a, norm_name=n)

def UrPUB(s, i, o, k, n, a, d):
	return blocks.UnetrPrUpBlock(spatial_dims=s, in_channels=i, out_channels=o, num_layer=1, kernel_size=k, stride=1, upsample_kernel_size=1, norm_name=n, conv_block=True, res_block=True)


def  MaxAvg(k):
	return blocks.MaxAvgPool(spatial_dims=3, kernel_size=k, stride=2, padding=0, ceil_mode=True)

def  Ups(s, i, o, k, n, a, d):  
	return blocks.FactorizedIncreaseBlock(in_channel=i, out_channel=o, spatial_dims=s, act_name=a, norm_name=n) 

def  Transp(s, i, o, k, n, a, d):
	return blocks.UpSample(spatial_dims=3, in_channels=i, out_channels=o, scale_factor=2, kernel_size=None, size=None, mode='deconv', pre_conv='default', interp_mode='LINEAR', align_corners=True, bias=True, apply_pad_pool=True)  

def TransPre(i, n, a, st):
    return blocks.Convolution(spatial_dims=3, in_channels=i, out_channels=8, kernel_size=1, norm=n, act=a, strides=st)

def  TransPost(o, st):
	return blocks.UpSample(spatial_dims=3, in_channels=8, out_channels=o, scale_factor=st, kernel_size=None, size=None, mode='deconv', pre_conv='default', interp_mode='LINEAR', align_corners=True, bias=True, apply_pad_pool=True)  


class EfficientNetBlock3D(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, n, a, stride=1, expand_ratio=6):
        super(EfficientNetBlock3D, self).__init__()
        
        # Expand the input channels using the expand_ratio
        expand_channels = in_channels * expand_ratio
        self.n = n
        self.a = a      
        self.nog1 = 4 if expand_channels%4==0 else (3 if expand_channels%3==0 else 1)
        self.nog2 = 4 if out_channels%4==0 else (3 if out_channels%3==0 else 1)  
        
        # Depth-wise convolution layer
        self.conv1 = nn.Conv3d(in_channels, expand_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(expand_channels)
        self.in1 = nn.InstanceNorm3d(expand_channels)
        self.gn1 = nn.GroupNorm(self.nog1, expand_channels)
        
        # Point-wise convolution layer
        self.conv2 = nn.Conv3d(expand_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.in2 = nn.InstanceNorm3d(out_channels)
        self.gn2 = nn.GroupNorm(self.nog2, out_channels)
        
        # Squeeze and Excitation layer 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(expand_channels, expand_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(expand_channels // 4, expand_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.swish = nn.SiLU()
        self.rel = nn.LeakyReLU()  #nn.ReLU()
        self.lrel = nn.LeakyReLU()
        self.prel = nn.PReLU()
        
    def forward(self, x):
        identity = x
        
        # Depth-wise convolution layer
        x = self.conv1(x)
        if self.n == 'BATCH':
            x = self.bn1(x)
        elif self.n == 'INSTANCE': 
            x = self.in1(x)
        elif 'GROUP' in self.n:
            x = self.gn1(x)
        # print(x.shape)
        
        if self.a == 'relu': 
            x = self.rel(x)
        elif self.a == 'leakyrelu': 
            x = self.lrel(x)
        elif self.a == 'prelu': 
            x = self.prel(x)
        else:
            x = self.swish(x)            
                
        se_weights = self.se(x)
        x = x * se_weights
        
        # Point-wise convolution layer
        x = self.conv2(x)
        if self.n == 'BATCH':
            x = self.bn2(x)
        elif self.n == 'INSTANCE': 
            x = self.in2(x)
        elif 'GROUP' in self.n:
            x = self.gn2(x)
        
        if self.a == 'relu': 
            x = self.rel(x)
        elif self.a == 'leakyrelu': 
            x = self.lrel(x)
        elif self.a == 'prelu': 
            x = self.prel(x)
        else:
            x = self.swish(x)
            
        # Add identity connection
        if x.shape == identity.shape:
            x += identity
             
        return x
		
    
class CMTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n, a, kernel_size=1, bias=False):
        super(CMTBlock, self).__init__()
        self.n = n
        self.a = a
        self.inter_channels = out_channels
        self.out_channels = out_channels
        self.n_heads = 4 if self.out_channels%2 == 0 else 3
        self.stride = 8 if self.out_channels<= 64 else 2
        self.nog = 4 if self.inter_channels%4==0 else (3 if self.inter_channels%3==0 else 1) 

        self.conv = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1, stride=self.stride, bias=bias)
        self.bn3 = nn.BatchNorm3d(self.inter_channels)
        self.in3 = nn.InstanceNorm3d(self.inter_channels)
        self.gn3 = nn.GroupNorm(self.nog, self.inter_channels)
        self.rel = nn.ReLU(inplace=True)
        self.swish = nn.SiLU()
        self.lrel = nn.LeakyReLU()
        self.prel = nn.PReLU()
        self.transformer = nn.TransformerEncoderLayer(d_model=self.inter_channels, nhead=self.n_heads)
        
        self.trans = blocks.UpSample(spatial_dims=3, in_channels=out_channels, out_channels=out_channels, scale_factor=2,\
                        kernel_size=1, size=None, mode='deconv', pre_conv='default', interp_mode='LINEAR',\
                        align_corners=True, bias=True, apply_pad_pool=True) 
    
    def forward(self, x):
        conv_out = self.conv(x)        
        if self.n == 'BATCH':
            conv_out = self.bn3(conv_out)
        elif self.n == 'INSTANCE': 
            conv_out = self.in3(conv_out)
        elif 'GROUP' in self.n:
            conv_out = self.gn3(conv_out)
            
        if self.a == 'relu': 
            conv_out = self.rel(conv_out)
        elif self.a == 'leakyrelu': 
            conv_out = self.lrel(conv_out)
        elif self.a == 'prelu': 
            conv_out = self.prel(conv_out)
        else:
            conv_out = self.swish(conv_out)
        
        N, C, D, H, W = conv_out.size()
        transformer_out = conv_out.permute(2, 3, 4, 0, 1).contiguous().view(D * H * W, N, C)
        transformer_out = self.transformer(transformer_out)
        transformer_out = transformer_out.view(D, H, W, N, C).permute(3, 4, 0, 1, 2).contiguous()
        
        out = torch.add(conv_out, transformer_out)
        if self.stride == 2:
            out = self.trans(out)
        else:
            out = self.trans(self.trans(self.trans(out)))
        return out
    

class TransformerBlock3D(nn.Module):
    def __init__(self, out_channels, in_channels=8, height=12, width=12, depth=12, dropout=0.1):
        super().__init__()
        self.bs4 = 4
        self.bs1 = 1  
        
        self.norm1 = nn.LayerNorm([self.bs4 * in_channels, height, width* depth])        
        self.norm2 = nn.LayerNorm([self.bs4, in_channels, height, width, depth])
        self.mlp = MLPBlock(depth, height*width*in_channels*self.bs4, dropout)  
        
        self.norm11 = nn.LayerNorm([self.bs1 * in_channels, height, width* depth])        
        self.norm21 = nn.LayerNorm([self.bs1, in_channels, height, width, depth])
        self.mlp1 = MLPBlock(depth, height*width*in_channels*self.bs1, dropout)  
        
        self.self_attn = nn.MultiheadAttention(embed_dim=width* depth, num_heads=in_channels, dropout=dropout)
        self.dropout = nn.Dropout(dropout)      
        
    def forward(self, x):
        batch_size, in_channels, height, width, depth = x.shape
        out = x.view(batch_size * in_channels, height, width* depth)
        out = self.norm1(out) if batch_size == 4 else self.norm11(out)
        out, _ = self.self_attn(out, out, out)
        out = out.view(batch_size, in_channels, height, width, depth)
        out = out + x
        out1 = self.norm2(out) if batch_size == 4 else self.norm21(out)
        out1 = self.mlp(out1)  if batch_size == 4 else self.mlp1(out1)          
        out1 = out1 + out
        return out1
    