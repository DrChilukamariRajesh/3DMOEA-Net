from torch import nn
from .blocks import *
from .block import Block3D

class MOED3D(nn.Module):
    def __init__(self, b, acts, norms, upsn, input_channels = 4, num_classes=3, initial_kernel=16):
        super().__init__()
        self.b = b
        self.upsn = upsn
        self.acts = acts
        self.norms = norms
        self.num_classes = num_classes
        self.initial_kernel = initial_kernel
        self.input_channels = input_channels        
        self.r1 = Block3D(bn=self.b[0], inc=self.input_channels,   outc=self.initial_kernel,   kernel=3, norm=self.norms[0], act=self.acts[0], dr=None)
        self.r2 = Block3D(bn=self.b[1], inc=self.initial_kernel*2, outc=self.initial_kernel*2, kernel=3, norm=self.norms[1], act=self.acts[1], dr=None)
        self.r3 = Block3D(bn=self.b[2], inc=self.initial_kernel*4, outc=self.initial_kernel*4, kernel=3, norm=self.norms[2], act=self.acts[2], dr=None)
        self.r4 = Block3D(bn=self.b[3], inc=self.initial_kernel*8, outc=self.initial_kernel*8, kernel=3, norm=self.norms[3], act=self.acts[3], dr=None)
        self.r5 = Block3D(bn=self.b[4], inc=self.initial_kernel*4, outc=self.initial_kernel*4, kernel=3, norm=self.norms[4], act=self.acts[4], dr=None)
        self.r6 = Block3D(bn=self.b[5], inc=self.initial_kernel*2, outc=self.initial_kernel*2, kernel=3, norm=self.norms[5], act=self.acts[5], dr=None)
        self.r7 = Block3D(bn=self.b[6], inc=self.initial_kernel,   outc=self.num_classes,       kernel=3, norm=self.norms[6], act=self.acts[6], dr=None)        
        self.m = MaxAvg(k=3)
        self.u1 = Ups(s=3, i=self.initial_kernel*8, o=self.initial_kernel*4, k=3, n=self.norms[4], a=self.acts[4], d=None)
        self.u2 = Ups(s=3, i=self.initial_kernel*4, o=self.initial_kernel*2, k=3, n=self.norms[5], a=self.acts[5], d=None)
        self.u3 = Ups(s=3, i=self.initial_kernel*2, o=self.initial_kernel,   k=3, n=self.norms[6], a=self.acts[6], d=None)        
        self.t1 = Transp(s=3, i=self.initial_kernel*8, o=self.initial_kernel*4, k=3, n=self.norms[4], a=self.acts[4], d=None)
        self.t2 = Transp(s=3, i=self.initial_kernel*4, o=self.initial_kernel*2, k=3, n=self.norms[5], a=self.acts[5], d=None)
        self.t3 = Transp(s=3, i=self.initial_kernel*2, o=self.initial_kernel,   k=3, n=self.norms[6], a=self.acts[6], d=None)        

    def forward(self, x):
        b, c, h, w, d = x.shape       
        x1 = self.r1(x, 1)
        x = self.m(x1)        
        x2 = self.r2(x, 2)
        x = self.m(x2)        
        x3 = self.r3(x, 3)
        x = self.m(x3)        
        x4 = self.r4(x, 4)
        if self.upsn[0] == 0:
            xu1 = self.t1(x4)
        elif self.upsn[0] == 1:
            xu1 = self.u1(x4)
        else:
            raise "Invalid upsampling "        
        xu1x3 = xu1+x3
        x5 = self.r5(xu1x3, 3)    
        if self.upsn[1] == 0:
            xu2 = self.t2(x5)
        elif self.upsn[1] == 1:
            xu2 = self.u2(x5)
        else:
            raise "Invalid upsampling "        
        xu2x2 = xu2+x2
        x6 = self.r6(xu2x2, 2)
        if self.upsn[2] == 0:
            xu3 = self.t3(x6)
        elif self.upsn[2] == 1:
            xu3 = self.u3(x6)
        else:
            raise "Invalid upsampling "        
        xu3x1 = xu3+x1
        x7 = self.r7(xu3x1, 1)        
        return x7