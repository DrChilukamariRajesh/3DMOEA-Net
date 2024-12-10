from .blocks import *
from torch import nn

class Block3D(nn.Module):
    def __init__(self, bn, inc, outc, kernel, norm, act, dr):
        super().__init__()
        self.bn = bn
        self.inc = inc
        self.outc = outc
        self.kernels = kernel
        self.norms = norm
        self.acts = act
        self.do = dr
        
        #block0
        self.ubb = UBB(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        
        #block1
        self.rb = RB(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        self.conv = CONV(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        
        #block2
        self.serb = SERB(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        self.conv1 = CONV(s=3, i=16, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)                
        
        #block3
        self.p3ab = P3AB(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        
        #block4
        self.urpub = UrPUB(s=3, i=self.inc, o=self.outc, k=self.kernels, n=self.norms, a=self.acts, d=self.do)
        
        #block5
        self.eff = EfficientNetBlock3D(in_channels=self.inc, out_channels=self.outc, kernel_size=self.kernels, n=self.norms, a=self.acts)  
        
        #block6
        self.tb = TransformerBlock3D(out_channels=self.outc)
        self.tpr1 = TransPre(i=self.inc, n=self.norms, a=self.acts, st=8)
        self.tpo1 = TransPost(o=self.outc, st=8)
        
        self.tpr2 = TransPre(i=self.inc, n=self.norms, a=self.acts, st=4)
        self.tpo2 = TransPost(o=self.outc, st=4)
        
        self.tpr3 = TransPre(i=self.inc, n=self.norms, a=self.acts, st=2)
        self.tpo3 = TransPost(o=self.outc, st=2)
        
        self.tpr4 = TransPre(i=self.inc, n=self.norms, a=self.acts, st=1)
        self.tpo4 = TransPost(o=self.outc, st=1)
        
        #block7
        self.cmt = CMTBlock(in_channels=self.inc, out_channels=self.outc, n=self.norms, a=self.acts)
        
        
    def forward(self, x, pos):        
        # print(f"position is {pos}")
        if self.bn == 0:
            # print("UBB")
            return self.ubb(x)
        
        elif self.bn == 1:
            # print("RB")
            x = self.rb(x)            
            if self.inc == self.outc:
                return x
            return self.conv(x)
        
        elif self.bn == 2:
            # print("ResNet bottleneck with a Squeeze-and-Excitation module")
            x = self.serb(x)
            return self.conv1(x)
        
        elif self.bn == 3:
            # print("P3AB")
            return self.p3ab(x)
        
        elif self.bn == 4:
            # print("UrPUB")
            return self.urpub(x)
        
        
        elif self.bn == 5:
            # print("Effiecient module")
            return self.eff(x)
        
        elif self.bn == 6:
            # print("TransformerBlock3D")
            if pos == 1:
                x = self.tpo1(self.tb(self.tpr1(x)))
            elif pos == 2:
                x = self.tpo2(self.tb(self.tpr2(x)))
            elif pos == 3:
                x = self.tpo3(self.tb(self.tpr3(x)))
            elif pos == 4:
                x = self.tpo4(self.tb(self.tpr4(x)))
            else:
                raise "Invalid position"         
            return x
        
        elif self.bn == 7:
            # print("CMT")
            return self.cmt(x)
			
        else:
            raise "Invalid Block number"