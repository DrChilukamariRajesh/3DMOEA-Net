from torch.optim import Adam, AdamW, Adamax, SGD
from monai.losses import DiceLoss, FocalLoss, TverskyLoss, DiceCELoss

acts = ('relu', 'leakyrelu', 'prelu', 'memswish')
norms = ('BATCH', ('GROUP', {'num_groups': 1}), 'INSTANCE', '')

def loss_functions(l):
    if l==0:
        return DiceLoss(to_onehot_y=False, sigmoid=True)
    elif l == 1:
        return DiceCELoss(to_onehot_y=False, sigmoid=True)
    elif l == 2:
        return TverskyLoss(include_background=True, to_onehot_y=False, sigmoid=True)
    else:
        return FocalLoss(include_background=True, to_onehot_y=False)
 
def optimizers(o, params,lr=1e-4):
    if o == 0:
        return SGD(params, lr=lr, weight_decay=1e-5, momentum=0.9)
    elif o == 1:
        return Adam(params, lr=lr, weight_decay=1e-5)
    elif o == 2:
        return AdamW(params, lr=lr, weight_decay=1e-5)
    else:
        return Adamax(params, lr=lr, weight_decay=1e-5)
    
def todec(b):
    return int(''.join(map(lambda x: str(int(x)), b)), 2)

def encoding(ch):
    b, a, n, j = [], [], [], 0
    
    for i in range(7):  #0-49
        b.append(todec(ch[j:j+3]))
        j += 3        
        a.append(acts[todec(ch[j:j+2])])
        j += 2        
        n.append(norms[todec(ch[j:j+2])])
        j += 2        
    upsn = ch[j:j+3]  #49-52
    j += 3
    ol = list(map(todec, (ch[j:j+2], ch[j+2:j+4])))  #52-56  optimizer-52,53  loss function-54,55
    return b, a, n, upsn, ol[0], ol[1]