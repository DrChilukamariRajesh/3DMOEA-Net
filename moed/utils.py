from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import csv
import time
import torch
import pandas
import numpy as np
from math import tanh
from thop import profile
from datetime import datetime
from moed.moed3d import MOED3D
from moed.searchspace import *


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    
def cal_params(model):
    return sum(p.numel() for p in model.parameters())


def convert_Time(ttime):
    return datetime.fromtimestamp(ttime).strftime('%Y-%m-%d %H:%M:%S')


def checkCache(file_names, ptcl):
    for file_name in file_names:
        f = pandas.read_csv(file_name)
        if ptcl in f['Ch'].values:
            tds = f[f['Ch'] == ptcl]['Val_Dice']
            pms = f[f['Ch'] == ptcl]['Params']
            flops = f[f['Ch'] == ptcl]['FLOPS']
            print(f"Ran already, returning saved values. Diceloss: {1-tds}, #of params: {pms} M, FLOPS: {flops} Gmacs")
            return True, 1-tds, pms, flops
    return False, None, None, None    
    
    
def create_res_file(file_name_res):
    if not os.path.exists(file_name_res):
        print("writing new res file")
        with open(file_name_res,'a') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(["Model_name", "Val_Dice", "Params", "FLOPS", "FPS", "Total time", "Dice", "dice_tc", "dice_wt", "dice_et", \
                 "HD95", "HD_tc", "HD_wt", "HD_et", "Ch", "start", "end", "best_metric_epoch", "Task"])
            
def saveValues(file_name_res,l):
    with open(file_name_res,'a') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(l)
    print(l)
    

def calc_pff(model, img_size=96, inc=4, outc=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(inc, outc, img_size, img_size, img_size).to(device)
    macs, params = profile(model, inputs=(input_tensor,))
    model.eval()
    n_frames = 100  # Number of frames for FPS estimation
    total_time = 0.0

    with torch.no_grad():
        for _ in range(n_frames):
            start_time = time.time()
            output = model(input_tensor.to(device))
            end_time = time.time()
            total_time += end_time - start_time
    fps = n_frames / total_time
    
    print(f"Parameters: {params / 1e6} M")
    print(f"FLOPs: {macs / 1e9} Gmacs")
    print(f"FPS: {fps}")
    return params/1e6, macs/1e9, fps

   
def float2bin(arr, fun='tan'):
    for i in range(arr.shape[0]):
        if arr[i] not in [0,1]:
            if fun == 'tan':
                arr[i] = 0 if tanh(arr[i]) >= np.random.rand() else 1
    return arr


def encoding_ch(ch, file_name_res):
    ptcl = ' '.join(map(str, ch))
    b, a, n, upsn, o, l = encoding(ch)
    model = MOED3D(b, a, n, upsn, input_channels=4, num_classes=3, initial_kernel=16)
    loss_function = loss_functions(l)    
    optimizer = optimizers(o, model.parameters())   
    return model, optimizer, loss_function, ptcl
