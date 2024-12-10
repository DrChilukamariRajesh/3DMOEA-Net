from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

class AverageMeter(object):
    def __init__(self):
        self.current_value = 0
        self.average_value = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.current_value = 0
        self.average_value = 0
        self.sum = 0
        self.count = 0

    def update(self, current_value, increment=1):
        self.current_value = current_value
        self.sum += current_value * increment
        self.count += increment
        self.average_value = self.sum / self.count

    @property
    def val(self):
        return self.average_value


#HD95
def cal_hd95(tensor1, tensor2):
    if len(tensor1.shape) > 4:
        distances = torch.cdist(tensor1.view(-1, tensor1.shape[-3]*tensor1.shape[-2]*tensor1.shape[-1]), 
                            tensor2.view(-1, tensor2.shape[-3]*tensor2.shape[-2]*tensor2.shape[-1]))
    else:
        distances = torch.cdist(tensor1, tensor2)
    num_points = distances.numel()
    k = int(0.95 * num_points)
    hd95, _ = torch.kthvalue(distances.view(-1), k)
    return hd95.item()