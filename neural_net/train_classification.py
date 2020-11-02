# Author - Yossi Adi

from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def test(loader, model, cuda, verbose=True):
    model.eval()
    count_sum = 0
    with torch.no_grad():
        for data in loader:
            if cuda:
                data = data.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            count_sum += torch.sum(pred)
    return count_sum