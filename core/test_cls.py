import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation

def test(net, criterion, testloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0
    torch.cuda.empty_cache()
    results = dict()
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    results['ACC'] = acc

    return results