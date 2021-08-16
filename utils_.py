import os
import sys
import errno
import os.path as osp
import numpy as np
import torch



def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
        self.avg = self.sum / self.count

class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_networks(networks,thre, result_dir, is_best,name='', loss='', criterion=None):
    if is_best:
        result_dir = os.path.join(result_dir, 'model_best')
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    state={'thre':thre,
           'state_dict': networks.state_dict()}
    filename = '{}/checkpoints/{}_{}.pth.tar'.format(result_dir, name, loss)
    torch.save(state, filename)
    if criterion:
        state = {'thre': thre,
                 'state_dict': networks.state_dict(),
                 'criterion_state_dict': criterion.state_dict()}
        torch.save(state, filename)

def save_GAN(netG, netD, result_dir, name=''):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = netG.state_dict()
    filename = '{}/{}_G.pth'.format(result_dir, name)
    torch.save(weights, filename)
    weights = netD.state_dict()
    filename = '{}/{}_D.pth'.format(result_dir, name)
    torch.save(weights, filename)


def load_networks(networks,  result_dir, name='', loss='', criterion=None,best_checkpoint=False):
    if best_checkpoint:
        filename='{}/model_best/checkpoints/{}_{}.pth.tar'.format(result_dir, name, loss)
    else:
        filename = '{}/checkpoints/{}_{}.pth.tar'.format(result_dir, name, loss)
    checkpoint=torch.load(filename)
    thre = checkpoint['thre']
    networks.load_state_dict(checkpoint['state_dict'])
    if criterion:
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
    return networks,thre,criterion