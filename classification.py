import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models import gan
from models.effnetv2 import effnetv2_m
from models.models import classifier32, classifier32ABN
from models.resnet import ResNet34,ResNet50
from models.resnetABN import resnet34ABN,resnet50ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR,ImageNet_OSR,ImageNet_CLS
from utils import Logger, save_networks, load_networks
from core import train_cls, test_cls

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='imagenet', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet|imagenet")
parser.add_argument('--dataroot', type=str, default='/data2')
parser.add_argument('--outf', type=str, default='./log_cls')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='resnet34')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)


best_acc = 0

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif 'imagenet' in options['dataset']:
        Data = ImageNet_CLS(known=options['known'],dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader

    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes
    global best_acc

    # Model
    print("Creating model: {}".format(options['model']))
    net=ResNet34(options['num_classes'])
        #net = effnetv2_m(num_classes=options['num_classes'])
       # net = classifier32(num_classes=options['num_classes'])
    feat_dim = 128

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], 50)
    else:
        file_name = '{}_{}'.format(options['model'], options['loss'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test_cls.test(net, criterion, testloader,  epoch=0, **options)
        print("Acc (%): {:.3f}\t".format(results['ACC']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    if options['dataset'] == 'tiny_imagenet' or options['dataset'] == 'imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])

    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        train_cls.train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test_cls.test(net, criterion, testloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t [BEST: {:.3f}/({} Epoch)".format(results['ACC'],best_acc,epoch+1))

            is_best = False
            if results['ACC'] > best_acc:
                is_best = True
                best_acc = results['ACC']

            save_networks(net, model_path, is_best,file_name, criterion=criterion)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    from split import get_splits
    splits = get_splits(args.dataset, num_split=0)
    known = splits['known_classes']

    if options['dataset'] == 'tiny_imagenet':
        img_size= 64
        options['lr'] = 0.001
    elif options['dataset'] == 'imagenet':
        img_size = 224
        options['lr'] = 0.001
    else:
        img_size = 32
        options['lr'] = args.lr

    options.update(
        {
            'known': known,
            'img_size': img_size
        }
    )

    results = dict()
    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if options['dataset'] == 'cifar100':
        file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
    else:
        file_name = options['dataset'] + '.csv'

    res = main_worker(options)
    results[str(0)] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name))