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
from shutil import copy
from tensorboardX import SummaryWriter
from models import gan
from models.models import classifier32, classifier32ABN
from models.resnet import ResNet34,ResNet50
from models.resnetABN import resnet34ABN,resnet50ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR,ImageNet_OSR
from utils_ import Logger, save_networks, load_networks
from core import train, train_cs, test_
from loss.center_loss import CenterLoss
from loss.ContrLoss import SupConLoss

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='imagenet', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet|imagenet")
parser.add_argument('--dataroot', type=str, default='/data2')
parser.add_argument('--outf', type=str, default='./log_')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.00005, help="learning rate for gan")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--weight_l2', type=float, default=1., help="weight for l2-distance in unknown detection")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--alpha', default = 10, type = int, help='Magnitude of the anchor point')
parser.add_argument('--lbda', default = 0.1, type = float, help='Weighting of Anchor loss component in CAC Loss')
parser.add_argument('--weight_cent', type=float, default=1., help="weight for center loss")
parser.add_argument('--weight_contr', type=float, default=1., help="weight for center loss")
parser.add_argument('--temp_contr', type=float, default=0.1,help='temperature for Controstive loss function')
parser.add_argument('--model', type=str, default='Resnet34')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--best_checkpoint', action='store_true', help="load best model's checkpoint", default=False)
parser.add_argument('--anchored_center_loss', action='store_true', help="add anchored center loss", default=False)
parser.add_argument('--learnable_center_loss', action='store_true', help="add learnable center loss", default=False)
parser.add_argument('--contr_loss', action='store_true', help="add learnable center loss", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=True)


global best_acc
global best_epoch

def main_worker(options,writer):
    torch.manual_seed(options['seed'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    train_log_feature_best_path = os.path.join(options['outf'], 'results', 'train_fea_best.txt'.format(options['seed']))
    train_log_gt_best_path = os.path.join(options['outf'], 'results', 'train_gt_best.txt'.format(options['seed']))
    train_log_pred_best_path = os.path.join(options['outf'], 'results', 'train_pred_best.txt'.format(options['seed']))

    eval_log_feature_all = open(os.path.join(options['outf'], 'results', 'eval_fea_all.txt'.format(options['seed'])), 'w')
    eval_log_gt_all = open(os.path.join(options['outf'], 'results', 'eval_gt_all.txt'.format(options['seed'])), 'w')
    eval_log_pred_all = open(os.path.join(options['outf'], 'results', 'eval_pred_all.txt'.format(options['seed'])), 'w')

    test_log_feature = open(os.path.join(options['outf'], 'results', 'test_fea_all.txt'.format(options['seed'])),'w')
    test_log_gt = open(os.path.join(options['outf'], 'results', 'test_gt_all.txt'.format(options['seed'])), 'w')
    test_log_pred = open(os.path.join(options['outf'], 'results', 'test_pred_all.txt'.format(options['seed'])), 'w')

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
        Data = ImageNet_OSR(known=options['known'], unknown=options['unknown'],dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader,outloader = Data.train_loader, Data.test_loader,Data.out_loader

    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes
    best_acc=0
    best_epoch=0

    # Model


    print("Creating model: {}".format(options['model']))
    if options['cs']:
        net = resnet34ABN(num_classes=options['num_classes'])
        #net=effnetv2_mABN(num_classes=options['num_classes'])
        #net = classifier32ABN(num_classes=options['num_classes'])
    else:
        net=ResNet34(num_classes=options['num_classes'])
        #net = effnetv2_m(num_classes=options['num_classes'])
        #net = classifier32(num_classes=options['num_classes'])
    #feat_dim = 128
    feat_dim = 512

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' in options['dataset']:
            netG = gan.Generator(1, nz, 64, 3)
            netD = gan.Discriminator(1, 3, 64)
        else:
            netG = gan.Generator32(1, nz, 64, 3)
            netD = gan.Discriminator32(1, 3, 64)
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if options['anchored_center_loss']:
        # initialising with anchors
        anchors = torch.diag(torch.Tensor([options['alpha'] for i in range(len(options['known']))]))
        net.set_anchors(anchors)
    elif options['learnable_center_loss'] :
        criterion_cent=CenterLoss(num_classes=options['num_classes'], feat_dim=512, use_gpu=use_gpu)
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=options['lr_cent'])
        options['criterion_cent'] = criterion_cent
        options['optimizer_centloss'] = optimizer_centloss
    if options['contr_loss']:
        criterion_contr=SupConLoss(temperature=options['temp_contr'])
        options['criterion_contr'] = criterion_contr


    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['cs'])
    else:
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['cs'])

    if options['eval']:
        net, thre,criterion = load_networks(net, model_path, file_name, criterion=criterion,best_checkpoint=options['best_checkpoint'])
        results = test_.test(net, criterion, testloader, outloader, log=[test_log_feature,test_log_gt,test_log_pred], bad_case=True,**options)
        print("Acc (%): {:.3f}  O_ACC (%): {:.3f}  AUROC (%): {:.3f}  OSCR (%): {:.3f}".format(results['ACC'],results['O_ACC'],results['AUROC'], results['OSCR']))
        print('Labels', end='\t')
        for k in options['known']:
            print(' {}'.format(str(k).zfill(3)), end='    ')
        print(' unknown     ave     [threshold]')
        print('F1(%)', end='   ')
        for f1 in results['F1']:
            print(' {:.3f} '.format(f1), end='')
        print('     {:.3f}     {:.3f}'.format(results['ave_F1'],results['thre']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    if options['dataset'] == 'tiny_imagenet' or options['dataset'] == 'imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))


    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])

    start_time = time.time()

    thres=[]
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train_log_feature_path=os.path.join(options['outf'], 'results', 'epoch{}_train_fea.txt'.format(epoch+1,options['seed']))
        train_log_gt_path = os.path.join(options['outf'], 'results', 'epoch{}_train_gt.txt'.format(epoch + 1, options['seed']))
        train_log_pred_path= os.path.join(options['outf'], 'results', 'epoch{}_train_pred.txt'.format(epoch + 1, options['seed']))

        train_log_feature=open(train_log_feature_path, 'w')
        train_log_gt = open(train_log_gt_path, 'w')
        train_log_pred = open(train_log_pred_path, 'w')

        if options['cs']:

            train_cs(net, netD, netG, criterion, criterionD,optimizer, optimizerD, optimizerG, trainloader,epoch=epoch,writer=writer,**options)

        train(net, criterion, optimizer, trainloader,epoch=epoch,log=[train_log_feature,train_log_gt,train_log_pred],writer=writer, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("Test", options['loss'])
            results = test_.test(net, criterion, testloader, outloader, log=[eval_log_feature_all,eval_log_gt_all,eval_log_pred_all],epoch=epoch, **options)
            print("ACC (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t [BEST: {:.3f}/({} Epoch)".format(results['ACC'], results['AUROC'], results['OSCR'],best_acc,best_epoch))
            writer.add_scalar('Eval/ACC', results['ACC'], epoch+1)
            writer.add_scalar('Eval/AUROC', results['AUROC'], epoch+1)
            writer.add_scalar('Eval/OSCR', results['OSCR'], epoch+1)
            thres.append(results['eval_thre'])
            is_best = False
            acc=results['ACC']
            if acc > best_acc:
                is_best = True
                best_acc = acc
                best_epoch=epoch+1
                copy(train_log_feature_path,train_log_feature_best_path)
                copy(train_log_gt_path, train_log_gt_best_path)
                copy(train_log_pred_path, train_log_pred_best_path)

            os.remove(train_log_feature_path)
            os.remove(train_log_gt_path)
            os.remove(train_log_pred_path)

            ave_thre=sum(thres)/len(thres)
            save_networks(net, ave_thre,model_path, is_best,file_name, criterion=criterion)
        
        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    results = dict()

    from split import get_splits

  #  for i in range(5):
    splits = get_splits(args.dataset, num_split=0)
    known = splits['known_classes']
    unknown =  splits['unknown_classes']


    if options['dataset'] == 'tiny_imagenet':
        img_size = 64
        options['lr'] = 0.001
    elif options['dataset'] == 'imagenet':
        img_size = 224
        options['lr'] = 0.001
    else:
        img_size = 224


    options.update(
        {
           # 'item':     i,
            'known':    known,
            'unknown':  unknown,
            'img_size': img_size
        }
    )

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if options['dataset'] == 'cifar100':
        file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
    else:
        file_name = options['dataset'] + '.csv'

    writer = SummaryWriter(dir_path)

    res = main_worker(options,writer)
    res['unknown'] = unknown
    res['known'] = known
    results[str(0)] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name))
