import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg

    return loss_all

