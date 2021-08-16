import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils_ import AverageMeter
from loss.CACLoss import CACLoss
from loss.center_loss import CenterLoss

def train(net, criterion, optimizer,trainloader, writer,epoch=None,log=None, **options):
    net.train()
    losses,total_losses= AverageMeter(),AverageMeter()

    if options['anchored_center_loss']:
        lossesCAC, lossesA, lossesT = AverageMeter(), AverageMeter(), AverageMeter()
    if options['learnable_center_loss']:
        lossesCEN = AverageMeter()
    if options['contr_loss']:
        lossesCON=AverageMeter()

    torch.cuda.empty_cache()
    
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):

        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            if options['learnable_center_loss']:
                criterion_cent=options['criterion_cent']
                optimizer_centloss=options['optimizer_centloss']
                optimizer_centloss.zero_grad()


            if options['anchored_center_loss']:
                x, y, dist = net(data, True)
            else:
                x, y= net(data,True,skip_distance=True)

            logits, ARPloss = criterion(x, y, labels)
            if log is not None:
                for feature in x:
                    for f in feature:
                        log[0].write('{:.6f} '.format(f.item()))
                    log[0].write('\n')
                log[0].flush()
                for l in labels:
                    log[1].write('{}\n'.format(l))
                log[1].flush()
                for p in logits.data.max(1)[1]:
                    log[2].write('{}\n'.format(p))
                log[2].flush()

            if options['anchored_center_loss']:
                cacLoss, anchorLoss, tupletLoss = CACLoss(dist, labels, **options)
                loss = ARPloss+ cacLoss * options['weight_cent']
            elif options['learnable_center_loss']:
                loss_cent = criterion_cent(x, labels)
                loss = ARPloss + loss_cent * options['weight_cent']
            else:
                loss = ARPloss
            if options['contr_loss']:
                indices = np.random.permutation(data.size(0))
                x2 = x[indices]
                features = torch.cat([x.unsqueeze(1), x2.unsqueeze(1)], dim=1)
                loss_contr = options['criterion_contr'](features, labels)
                loss = loss + loss_contr * options['weight_contr']
            loss.backward()
            optimizer.step()
            if options['learnable_center_loss']:
                # by doing so, weight_cent would not impact on the learning of centers
                for param in criterion_cent.parameters():
                    param.grad.data *= (1. / options['weight_cent'])
                optimizer_centloss.step()

        if options['anchored_center_loss']:
            lossesCAC.update(cacLoss.item(), labels.size(0))
            lossesA.update(anchorLoss.item(), labels.size(0))
            lossesT.update(tupletLoss.item(), labels.size(0))
        if options['learnable_center_loss']:
            lossesCEN.update(loss_cent.item(), labels.size(0))
        if options['contr_loss']:
            lossesCON.update(loss_contr.item(), labels.size(0))

        losses.update(ARPloss.item(),labels.size(0))
        total_losses.update(loss.item(),labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            niter = epoch * len(trainloader) + batch_idx
            if options['anchored_center_loss']:
                writer.add_scalar('Train/cacLoss', lossesCAC.val, niter)
                writer.add_scalar('Train/anchorLoss', lossesA.val, niter)
                writer.add_scalar('Train/tupletLoss', lossesT.val, niter)
            if options['learnable_center_loss']:
                writer.add_scalar('Train/centLoss', lossesCEN.val, niter)
            if options['contr_loss']:
                writer.add_scalar('Train/contrLoss', lossesCON.val, niter)
            writer.add_scalar('Train/oriLoss', losses.val, niter)
            writer.add_scalar('Train/Losses', total_losses.val, niter)
        
        loss_all += total_losses.avg

    return loss_all

def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG,
        trainloader,writer,epoch=None, log=None,**options):
    print('train with confusing samples')
    losses,lossesG,lossesD,total_losses =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    if options['anchored_center_loss']:
        lossesCAC, lossesA, lossesT = AverageMeter(), AverageMeter(), AverageMeter()
    if options['learnable_center_loss']:
        lossesCEN = AverageMeter()
    if options['contr_loss']:
        lossesCON = AverageMeter()
    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):

        gan_target = torch.FloatTensor(labels.size()).fill_(0)

        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()
        
        data, labels = Variable(data), Variable(labels)
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda(),skip_distance=True)
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()

        if options['learnable_center_loss']:
            criterion_cent=options['criterion_cent']
            optimizer_centloss=options['optimizer_centloss']
            optimizer_centloss.zero_grad()

        # if options['contr_loss']:
        #     bsz = labels.shape[0]
        #     data = torch.cat([data[0], data[1]], dim=0)

        if options['anchored_center_loss']:
            x, y, dist = net(data, True)
        else:
            x, y = net(data, True, skip_distance=True)
            
        logits, ARPloss = criterion(x, y, labels)
        if log is not None:
            for feature in x:
                for f in feature:
                    log[0].write('{:.6f} '.format(f.item()))
                log[0].write('\n')
            log[0].flush()
            for l in labels:
                log[1].write('{}\n'.format(l))
            log[1].flush()
            for p in logits.data.max(1)[1]:
                log[2].write('{}\n'.format(p))
            log[2].flush()

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda(),skip_distance=True)
        F_loss_fake = criterion.fake_loss(x).mean()
        ARPloss += options['beta'] * F_loss_fake

        if options['anchored_center_loss']:
            cacLoss, anchorLoss, tupletLoss = CACLoss(dist, labels, **options)
            loss = ARPloss + cacLoss * options['weight_cent']

        elif options['learnable_center_loss']:
            loss_cent = criterion_cent(x, labels)
            loss = ARPloss + loss_cent * options['weight_cent']
        else:
            loss = ARPloss
        if options['contr_loss']:
            indices = np.random.permutation(data.size(0))
            x2 = x[indices]
            features = torch.cat([x.unsqueeze(1), x2.unsqueeze(1)], dim=1)
            loss_contr = options['criterion_contr'](features, labels)
            loss = loss + loss_contr * options['weight_contr']

        loss.backward()
        optimizer.step()

        if options['learnable_center_loss']:
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / options['weight_cent'])
            optimizer_centloss.step()

        if options['anchored_center_loss']:
            lossesCAC.update(cacLoss.item(), labels.size(0))
            lossesA.update(anchorLoss.item(), labels.size(0))
            lossesT.update(tupletLoss.item(), labels.size(0))
        if options['learnable_center_loss']:
            lossesCEN.update(loss_cent.item(), labels.size(0))
        if options['contr_loss']:
            lossesCON.update(loss_contr.item(), labels.size(0))

        losses.update(ARPloss.item(),labels.size(0))
        total_losses.update(loss.item(),labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
            .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
            niter = epoch * len(trainloader) + batch_idx
            if options['anchored_center_loss']:
                writer.add_scalar('Train_cs/cacLoss', lossesCAC.val, niter)
                writer.add_scalar('Train_cs/anchorLoss', lossesA.val, niter)
                writer.add_scalar('Train_cs/tupletLoss', lossesT.val, niter)
            if options['learnable_center_loss']:
                writer.add_scalar('Train_cs/centLoss', lossesCEN.val, niter)
            if options['contr_loss']:
                writer.add_scalar('Train/contrLoss', lossesCON.val, niter)
            writer.add_scalar('Train_cs/GLoss', lossesG.val, niter)
            writer.add_scalar('Train_cs/DLoss', lossesD.val, niter)
            writer.add_scalar('Train_cs/oriLoss', losses.val, niter)
            writer.add_scalar('Train_cs/Losses', total_losses.val, niter)

        loss_all += total_losses.avg

    return loss_all
