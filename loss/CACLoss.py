import torch

def CACLoss( distances, gt,**options):
    '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
    true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
    non_gt = torch.Tensor(
        [[i for i in range(len(options['known'])) if gt[x] != i] for x in range(len(distances))]).long().cuda()
    others = torch.gather(distances, 1, non_gt)

    anchor = torch.mean(true)

    tuplet = torch.exp(-others + true.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

    total = options['lbda'] * anchor + tuplet

    return total, anchor, tuplet