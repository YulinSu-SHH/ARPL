import os
import os.path as osp
import numpy as np

import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation_


def test(net, criterion, testloader, outloader, thre=None,log=None,bad_case=None,epoch=None, **options):
    net.eval()
    torch.cuda.empty_cache()

    _data_k,_data_u,_x1,_x2,_pred_k, _pred_u, _labels_k,_predictions_k,_predictions_u = [], [], [],[],[],[],[],[],[]

    with torch.no_grad():
        for batch_idx, (data_k, labels_k) in enumerate(testloader):
            if options['use_gpu']:
                data_k, labels_k = data_k.cuda(), labels_k.cuda()
            with torch.set_grad_enabled(False):
                x_k, y_k = net(data_k, True,skip_distance=True)
                logits_k, _ = criterion(x_k, y_k)
                x1,predictions_k= torch.max(F.softmax(logits_k, dim=1), 1)
                #x1,predictions_k = logits_k.data.max(1)

                _data_k.append(data_k)
                _labels_k.append(labels_k.data.cpu().numpy())
                _pred_k.append(logits_k.data.cpu().numpy())
                _x1.append(x1.data.cpu().numpy())
                _predictions_k.append(predictions_k.data.cpu().numpy())

            if log:
                for feature in x_k:
                    for f in feature:
                        log[0].write('{:.6f} '.format(f.item()))
                    log[0].write('\n')
                log[0].flush()

        for batch_idx, (data_u, labels_u) in enumerate(outloader):
            if options['use_gpu']:
                data_u, labels_u = data_u.cuda(), labels_u.cuda()
            
            with torch.set_grad_enabled(False):
                x_u, y_u = net(data_u, True,skip_distance=True)
                logits_u, _ = criterion(x_u, y_u)
                x2,predictions_u=torch.max(F.softmax(logits_u, dim=1), 1)
               # x2,predictions_u = logits_u.data.max(1)
                _data_u.append(data_u)
                _pred_u.append(logits_u.data.cpu().numpy())
                _x2.append(x2.data.cpu().numpy())
                _predictions_u.append(predictions_u.data.cpu().numpy())

            if log:
                for feature in x_u:
                    for f in feature:
                        log[0].write('{:.6f} '.format(f.item()))
                    log[0].write('\n')
                log[0].flush()

    _predictions_k=np.concatenate(_predictions_k,0)
    _predictions_u = np.concatenate(_predictions_u, 0)
    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels_k = np.concatenate(_labels_k, 0)
    _labels_u = np.full_like(np.arange(_pred_u.shape[0], dtype=int), options['num_classes'])
    _x1=np.concatenate(_x1,0)
    _x2=np.concatenate(_x2,0)
    _data_k=torch.cat(_data_k,0)
    _data_u=torch.cat(_data_u,0)

    # Accuracy
    correct = (_predictions_k == _labels_k.data).sum()
    acc = float(correct) * 100. / float(len(_labels_k))

    # Out-of-Distribution detction evaluation
    x1_max, x2_max = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation_.metric_ood(x1_max, x2_max)['Bas']

    # OSCR
    _oscr_socre ,eval_thre= evaluation_.compute_oscr(_pred_k, _pred_u, _labels_k)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['eval_thre'] = eval_thre

    # F1-score
    if options['eval']:
        if thre is not None:
            f1_score ,ave_f1_score,o_acc= evaluation_.compute_f1(_predictions_k, _x1,_predictions_u,_x2, _labels_k, _labels_u,acc,options, thre=thre)
        else:
            f1_score,ave_f1_score,o_acc,thre = evaluation_.compute_f1(_predictions_k, _x1,_predictions_u,_x2, _labels_k, _labels_u,acc, options)



        # Outsample detection
        _predictions_k[_x1 < thre] = options['num_classes']
        _predictions_u[_x2 < thre] = options['num_classes']
        results['F1'] = f1_score * 100.
        results['thre'] = thre
        results['O_ACC'] = o_acc
        results['ave_F1'] = ave_f1_score * 100.

        if bad_case:
            dir_path = os.path.join(options['outf'], 'images')
            img_path1 = os.path.join(dir_path, 'in_cls')
            img_path2 = os.path.join(dir_path, 'out_cls')
            img_path3 = os.path.join(dir_path, 'out_ood')
            if not os.path.exists(img_path1):
                os.makedirs(img_path1)
            if not os.path.exists(img_path2):
                os.makedirs(img_path2)
            if not os.path.exists(img_path3):
                os.makedirs(img_path3)
            f_idx1 ,f_idx2= 0,0
            for c in (_predictions_k == _labels_k.data):
                if c == False:
                    f_data = _data_k[f_idx1]
                    prediction = _predictions_k[f_idx1]
                    label = _labels_k[f_idx1]
                    f_img = transforms.ToPILImage()(f_data.cpu()).convert('RGB')
                    if label==options['num_classes']:
                        img_name = 'idx_' + str(f_idx1) + '_pre_unknown_label_' + str(label) + '_.png'
                        f_img.save(os.path.join(img_path2, img_name))
                    else:
                        img_name = 'idx_' + str(f_idx1) + '_pre_' + str(
                        prediction) + '_label_' + str(label) + '.png'
                        f_img.save(os.path.join(img_path1, img_name))
                f_idx1 = f_idx1 + 1
            for c in (_predictions_u == _labels_u.data):
                if c == False:
                    f_data = _data_u[f_idx2]
                    prediction = _predictions_u[f_idx2]
                    f_img = transforms.ToPILImage()(f_data.cpu()).convert('RGB')
                    img_name = 'idx_' + str(f_idx2) + '_pre_' + str(prediction) + '_label_unkown.png'
                    f_img.save(os.path.join(img_path3, img_name))
                f_idx2 = f_idx2 + 1

    if log:
        for l in _labels_k:
            log[1].write('{}\n'.format(l))
        for l in _labels_u:
            log[1].write('{}\n'.format(l))
        log[1].flush()
        for p in _predictions_k:
            log[2].write('{}\n'.format(p))
        for p in _predictions_u:
            log[2].write('{}\n'.format(p))
        log[2].flush()

    return results
