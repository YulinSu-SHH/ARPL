import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')



def cal_tsne(feature_path,gt_path,pred_path,mode,num_classes=11):
    color = ['brown', 'gold', 'lawngreen', 'grey', 'c', 'b', 'crimson', 'darkgray', 'aquamarine', 'coral', 'k']
    label = ['2', '13', '44', '64', '76', '111', '128', '136', '167', '187', 'unknown']
    feature = np.loadtxt(feature_path)
    gt = np.loadtxt(gt_path)
    pred=np.loadtxt(pred_path)
    en_fea_embed = TSNE(n_components=2).fit_transform(feature)
    en_fea_embed_savepath='./fea_embed'
    if not os.path.exists(en_fea_embed_savepath):
        os.makedirs(en_fea_embed_savepath)
    log= open(os.path.join(en_fea_embed_savepath,'{}_fea_embed.txt'.format(mode)), 'a+')
    for feature in en_fea_embed:
        for f in feature:
            log.write('{:.6f} '.format(f.item()))
        log.write('\n')
    log.flush()

    count = np.zeros(num_classes)
    plt.figure(figsize=(30, 30))

    for i in range(en_fea_embed.shape[0] ):
        count[gt[i].astype(np.int32)] +=1
        if count[gt[i].astype(np.int32)] == 1:
            plt.scatter(en_fea_embed[i, 0], en_fea_embed[i, 1], c = color[gt[i].astype(np.int32)], marker = 'o',linewidths=0.5 ,label = label[gt[i].astype(np.int32)])
        else:
            plt.scatter(en_fea_embed[i, 0], en_fea_embed[i, 1], c = color[gt[i].astype(np.int32)], marker = 'o',linewidths=0.5)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.legend(loc = 0, prop = {'size':30},labels=label)
    plt.savefig(os.path.join(en_fea_embed_savepath,'{}_fea_gt.png'.format(mode)), bbox_inches='tight')

    plt.close()

    count = np.zeros(num_classes)
    plt.figure(figsize=(30, 30))

    for i in range(en_fea_embed.shape[0]):
        count[pred[i].astype(np.int32)] += 1
        if count[pred[i].astype(np.int32)] == 1:
            plt.scatter(en_fea_embed[i, 0], en_fea_embed[i, 1], c=color[pred[i].astype(np.int32)], marker='o',linewidths=0.5, label=label[pred[i].astype(np.int32)])
        else:
            plt.scatter(en_fea_embed[i, 0], en_fea_embed[i, 1], c=color[pred[i].astype(np.int32)], marker='o',linewidths=0.5)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.legend(loc=0, prop={'size': 30},labels=label)
    plt.savefig(os.path.join(en_fea_embed_savepath,'{}_fea_pred.png'.format(mode)), bbox_inches='tight')

    plt.close()

def tsne():
    # train_ori
    cal_tsne(feature_path='../log_ori/results/train_fea_best.txt',
             gt_path='../log_ori/results/train_gt_best.txt',
             pred_path='../log_ori/results/train_pred_best.txt',
             mode='train_ori',num_classes=10)

    # val_ori
    cal_tsne(feature_path='../log_ori/results/test_fea_all.txt',
             gt_path='../log_ori/results/test_gt_all.txt',
             pred_path='../log_ori/results/test_pred_all.txt',
             mode='eval_ori',num_classes=11)

    # train_cenloss
    cal_tsne(feature_path='../log_cenloss/results/train_fea_best.txt',
             gt_path='../log_cenloss/results/train_gt_best.txt',
             pred_path='../log_cenloss/results/train_pred_best.txt',
             mode='train_cenloss',num_classes=10)

    # val_cenloss
    cal_tsne(feature_path='../log_cenloss/results/test_fea_all.txt',
             gt_path='../log_cenloss/results/test_gt_all.txt',
             pred_path='../log_cenloss/results/test_pred_all.txt',
             mode='eval_cenloss',num_classes=11)

    # train_anchorcenloss
    cal_tsne(feature_path='../log_anchorcenloss/results/train_fea_best.txt',
             gt_path='../log_anchorcenloss/results/train_gt_best.txt',
             pred_path='../log_anchorcenloss/results/train_pred_best.txt',
             mode='train_anchorcenloss',num_classes=10)

    # val_anchorcenloss
    cal_tsne(feature_path='../log_anchorcenloss/results/test_fea_all.txt',
             gt_path='../log_anchorcenloss/results/test_gt_all.txt',
             pred_path='../log_anchorcenloss/results/test_pred_all.txt',
             mode='eval_anchorcenloss',num_classes=11)


if __name__=='__main__':
    tsne()