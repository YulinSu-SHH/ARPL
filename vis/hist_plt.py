import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
matplotlib.use('Agg')
def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def cal_var(path_fea,path_gt,log,num_classes=11):
    
    feature=np.loadtxt(path_fea)
    gt = np.loadtxt(path_gt)
    feature_ave = dict()
    labelnum=num_classes
    count = np.zeros(labelnum)
    feature_var = np.zeros(labelnum)
    for i in range(feature.shape[0]):
        count[gt[i].astype(np.int32)] += 1
        if count[gt[i].astype(np.int32)] == 1:
            feature_ave[gt[i].astype(np.int32)] = feature[i]
        else:
            feature_ave[gt[i].astype(np.int32)] += feature[i]

    for i in range(labelnum):
        feature_ave[i] /= count[i]

    for i in range(feature.shape[0]):
        feature_var[gt[i].astype(np.int32)] += ((feature[i] - feature_ave[gt[i].astype(np.int32)]) ** 2).sum()

    for i in range(labelnum):
        feature_var[i] /= (count[i] - 1.)
    print_log(feature_var,log)

    return feature_var

def var_hist_plt(out_path='./var_hist'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    log = open(os.path.join(out_path, 'log.txt'), 'w')
    out_path_train=os.path.join(out_path,'var_hist_train.png')
    out_path_eval = os.path.join(out_path, 'var_hist_eval.png')
    label =['2', '13', '44', '64', '76', '111', '128', '136', '167', '187','unknown']

    train_ori=cal_var(path_fea='C:/Users/29624/Desktop/log_ori/results/train_fea_best.txt',
                      path_gt='C:/Users/29624/Desktop/log_ori/results/train_gt_best.txt',
                      num_classes=10,log=log)
    train_cenloss=cal_var(path_fea='C:/Users/29624/Desktop/log_cenloss/results/train_fea_best.txt',
                          path_gt='C:/Users/29624/Desktop/log_cenloss/results/train_gt_best.txt',
                          num_classes=10,log=log)
    train_anchorcenloss = cal_var(path_fea='C:/Users/29624/Desktop/log_anchorcenloss/results/train_fea_best.txt',
                                  path_gt='C:/Users/29624/Desktop/log_anchorcenloss/results/train_gt_best.txt',
                                  num_classes=10,log=log)
    plt.figure(figsize=(20, 10))
    x = np.arange(len(label[:-1]))
    width = 0.15
    plt.bar(x - 1.5*width, train_ori, width, label='train_ori', color='lightcoral')
    plt.bar(x , train_cenloss, width, label='train_cenloss', color='peru')
    plt.bar(x + 1.5*width, train_anchorcenloss, width, label='train_anchorcenloss', color='burlywood')
    plt.xlabel('Classes', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    #plt.title('mean of each class')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=label[:-1], fontsize=13)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=15)
    plt.savefig(out_path_train, bbox_inches='tight')

    val_ori = cal_var(path_fea='../log_ori/results/test_fea_all.txt',
                      path_gt='../log_ori/results/test_gt_all.txt',
                      num_classes=11,log=log)
    val_cenloss = cal_var(path_fea='../log_cenloss/results/test_fea_all.txt',
                          path_gt='../log_cenloss/results/test_gt_all.txt',
                          num_classes=11,log=log)
    val_anchorcenloss = cal_var(path_fea='../log_anchorcenloss/results/test_fea_all.txt',
                                path_gt='../log_anchorcenloss/results/test_gt_all.txt',
                                num_classes=11,log=log)
    plt.figure(figsize=(20, 10))
    x = np.arange(len(label))
    width = 0.15
    plt.bar(x - 1.5*width, val_ori, width, label='eval_ori', color='lawngreen')
    plt.bar(x , val_cenloss, width, label='eval_cenloss', color='c')
    plt.bar(x + 1.5*width, val_anchorcenloss, width, label='eval_anchorcenloss', color='cornflowerblue')
    plt.xlabel('Classes', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    # plt.title('mean of each class')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=label, fontsize=13)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=15)
    plt.savefig(out_path_eval, bbox_inches='tight')
    

if __name__=='__main__':
    var_hist_plt()
