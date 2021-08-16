import torch
import os
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def plot_confusion_matrix(cm, classes, log,normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print_log("Normalized confusion matrix",log)
    else:
        print_log('Confusion matrix, without normalization',log)
    print_log(title,log)
    print_log(cm,log)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def cm_plt(out_path='./confusion_matrixs'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    labels =['2', '13', '44', '64', '76', '111', '128', '136', '167', '187','unknown']
    log = open(os.path.join(out_path, 'log.txt'), 'w')

    train_ori_gt=np.loadtxt('../log_ori/results/train_gt_best.txt')
    train_ori_pred=np.loadtxt('../log_ori/results/train_pred_best.txt')
    train_cenloss_gt = np.loadtxt('../log_cenloss/results/train_gt_best.txt')
    train_cenloss_pred = np.loadtxt('../log_cenloss/results/train_pred_best.txt')
    train_anchorcenloss_gt = np.loadtxt('../log_anchorcenloss/results/train_gt_best.txt')
    train_anchorcenloss_pred = np.loadtxt('../log_anchorcenloss/results/train_pred_best.txt')

    eval_ori_gt = np.loadtxt('../log_ori/results/test_gt_all.txt')
    eval_ori_pred = np.loadtxt('../log_ori/results/test_pred_all.txt')
    eval_cenloss_gt = np.loadtxt('../log_cenloss/results/test_gt_all.txt')
    eval_cenloss_pred = np.loadtxt('../log_cenloss/results/test_pred_all.txt')
    eval_anchorcenloss_gt = np.loadtxt('../log_anchorcenloss/results/test_gt_all.txt')
    eval_anchorcenloss_pred = np.loadtxt('../log_anchorcenloss/results/test_pred_all.txt')

    train_ori_cm = confusion_matrix(train_ori_gt, train_ori_pred)
    train_cenloss_cm = confusion_matrix(train_cenloss_gt, train_cenloss_pred)
    train_anchorcenloss_cm = confusion_matrix(train_anchorcenloss_gt, train_anchorcenloss_pred)

    eval_ori_cm = confusion_matrix(eval_ori_gt, eval_ori_pred)
    eval_cenloss_cm = confusion_matrix(eval_cenloss_gt, eval_cenloss_pred)
    eval_anchorcenloss_cm = confusion_matrix(eval_anchorcenloss_gt, eval_anchorcenloss_pred)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(train_ori_cm, labels,title='Train_ori Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path,'train_ori_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(train_cenloss_cm, labels,title='Train_cenloss Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path, 'train_cenloss_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(train_anchorcenloss_cm, labels,title='Train_anchorcenloss Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path, 'train_anchorcenloss_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(eval_ori_cm, labels,title='Eval_ori Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path, 'eval_ori_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(eval_cenloss_cm, labels,title='Eval_cenloss Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path, 'eval_cenloss_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(eval_anchorcenloss_cm, labels,title='Eval_anchorcenloss Confusion matrix',log=log)
    plt.savefig(os.path.join(out_path, 'eval_anchorcenloss_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    cm_plt()
