import os
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

from .metrics import *


def plot_roc_curve(y_true, y_pred_prob, save_plot = None):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    n_classes = len(set(y_true))
    lw = 2

    y_test = label_binarize(y_true, classes=list(range(n_classes)))
    if type(y_pred_prob) is pd.DataFrame:
        y_pred_prob = y_pred_prob.values

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'violet', 'red', 'gold'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 # label='ROC curve of Class_{0} (area = {1:0.2f})'
                 label='Class-{0}'
                 ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to multi-class')
    plt.legend(loc="lower right")

    if save_plot is not None:
        if os.path.isdir('plt')==False:
            os.makedirs('plt')
        plt.savefig('plt/%s.pdf'%save_plot, bbox_inches='tight')
    plt.show()

def cm_analysis(y_true, y_pred, labels=None, ymap=None, save_plot=None, figsize=(5,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    labels = list(set(y_true)) if labels is None else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/\n%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True label'
    cm.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues", annot_kws={"size": 8} )
    if save_plot is not None:
        if os.path.isdir('plt')==False:
            os.makedirs('plt')
        plt.savefig('plt/%s.pdf'%save_plot, bbox_inches='tight')
    plt.show()

def precision_recall_curve_(y_true, y_pred, y_pred_prob, pos_class=None, save_plot=None):
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    # https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier

    n_classes = len(set(y_true))

    y_test = label_binarize(y_true, classes=list(range(n_classes)))
    if type(y_pred_prob) is pd.DataFrame:
        y_pred_prob = y_pred_prob.values

    precision = [0]*n_classes
    recall = [0]*n_classes
    loop = range(n_classes)[pos_class:pos_class+1] if pos_class is not None else range(n_classes)
    for i in loop:
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_pred_prob[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='Class_{}'.format(i+1))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")

    if save_plot is not None:
        if os.path.isdir('plt') == False:
            os.makedirs('plt')
        plt.savefig('plt/%s.pdf' % save_plot, bbox_inches='tight')
    plt.show()