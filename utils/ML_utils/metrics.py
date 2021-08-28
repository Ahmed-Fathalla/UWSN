from sklearn.metrics import recall_score,precision_score,f1_score,\
                            accuracy_score, roc_auc_score, \
                            classification_report, confusion_matrix
# import numpy as np
# from sklearn.metrics import roc_curve, auc

metrics_ = [recall_score, precision_score, f1_score, accuracy_score, roc_auc_score]

def get_results(y_true, y_pred, y_pred_prob, metric_lst=metrics_):
    res_1,res_2,res_3,res_4,res_5,res_6,out = [], [],[], [],[], [], []
    labels=sorted(list(range(y_pred_prob.shape[1])))
    for m in metric_lst[:3]:
        res_1.append(m(y_true, y_pred, labels=labels, pos_label=labels[0], average='micro'))
        res_2.append(m(y_true, y_pred, labels=labels, pos_label=labels[1], average='micro'))
        res_3.append(m(y_true, y_pred, labels=labels, pos_label=labels[2], average='micro'))
        res_4.append(m(y_true, y_pred, labels=labels, pos_label=labels[3], average='micro'))
        res_5.append(m(y_true, y_pred, labels=labels, pos_label=labels[4], average='micro'))
        res_6.append(m(y_true, y_pred, labels=labels, pos_label=labels[5], average='micro'))
    o = [accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred_prob, multi_class="ovr")]
    out.append(res_1 + o)
    out.append(res_2 + o)
    out.append(res_3 + o)
    out.append(res_4 + o)
    out.append(res_5 + o)
    out.append(res_6 + o)
    return out