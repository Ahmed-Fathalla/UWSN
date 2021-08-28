from sklearn.model_selection import StratifiedKFold
import copy
import gc

from .metrics import get_results, metrics_
from .visualizations import *
from ..utils import *

def kfold_validation(model, X, y, n_splits=10, round_=4, metric_lst=metrics_, save_plot=None, get_folds_results=True, file_=True):
    y = y - 1
    clf = model

    if hasattr(model,'__class__'):
        clf_name = model.__class__.__name__.replace('Classifier','')

    result_df = pd.DataFrame(y.values, columns=['y_true'])
    result_df['prob_1'] = -1
    result_df['prob_2'] = -1
    result_df['prob_3'] = -1
    result_df['prob_4'] = -1
    result_df['prob_5'] = -1
    result_df['prob_6'] = -1

    y_pred_col = 'y_pred'
    result_df[y_pred_col] = -1
    result_df['fold'] = -1

    train_test_results = []
    skf = StratifiedKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y),1):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        m = copy.deepcopy(clf)
        m.fit(X_train, y_train)

        train_pred = m.predict(X_train)
        test_pred  = m.predict(X_test)

        prob = []
        if hasattr(m, 'predict_proba'):
            pred_prob_train = m.predict_proba(X_train)
            pred_prob_test  = m.predict_proba(X_test)
        else:
            assert False, 'foxx, no predict_proba'

        result_df.loc[test_index, [y_pred_col]] = test_pred.reshape(-1, 1)
        result_df.loc[test_index, ['prob_1']  ] = pred_prob_test[:, 0].reshape(-1, 1)
        result_df.loc[test_index, ['prob_2']  ] = pred_prob_test[:, 1].reshape(-1, 1)
        result_df.loc[test_index, ['prob_3']  ] = pred_prob_test[:, 2].reshape(-1, 1)
        result_df.loc[test_index, ['prob_4']  ] = pred_prob_test[:, 3].reshape(-1, 1)
        result_df.loc[test_index, ['prob_5']  ] = pred_prob_test[:, 4].reshape(-1, 1)
        result_df.loc[test_index, ['prob_6']  ] = pred_prob_test[:, 5].reshape(-1, 1)

        result_df.loc[test_index, ['fold']] = fold
        fold_train_score = get_results( y_true=y_train,
                                        y_pred=train_pred,
                                        y_pred_prob=pred_prob_train,
                                        metric_lst=metric_lst)
        fold_test_score = get_results(  y_true=y_test,
                                        y_pred=test_pred,
                                        y_pred_prob=pred_prob_test,
                                        metric_lst=metric_lst)

        train_test_results +=[fold_train_score[0],fold_test_score[0]]

        del m
        gc.collect()

    if get_folds_results:
        lst = []
        for i in range(1, 11):
            lst += ['Fold_%d train' % i, 'Fold_%d test' % i]
        df_to_file(
            train_test_results,
            fold_col=lst,
            cols=[m.__name__ for m in metric_lst],
            round_=round_,
            print_=False,
            file = 'plt/%s %s %d-KFold.txt'%(save_plot, clf_name, n_splits)
        )

    if True:
        df_to_file(
            get_results(
                y_true=result_df['y_true'],
                y_pred=result_df['y_pred'],
                y_pred_prob=result_df[['prob_1','prob_2','prob_3','prob_4','prob_5','prob_6']],
                metric_lst=metric_lst
            ),
            fold_col=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6'],
            cols=[m.__name__ for m in metric_lst],
            round_=round_,
            file='plt/%s %s %d-KFold.txt' % (save_plot, clf_name, n_splits),
            print_=False,
            pre = '\n'*3
        )

    if save_plot is not None:
        plot_roc_curve(y_true=result_df['y_true'].values,
                       y_pred_prob = result_df[['prob_1','prob_2','prob_3','prob_4','prob_5','prob_6']],
                       save_plot='%s %s %d-KFold_Roc'%(save_plot, clf_name, n_splits)
                       )

        cm_analysis(y_true=result_df['y_true'],
                    y_pred=result_df['y_pred'],
                    save_plot='%s %s %d-KFold_Confusion'%(save_plot, clf_name, n_splits)
                    )

        pos_class = result_df['y_true'].value_counts()[-1:].index[0]
        # print('pos_class = ', pos_class)
        precision_recall_curve_(y_true=result_df['y_true'],
                                y_pred=result_df['y_pred'],
                                y_pred_prob=result_df[['prob_1', 'prob_2', 'prob_3', 'prob_4', 'prob_5', 'prob_6']].values,
                                pos_class = pos_class,
                                save_plot= '%s %s %d-KFold_PR_curve'%(save_plot, clf_name, n_splits)
                                )

    return result_df #, result_df['y_true'], result_df['y_pred'], prob

from ..utils import get_index_from_pkl



def hold_out_validation(model, X, y, train_test_pkl, round_=4, metric_lst=metrics_, save_plot=None):
    y = y - 1
    clf = model

    if hasattr(model,'__class__'):
        clf_name = model.__class__.__name__.replace('Classifier','')

    train_index, test_index = get_index_from_pkl(train_test_pkl)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    prob = []
    if hasattr(clf, 'predict_proba'):
        pred_prob_train = clf.predict_proba(X_train)
        pred_prob_test  = clf.predict_proba(X_test)
    else:
        assert False, 'foxx, no predict_proba'

    train_score = get_results( y_true=y_train,
                               y_pred=train_pred,
                               y_pred_prob=pred_prob_train,
                               metric_lst=metric_lst)
    test_score = get_results( y_true=y_test,
                              y_pred=test_pred,
                              y_pred_prob=pred_prob_test,
                              metric_lst=metric_lst)

    df_to_file( train_score,
                fold_col=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6'],
                cols=[m.__name__ for m in metric_lst],
                round_=round_,
                print_=False,
                file='plt/%s %s Hold_out_.txt' % (save_plot, clf_name),
                pre='\n' * 3 + 'Train:' + '\n'
                )

    df_to_file( test_score,
                fold_col=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6'],
                cols=[m.__name__ for m in metric_lst],
                round_=round_,
                print_=False,
                file='plt/%s %s Hold_out_.txt' % (save_plot, clf_name),
                pre = '\n'*3+'test:'+'\n'
                )

    if save_plot is not None:
        plot_roc_curve(y_true=y_test.values,
                       y_pred_prob = pred_prob_test,
                       save_plot='%s %s Hold_out_Roc'%(save_plot, clf_name)
                       )

        cm_analysis(y_true=y_test,
                    y_pred=test_pred,
                    save_plot='%s %s Hold_out_Confusion'%(save_plot, clf_name)
                    )

        pos_class = y_test.value_counts()[-1:].index[0]
        precision_recall_curve_(y_true=y_test,
                                y_pred=test_pred,
                                y_pred_prob=pred_prob_test,
                                pos_class = pos_class,
                                save_plot= '%s %s Hold_out_PR_curve'%(save_plot, clf_name)
                                )








