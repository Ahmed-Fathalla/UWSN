import numpy as np
import pandas as pd


from utils.ML_utils.Validation import kfold_validation, hold_out_validation
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utils.utils import set_train_test_index_pkl, get_data

set_train_test_index_pkl(df['y'], test_size=0.2, random_state=42)

df = pd.read_csv('Simulation_exp/Log Sun_2021-05-09 00_50_56AM_ML_df.csv')

for exp_type in ['Exp_1', 'Exp_2']:
    x, y = get_data(df, exp_type)
    for model,par in [
                (KNeighborsClassifier(), {"n_neighbors":2}), # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                (GaussianNB(), {}),  # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
                (DecisionTreeClassifier(), {}), # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
                (CatBoostClassifier(), {'verbose':0, 'n_estimators':100,'loss_function':'MultiClass'})
               ]:
        print( 'model = ' , model, par )
        clf = model
        if len(par.keys())>0:
            clf.set_params(**par)
        hold_out_validation(clf, x, y, 'train_test_pkl.pkl', round_=4, save_plot=exp_type)
        kfold_validation(clf, x, y, n_splits=10, round_=4, save_plot=exp_type)