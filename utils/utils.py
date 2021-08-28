import sys
import numpy as np
import pandas as pd
import warnings;warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import operator
from tabulate import tabulate
from .pkl_utils import load_pkl, dump

def get_data(df, exp='Exp_1'):
    x,y = None, None
    if exp=='Exp_2':
        x = df[['UW_speed', 'UW_direction', 'Sig_1','Sig_2', 'Sig_3', 'Sig_4', 'Sig_5', 'Sig_6','y']].copy()
        y = x.pop('y')
    elif exp=='Exp_1':
        x = df[['UW_speed', 'UW_direction', 'sensor_x_orgin',	'sensor_y_orgin' ,'y']].copy()
        y = x.pop('y')
    return x,y

from sklearn.model_selection import train_test_split

def set_train_test_index_pkl(series, test_size=0.2 , random_state=42):
    X_train, X_test = train_test_split(list(range(len(series))), test_size=test_size, random_state=random_state, stratify=series)
    train_test_pkl={
        'train_index':X_train ,
        'test_index':X_test ,
    }
    dump('train_test_pkl.pkl',train_test_pkl)

def get_index_from_pkl(train_test_pkl):
    t = load_pkl(train_test_pkl)
    train_index = t['train_index']
    test_index = t['test_index']
    return train_index, test_index

def sort_tuble(tub, item = 2, ascending = True):
    tub = sorted(tub, key=operator.itemgetter(item), reverse=False)
    if ascending:
        return tub[0]
    else:
        return tub[-1]

def df_to_file(df, cols=None, fold_col=None, round_=5, file=None, padding='left', rep_newlines='\t', print_=True, wide_col='', pre='', post=''):
    if type(df) is list:
        df = pd.DataFrame(np.array(df), columns=cols)
    elif type(df) is pd.DataFrame:
        ...

    headers = [wide_col+str(i)+wide_col for i in df.columns.values]

    df = df.round(round_)
    df['fold'] = fold_col
    df = df[['fold']+headers]

    c = rep_newlines + tabulate(df.values,
                                headers=headers,
                                stralign=padding,
                                disable_numparse=1,
                                tablefmt = 'grid' # 'fancy_grid' ,
                                ).replace('\n', '\n'+rep_newlines)
    if print_:print(c)
    if file is not None:
        with open(file, 'a', encoding="utf-8") as myfile:
            myfile.write( pre + c + post + '\n')