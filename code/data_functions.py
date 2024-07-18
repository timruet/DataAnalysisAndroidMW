import numpy as np
import ujson as json
from datetime import datetime, timedelta
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from scipy.sparse import csr_matrix
from dateutil.relativedelta import relativedelta



def try_parsing_date(text):
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')



def load_features(X_filename, y_filename, meta_filename):
    print('Loading features...')
    with open(X_filename, 'rt') as f:
        X = json.load(f)

    with open(y_filename, 'rt') as f:
        y = json.load(f)

    with open(meta_filename, 'rt') as f:
        t = json.load(f)
        
    #t = [o['dex_date'] for o in t]
    #t = [try_parsing_date(o) for o in t]
    #t = [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') for o in t]
    t = [datetime.strptime(o['dex_date'].replace('T', ' '), '%Y-%m-%d %H:%M:%S') for o in t]

    return X, y, t 



def vectorize(X, y, t):
    """Transform input data into appropriate forms for an sklearn classifier.
​
    Args:
        X (list): A list of dictionaries of input features for each sample.
        y (list): A list of ground truths for the data.
        t (list): A list of datetimes for the data.
​
    """
    print('Vectorizing features...')
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    y = np.asarray(y)
    t = np.asarray(t)
    return X, y, t, vec




def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


#split data into train and test according to a given date
def split(X,y,t,s):

    list_idx_train = []
    list_idx_test = []

    for idx, x in enumerate(t):
        if x < s:
            list_idx_test.append(idx)
        else:
            list_idx_train.append(idx)



    X_train = X
    y_train = y
    t_train = t
    X_test = X
    y_test = y
    t_test = t

    
    X_train = delete_rows_csr(X_train, list_idx_train)
    X_test = delete_rows_csr(X_test, list_idx_test)


    y_train = np.delete(y_train,list_idx_train,0)
    t_train = np.delete(t_train,list_idx_train,0)

    y_test = np.delete(y_test,list_idx_test,0)
    t_test = np.delete(t_test,list_idx_test,0)

    
    return X_train, y_train, t_train, X_test, y_test, t_test


def splitIntervals(X,y,t,m):
    split_date = datetime.strptime('2015-1-1 0:0:0', '%Y-%m-%d %H:%M:%S') + relativedelta(months=+m)


    intervalData = []
    while split_date <= datetime.strptime('2019-1-1 0:0:0', '%Y-%m-%d %H:%M:%S'):
        X_1, y_1, t_1, X_2, y_2, t_2 = split(X,y,t,split_date)
        intervalData.append((X_1,y_1,t_1))
        split_date = split_date + relativedelta(months=+m)
        X = X_2
        y = y_2
        t = t_2
    
    return intervalData

def setMWRate(X,y,t):
    n = 0
    idxListMW = []
    idxListGW = []
    for idx, item in enumerate(y):
        if item == 1:
            n = n+1
            idxListMW.append(idx)
        else:
            idxListGW.append(idx)
    MWRate = n/len(y)
    GWRate = (len(y)-n)/len(y)

    if MWRate > 0.1:
        while MWRate > 0.1:
            X = delete_rows_csr(X, [idxListMW[0]])
            y = np.delete(y,[idxListMW[0]],0)
            t = np.delete(t,[idxListMW[0]],0)
            idxListMW = np.delete(idxListMW,0,0)
            n = 0
            idxListMW = []
            idxListGW = []
            for idx, item in enumerate(y):
                if item == 1:
                    n = n+1
                    idxListMW.append(idx)
                else:
                    idxListGW.append(idx)
                    MWRate = n/len(y)
                    GWRate = (len(y)-n)/len(y)
    else:
        while MWRate < 0.1:
            X = delete_rows_csr(X, [idxListGW[0]])
            y = np.delete(y,[idxListGW[0]],0)
            t = np.delete(t,[idxListGW[0]],0)
            idxListMW = np.delete(idxListGW,0,0)
            n = 0
            idxListMW = []
            idxListGW = []
            for idx, item in enumerate(y):
                if item == 1:
                    n = n+1
                    idxListMW.append(idx)
                else:
                    idxListGW.append(idx)
            MWRate = n/len(y)
            GWRate = (len(y)-n)/len(y)
    return X,y,t
        




#deprecated
def split_old(X,y,t,s):
    print('Splitting data...')
    split_date = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')

    X_train = []
    y_train = []
    t_train = []
    X_test = []
    y_test = []
    t_test = []


    for idx, x in enumerate(t):
        if x < split_date:
            X_train.append(X[idx])
            y_train.append(y[idx])
            t_train.append(t[idx])
        else:
            X_test.append(X[idx])
            y_test.append(y[idx])
            t_test.append(t[idx])
    
    return X_train, y_train, t_train, X_test, y_test, t_test
