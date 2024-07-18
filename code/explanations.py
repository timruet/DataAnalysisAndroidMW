import os
from pathlib import Path
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from string import ascii_uppercase
from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY
import numpy as np
from collections import OrderedDict

from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import json
import datetime

from data_functions import vectorize

def get_family_name(avclass2_string):
    
    if avclass2_string == '[]':
        return '[]'

    family = 'generic'
    labels = avclass2_string.split(',')
    for label in labels:
        if label.startswith('FAM'):
            family = label.split('FAM:')[1].split('|')[0]
    return family

def fix_date(date):
    return str(date).replace('t', ' ')

def get_family_data_index(s, df):
    index_list = []
    for idx, family in enumerate(df['families']):
        if (df.iloc[idx].at["label"] == 1) & (family == s):
            index_list.append(idx)
    return index_list

def get_average_sample_weights(X, coef_ , vec, index_list):
    feature_names_ = vec.feature_names_
    data = X[index_list,:]
    num = data.shape[0]
    dim = data.shape[1]

    data_coef_ = data.multiply(coef_)
    data_sum = data_coef_.sum(0)
    multiplier = np.array([[1/num for i in range(dim)]])
    data = np.multiply(data_sum, multiplier)

    dict = OrderedDict()
    for idx, name in enumerate(feature_names_):
        dict[name] = data[0,idx]

    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse = True)}
    return list(dict.items())
'''
def get_average_highest_sample_weights(X, coef_, vec, index_list):
    feature_names_ = vec.feature_names_
    data = X[index_list,:]
    num = data.shape[0]
    dim = data.shape[1]

    data_coef_ = data.multiply(coef_)
    i = 0
    for item in data_coef_[i]:
'''

thesis_path = Path('.').resolve().parents[0]
data_path = os.path.join(thesis_path, 'datasets', 'data-transcending-paper')

with open(os.path.join(data_path, 'extended-features-X-updated.json'), 'r') as fin:
    X = json.load(fin)
    
with open(os.path.join(data_path, 'extended-features-y-updated.json'), 'r') as fin:
    y = json.load(fin)
    
with open(os.path.join(data_path, 'extended-features-meta-updated.json'), 'r') as fin:
    meta = json.load(fin)

shas = [o['sha256'].lower() for o in meta]
t = [datetime.datetime.strptime(o['dex_date'].replace('T', ' '), '%Y-%m-%d %H:%M:%S') for o in meta]

feat_data = {
    'dex_date': t,
    'sha256': shas,
    'label': y,
}

df = pd.DataFrame(feat_data)



df2 = pd.read_csv(os.path.join(data_path, 'meta_info_file.tsv'), delimiter='\t')

# only consider some of the columns
df2 = df2[['sha256', 'families', 'dex_date', 'old_detections', 'new_detections']]

# filter out rows where sha256 or families are nan
df2 = df2[pd.notnull(df2['sha256'])]

# sort like in other metadata file 
#df2 = df2.sort_values(by='sha256').reset_index()
df2['families'] = df2['families'].fillna('[]')
df2['families'] = df2['families'].apply(get_family_name)

df2['dex_date'] = df2['dex_date'].apply(fix_date)
df2['dex_date'] = pd.to_datetime(df2['dex_date'])

df3 = df2.merge(df, left_on='sha256', right_on='sha256', how='right')

'''
print('Training classifier...')
clf = svm.LinearSVC(C=1, max_iter = 1000000)
clf.fit(X, y)

svm = '../models/LinearSVM_Model_TranscendingData.pkl'
with open(svm, 'wb') as file:
    pickle.dump(clf, file)
'''
with open('../models/LinearSVM_Model_TranscendingData.pkl', 'rb') as file:  
    clf = pickle.load(file)


X,y,t,vec = vectorize(X,y,t)
index_list_dowgin = get_family_data_index('dowgin', df3)
index_list_dnotua = get_family_data_index('dnotua', df3)
index_list_smsreg = get_family_data_index('smsreg', df3)
coef_ = csr_matrix(clf.coef_)

dowgin = get_average_sample_weights(X, coef_, vec, index_list_dowgin)
dnotua = get_average_sample_weights(X, coef_, vec, index_list_dnotua)
smsreg = get_average_sample_weights(X, coef_, vec, index_list_smsreg)

print(dowgin[0:10])
print(dnotua[0:10])
print(smsreg[0:10])
'''
dowgin = get_average_highest_sample_weights(X, coef_, vec, index_list_dowgin)
dnotua = get_average_highest_sample_weights(X, coef_, vec, index_list_dnotua)
smsreg = get_average_highest_sample_weights(X, coef_, vec, index_list_smsreg)
'''










