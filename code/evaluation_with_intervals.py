from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle

from data_functions import load_features
from data_functions import vectorize
from data_functions import split, splitIntervals
from data_functions import setMWRate


'''requires folder structure:       coding
                                /     |      \
                        datasets    code    models
                          /                     \
                data-transcending-paper       SVM_Model_2014TranscendingData
'''

#get data
X, y, t = load_features("../datasets/data-transcending-paper/extended-features-X.json", "../datasets/data-transcending-paper/extended-features-y.json", "../datasets/data-transcending-paper/extended-features-meta.json")
X,y,t,vec = vectorize(X,y,t)
s =  datetime.strptime('2015-1-1 0:0:0', '%Y-%m-%d %H:%M:%S')
X_train, y_train, t_train, X_test, y_test, t_test = split(X,y,t,s)
intervalData = splitIntervals(X_test, y_test, t_test, 1)

'''
for i in range(len(intervalData)):
    X,y,t = setMWRate(intervalData[i][0], intervalData[i][1], intervalData[i][2])
    intervalData[i] = (X,y,t)
'''


# Load the Model back from file, where SVM_Model is the name of the pkl file
with open('../models/LinearSVM_Model_2014TranscendingData.pkl', 'rb') as file:  
    clf = pickle.load(file)


#evaluate
i = 0
f1_scores = []
precision_scores = []
recall_scores = []
for X_test, y_test, t_test in intervalData:
    y_predict = clf.predict(X_test)

    f1 = f1_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)

    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

    print('\n')
    print('Intervall:' + ' ' + str(i))
    print('F1:' + str(f1))
    print('Precision:' + str(precision))
    print('Recall:' + str(recall))

    
    i = i+1

#save scores
ScoresSVM = '../models/ScoresLinearSVMUpdated.pkl'
with open(ScoresSVM, 'wb') as file:
    pickle.dump(f1_scores, file)
    pickle.dump(precision_scores, file)
    pickle.dump(recall_scores, file)


