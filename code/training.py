from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import pickle

from data_functions import load_features
from data_functions import vectorize
from data_functions import split


'''requires folder structure:       coding
                                /     |      \
                        datasets    code    models
                          /                     \
                data-transcending-paper       SVM_Model_2014TranscendingData
'''

X, y, t = load_features("../datasets/data-transcending-paper/extended-features-X.json", "../datasets/data-transcending-paper/extended-features-y.json", "../datasets/data-transcending-paper/extended-features-meta.json")
X,y,t,vec = vectorize(X,y,t)
s =  datetime.strptime('2015-1-1 0:0:0', '%Y-%m-%d %H:%M:%S')
X_train, y_train, t_train, X_test, y_test, t_test = split(X,y,t,s)

print('Training classifier...')
clf = svm.LinearSVC(C=1, max_iter = 1000000)
clf.fit(X_train,y_train)

print('Saving model')
SVM_Model = '../models/LinearSVM_Model_2014TranscendingData.pkl'
with open(SVM_Model, 'wb') as file:
    pickle.dump(clf, file)


# Load the Model back from file, where SVM_Model is the name of the pkl file
'''with open(SVM_Model, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model'''
