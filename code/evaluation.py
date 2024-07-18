from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


X, y, t = load_features("../datasets/data-transcending-paper/extended-features-X-updated.json", "../datasets/data-transcending-paper/extended-features-y-updated.json", "../datasets/data-transcending-paper/extended-features-meta-updated.json")
X,y,t,vec = vectorize(X,y,t)
s =  datetime.strptime('2015-1-1 0:0:0', '%Y-%m-%d %H:%M:%S')
X_train, y_train, t_train, X_test, y_test, t_test = split(X,y,t,s)


# Load the Model back from file, where SVM_Model is the name of the pkl file
with open('../models/LinearSVM_Model_2014TranscendingDataUpdated.pkl', 'rb') as file:  
    clf = pickle.load(file)

print('Predicting...')
y_predict = clf.predict(X_test)

f1_score = f1_score(y_test, y_predict)
precision_score = precision_score(y_test, y_predict)
recall_score = recall_score(y_test, y_predict)

print('F1:' + str(f1_score))
print('Precision:' + str(precision_score))
print('Recall:' + str(recall_score))

