from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
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

# Load the Model back from file, where SVM_Model is the name of the pkl file
with open('../models/ScoresLinearSVM.pkl', 'rb') as file:  
    f1_scores = pickle.load(file)
    precision_scores = pickle.load(file)
    recall_scores = pickle.load(file)

num_intervals = []
num_interval_MW = 0
num_interval_GW = 0
num_intervals_MW = []
num_intervals_GW = []
for interval in intervalData:
    num_intervals.append(len(interval[1]))
    for item in interval[1]:
        if item == 1:
            num_interval_MW = num_interval_MW +1
        else:
            num_interval_GW = num_interval_GW +1
    num_intervals_MW.append(num_interval_MW)
    num_intervals_GW.append(num_interval_GW)
    

#plot
plt.figure(1)
n = math.floor((len(f1_scores)/2))
f1_scores = f1_scores[0:n]
precision_scores = precision_scores[0:n]
recall_scores = recall_scores[0:n]
x = np.linspace ( start = 1    # lower limit
                , stop = 24      # upper limit
                , num = 24      # number of points
                )

plt.xticks(np.arange(min(x)-1, max(x)+1, 1.0))
plt.yticks(np.arange(0, 1.1 , 0.1))
plt.grid(axis = 'y')

ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[6] = '6 \n 2015'
labels[18] = '18 \n 2016'
ax.set_xticklabels(labels)
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
labels = ax.get_xticklabels()
labels = [label.set_fontweight('bold') for label in [labels[0],labels[6],labels[12]]]

        

plt.xlim(xmin=0.0, xmax = 25)
plt.ylim(ymin=0.0)

plt.scatter(x, precision_scores)
plt.plot(x,precision_scores, label='Precision')
plt.plot(x,f1_scores, label='F1')
plt.scatter(x,f1_scores)
plt.plot(x,recall_scores, label='Recall')
plt.scatter(x, recall_scores)
plt.legend(loc="lower left")



plt.figure(2, figsize=(8, 4))
x = np.linspace ( start = 1    # lower limit
                , stop = 24      # upper limit
                , num = 24      # number of points
                )
plt.grid(axis = 'y')
plt.xlim(xmin=0, xmax=25)
plt.ylim(ymax = 9000)
n = math.floor(len(num_intervals)/2)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))

ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[5] = '6 \n 2015'
labels[17] = '18 \n 2016'
ax.set_xticklabels(labels)
labels = ax.get_xticklabels()
labels = [label.set_fontweight('bold') for label in [labels[11],labels[23]]]
'''
tick_label = [str(math.floor(z)) for z in np.arange(min(x), max(x)+1, 1)]
for idx,label in enumerate(tick_label):
    if idx == 5:
        tick_label[idx] = f'{label} \n 2015'
    elif idx == 17:
        tick_label[idx] = f'{label} \n 2016'
'''      
    
bar = plt.bar(x, height= num_intervals[0:n], width=0.5, align='center')
plt.ylabel('Number of samples')
# Add counts above the two bar graphs
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height+100, f'{height:.0f}', ha='center', va='bottom', rotation='vertical')

plt.show()

