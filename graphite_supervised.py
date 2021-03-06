# graphite_supervised.py

import numpy as np 
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import sys
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

## parameters #######
num_samples = 6000
split = .75
kn_neighbors = 5
rf_estimators = 7
save = 1
####################

if len (sys.argv) < 2:
    print ('syntax is: python graphite_supervised.py input1.csv input2.csv ...')
    quit()

intensities_size = 1738
n_classes = len (sys.argv) - 1 # number of classes depends on input files
X = np.zeros ((num_samples * n_classes, intensities_size))
y = np.zeros ((num_samples * n_classes,))
print ('data size  : ' + str (X.shape))
print ('labels size: ' + str (y.shape) + '\n')

class_num = 0;
for i in range (1, len (sys.argv)):
    arg = Path (sys.argv[i])   
    print ('processing ' +  str (arg) + '...')
    df = pd.read_csv(arg)
    a = df.to_numpy()
    idx = (i - 1) * num_samples
    for k in range (0, num_samples - 1):
        X[idx + k] = a[k]
        y[idx + k] = class_num
    class_num += 1

print ('\ntotal classes: ', n_classes)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = (1 - split),
                                                   random_state = 42,
                                                   shuffle=True)

print('\nX train  : ', X_train.shape)
print('X test   : ', X_test.shape)
print('y train  : ', y_train.shape)
print('y test   : ', y_test.shape)

clfs = []

clfs.append(LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto' ))
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=kn_neighbors))
clfs.append(RandomForestClassifier(n_estimators=rf_estimators))

results = 'samples: ' + str (num_samples) + '\nkn_neighbors: ' + str (kn_neighbors)  + '\nrf_estimators: ' + str (rf_estimators) + '\nsplit: ' + str (split) + '\n\n'

print ("\nrunning classifications...")
for classifier in clfs:
    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('clf', classifier)
    ])
    print('---------------------------------')
    print(str(classifier))
    print('---------------------------------')  
    shuffle = KFold (n_splits=5, random_state=5, shuffle=True)
    scores = cross_val_score (pipeline, X, y, cv=shuffle)

    print("model scores: ", scores)
    print("average score: ", scores.mean ())
    results += str (classifier) + ": " + str (scores.mean () * 100) + "%\n"
    pipeline.fit (X_train, y_train)
    ncvscore = pipeline.score(X_test, y_test)
    print("non cross-validated score: ", ncvscore)

results += '\n'
if save == 1:
    text_file = open("results/classification_" + str (int (y.max ())) + "_" + str (num_samples) + ".txt", "w")
    n = text_file.write(results)
    text_file.close()

# eof
