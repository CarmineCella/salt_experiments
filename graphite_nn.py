# graphite_nn.py

from keras.models import Sequential
from keras.layers import Dense

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from pathlib import Path

## parameters #######
num_samples = 6000
split = .75
save = 1
####################

if len (sys.argv) < 2:
    print ('syntax is: python graphite_nn.py input1.csv input2.csv ...')
    quit()

intensities_size = 1738
n_classes = len (sys.argv) - 1 # number of classes depends on input files
X = np.zeros ((num_samples * n_classes, intensities_size))
y = np.zeros ((num_samples * n_classes,))
print ('data size  : ' + str (X.shape))
print ('labels size: ' + str (y.shape) + '\n')

for i in range (1, len (sys.argv)):
    arg = Path (sys.argv[i])   
    print ('processing ' +  str (arg) + '...')
    df = pd.read_csv(arg)
    a = df.to_numpy()
    idx = (i - 1) * num_samples
    for k in range (0, num_samples - 1):
        X[idx + k] = a[k]
        y[idx + k] = i

print ('\ntotal classes: ', int (y.max ()))

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = (1 - split),
                                                   random_state = 42,
                                                   shuffle=True)

print('\nX train  : ', X_train.shape)
print('X test   : ', X_test.shape)
print('y train  : ', y_train.shape)
print('y test   : ', y_test.shape)

# define the NN model
model = Sequential()
model.add(Dense(1024, input_dim=intensities_size, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the NN model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the NN model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the NN model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# eof

