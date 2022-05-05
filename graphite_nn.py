# graphite_nn.py

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.optimizers import SGD

## parameters #######
num_samples = 6000
n_epochs = 20
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

class_num = 0
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

# preprocess dataset
X = StandardScaler().fit_transform(X)

# make splits
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = (1 - split),
                                                   random_state = 42,
                                                   shuffle=True)

print('\nX train  : ', X_train.shape)
print('X test   : ', X_test.shape)
print('y train  : ', y_train.shape)
print('y test   : ', y_test.shape)
print ()

# define the NN model
model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(intensities_size,)),
    tf.keras.layers.Dense(128, input_dim=intensities_size, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes)
])

# compile the NN model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit the NN model on the dataset
history = model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=n_epochs)

# evaluate the NN model
train_loss, train_acc = model.evaluate(X_train,  y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\ntrain accuracy:', train_acc)
print('test accuracy :', test_acc)

# plot loss during training
plt.figure (1)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
if save == 1:
    plt.savefig ('results/nn_loss_' + str (n_epochs) + '_epochs_' + str (n_classes) + '_classes.png')
plt.figure (2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
if save == 1:
    plt.savefig ('results/nn_accuracy_' + str (n_epochs) + '_epochs_' + str (n_classes) + '_classes.png')

plt.show()

# eof

