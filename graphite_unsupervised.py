# graphite_unsupervised.py

import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from pathlib import Path

intensities_size = 1738
num_samples = 10000

if len (sys.argv) < 2:
    print ('syntax is: python graphite_unsupervised.py input1.csv input2.csv ...')
    quit()

n_clusters = len (sys.argv) - 1 # number of cluster depends on input files
print ('cluster(s):', n_clusters)

T = np.zeros ((num_samples * n_clusters, intensities_size))
print ('data size : ' + str (T.shape) + '\n')

for i in range (1, len (sys.argv)):
    arg = Path (sys.argv[i])   
    print ('processing ' +  str (arg) + '...')
    df = pd.read_csv(arg)
    a = df.to_numpy() + (i  * 100)
    idx = (i - 1) * num_samples
    for k in range (0, num_samples - 1):
        T[idx + k] = a[k]

# preprocessing
print ('\nnormalizing...')
T = preprocessing.Normalizer().fit_transform(T)

# Clustering using KMeans
print ('\nclustering...')
kmean_model = KMeans(n_clusters=n_clusters)
kmean_model.fit(T)
centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
print('centroid(s):', centroids.shape)
print('label(s)   :', labels.size)

print ('\ndimensionality reduction...')
# Dimesionality reduction to 2
dim_model = PCA(n_components=2, random_state=0)
dim_model.fit(T) # fit the model
T = dim_model.transform(T) # transform the 'normalized model'

# transform the 'centroids of KMean'
centroid_dim = dim_model.transform(centroids)
print ('[reduced centroid(s)]')
print(centroid_dim)

# colors for plotting
colors = ['blue', 'red', 'green', 'orange'] # FIXME only 4 classes
# assign a color to each features (note that we are using features as target)
features_colors = [colors[labels[i]] for i in range(len(T))]

# plot the reduced components
plt.scatter(T[:, 0], T[:, 1],
            c=features_colors, marker='o',
            alpha=0.4
        )

# plot the centroids
plt.scatter(centroid_dim[:, 0], centroid_dim[:, 1],
            marker='x', s=100,
            linewidths=3, c='black'
        )

for i in range (0, n_clusters):
    plt.text(centroid_dim[i][0],
    centroid_dim[i][1], os.path.splitext (os.path.basename(sys.argv[i + 1]))[0], 
    color='black', fontsize=14)

plt.title ('Graphite (unsupservied): ' + str (n_clusters) + ' classe(s)')
plt.xlabel ('Dim 1')
plt.ylabel ('Dim 2')
plt.savefig ('clustering_' + str (n_clusters) + '.png')

plt.show()