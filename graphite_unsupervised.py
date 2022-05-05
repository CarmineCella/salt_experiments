# graphite_unsupervised.py

import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path
from sklearn.utils import shuffle

intensities_size = 1738
num_samples = 6000

if len (sys.argv) < 2:
    print ('syntax is: python graphite_unsupervised.py input1.csv input2.csv ...')
    quit()

n_clusters = len (sys.argv) - 1 # number of cluster depends on input files
print ('cluster(s) :', n_clusters)
print ('sample(s)  :', num_samples)
print ('features(s):', intensities_size)

T = np.zeros ((num_samples * n_clusters, intensities_size))
print ('data size  : ' + str (T.shape) + '\n')

for i in range (1, len (sys.argv)):
    arg = Path (sys.argv[i])   
    print ('processing ' +  str (arg) + '...')
    df = pd.read_csv(arg)
    a = df.to_numpy()
    idx = (i - 1) * num_samples
    for k in range (0, num_samples - 1):
        T[idx + k] = a[k] #np.abs (np.fft.fft (a[k]))

T_orig = np.copy(T)
T = shuffle (T)

# preprocessing
print ('\nnormalizing...')
T = preprocessing.Normalizer().fit_transform(T)
T_orig = preprocessing.Normalizer().fit_transform(T_orig)

# Clustering using KMeans
print ('\nclustering...')
kmean_model = KMeans(n_clusters=n_clusters)
kmean_model.fit(T)
centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
print('centroid(s):', centroids.shape)
print('label(s)   :', labels.size)

dim_method = 'pca'
print ('\ndimensionality reduction (' + dim_method + ')...')
# Dimesionality reduction to 2
if dim_method == 'isomap':
    dim_model = Isomap(n_components=2)
    dim_model_orig = Isomap(n_components=2)
elif dim_method == 'pca':
     dim_model = PCA(n_components=2)
     dim_model_orig = PCA(n_components=2)
elif dim_method == 'tsvd':
     dim_model = TruncatedSVD(n_components=2)
     dim_model_orig = TruncatedSVD(n_components=2)
else:
    print ('invalid reduction method')
    exit ()

T = dim_model.fit_transform (T)
T_orig = dim_model_orig.fit_transform (T_orig)

# transform the 'centroids of KMean'
centroid_dim = dim_model.transform(centroids)
print ('[reduced centroid(s)]')
print(centroid_dim)

# colors for plotting
colors = ['blue', 'red', 'green', 'orange'] # FIXME only 4 classes
# assign a color to each features (note that we are using features as target)
features_colors = [colors[labels[i]] for i in range(len(T))]

plt.figure(1)
# plot the reduced components with GT colors
for i in range (0, n_clusters):
    plt.scatter(T_orig[i*num_samples:i*num_samples+num_samples, 0], T_orig[i*num_samples:i*num_samples+num_samples, 1],
                c=colors[i], marker='.',
                alpha=0.4, 
                label=os.path.splitext (os.path.basename(sys.argv[i + 1]))[0]
            )
plt.title ('Graphite: ground truth')
plt.xlabel (dim_method + ' 1')
plt.ylabel (dim_method + ' 2')
plt.legend ()
plt.savefig ('results/gt_' + str (n_clusters) + '_' + str (dim_method)  + '_' + str (num_samples) + '.png')

plt.figure (2)
# plot the reduced components
plt.scatter(T[:, 0], T[:, 1],
            c=features_colors, marker='.',
            alpha=0.4
        )

# plot the centroids
plt.scatter(centroid_dim[:, 0], centroid_dim[:, 1],
            marker='x', s=100,
            linewidths=3, c='black'
        )

for i in range (0, n_clusters):
    plt.text(centroid_dim[i][0],
    centroid_dim[i][1], 'class ' + str (i + 1), 
    color='black', fontsize=14)

plt.title ('Graphite: ' + str (n_clusters) + ' classe(s) - unsupervised')
plt.xlabel (dim_method + ' 1')
plt.ylabel (dim_method + ' 2')
plt.savefig ('results/clustering_' + str (n_clusters) + '_' + str (dim_method) + '_' + str (num_samples) + '.png')

plt.show()

