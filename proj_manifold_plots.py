# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from mpl_toolkits.mplot3d import Axes3D

 
digits = datasets.load_digits()
X = digits.data
y = digits.target
n_samples, n_features = X.shape

df = pd.read_csv('v3 output.csv', sep = ',', low_memory = False)
df = df.loc[:, :'n2047']
df = df.loc[3:,]
dataset = df.values

# Declaring Model
model = KMeans(n_clusters=19)

# Fitting Model
model.fit(dataset)

# Prediction on the entire data
all_predictions = model.predict(dataset)

# Printing Predictions
print(all_predictions)

'''def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    sc = ax.scatter(X[:,0], X[:,1], X[:,2], lw=0, s=40, c=X/1.)
 
    # shown_images = np.array([[1., 1.]])
    # for i in range(X.shape[0]):
    #     if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
    #     shown_images = np.r_[shown_images, [X[i]]]
    #     ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))
 
    plt.xticks([]), plt.yticks([])
    plt.title(title)
'''
#X_PCA = decomposition.PCA(n_components = 3).fit_transform(dataset)
#X_LDA = discriminant_analysis.LinearDiscriminantAnalysis(n_components = 2).fit_transform(X, y)
X_tSNE = manifold.TSNE(n_components = 3).fit_transform(dataset)
#X_MDS = manifold.MDS(n_components=3).fit_transform(dataset)
'''
nbrs = NearestNeighbors(n_neighbors=3).fit(dataset)


#embedding_plot(X_PCA, "PCA")
#embedding_plot(X_LDA, "LDA")
#embedding_plot(X_tSNE, "TSNE")
#embedding_plot(X_MDS, "MDS")
embedding_plot(nbrs, "K-means")
plt.show()'''


model_manifold = KMeans(n_clusters=19)

# Fitting Model
model_manifold.fit(X_tSNE)

# Prediction on the entire data
all_predictions_after_manifold = model_manifold.predict(X_tSNE)

# Printing Predictions
print(all_predictions_after_manifold)


