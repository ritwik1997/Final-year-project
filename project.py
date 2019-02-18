import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
 
    plt.figure()
    ax = plt.subplot(aspect = 'equal')
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c = X/1.)
 
    plt.xticks([]), plt.yticks([])
    plt.title(title)

X_PCA = decomposition.PCA(n_components = 3).fit_transform(dataset)
#X_LDA = discriminant_analysis.LinearDiscriminantAnalysis(n_components = 2).fit_transform(X, y)
X_MDS = manifold.Isomap(n_components = 3).fit_transform(dataset)
X_LLE = manifold.LocallyLinearEmbedding(n_components = 3).fit_transform(dataset)


embedding_plot(X_PCA, "PCA")
#embedding_plot(X_LDA, "LDA")
embedding_plot(X_MDS, "Isomap")
embedding_plot(X_LLE, "LLE")
plt.show()
