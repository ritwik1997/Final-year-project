import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import cross_validate
import pandas as pd
#from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.manifold import Isomap
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean 
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn import model_selection 

start= time.time()

df = pd.read_csv('v3 output.csv')
df.drop(['image','image name','size','width','height','Selected'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
#print(df.head())
df.fillna(0,inplace=True)



X = np.array(df.drop(['category'], 1).astype(float))
X = preprocessing.scale(X)
#print(X)
y = np.array(df['categorynumber'])
#print(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)
print(accuracy)
print(y)
print(len(y))
print(y_test)
print(len(y_test))


precision, recall, fscore, support = score(y, y_test)


print('precision: {}'.format(mean(precision)))
print('recall: {}'.format(mean(recall)))
print('fscore: {}'.format(mean(fscore)))


