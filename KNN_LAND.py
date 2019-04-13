
########## Without Manifold 

'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean

start = time.time()


df = pd.read_csv('output_terrain.csv')
df.drop(['image','image name','size','width','height'], 1, inplace=True)

df.fillna(0,inplace=True)



X = np.array(df.drop(['category'], 1).astype(float))
X = preprocessing.scale(X)
#print(X)
y = np.array(df['categorynumber'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)



y_pred=neigh.predict(X_test)
accuracy = neigh.score(X_test,y_test)
print(accuracy)

print(confusion_matrix(y_test,y_pred))
end=time.time()
print('Time taken is :  ', end-start)

precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(mean(precision)))
print('recall: {}'.format(mean(recall)))
print('fscore: {}'.format(mean(fscore)))

'''

######## With Manifold


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import time
from sklearn.manifold import Isomap
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean 
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS 

from sklearn import manifold, datasets, decomposition, discriminant_analysis




df = pd.read_csv('output_terrain_300images.csv')
df.drop(['image','image name','size','width','height'], 1, inplace=True)

df.fillna(0,inplace=True)



X = np.array(df.drop(['category'], 1).astype(float))
X = preprocessing.scale(X)
#print(X)
y = np.array(df['categorynumber'])

#X_Isomap = manifold.Isomap(n_components=1000).fit_transform(X)
#X_MDS = manifold.MDS(n_components=500).fit_transform(X)
#X_LLE = manifold.LocallyLinearEmbedding(n_components=1000).fit_transform(X)
#X_tSNE = manifold.TSNE(n_components=3).fit_transform(X)
X_SpectralEmbedding = manifold.SpectralEmbedding(n_components=1000).fit_transform(X)

start=time.time()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_SpectralEmbedding, y, test_size=0.2)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)



y_pred=neigh.predict(X_test)
accuracy = neigh.score(X_test,y_test)
print(accuracy)

print(confusion_matrix(y_test,y_pred))
end=time.time()
print('Time taken is :  ', end-start)

precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(mean(precision)))
print('recall: {}'.format(mean(recall)))
print('fscore: {}'.format(mean(fscore)))


