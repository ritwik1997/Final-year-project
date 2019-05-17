# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:58:34 2019

@author: rrajpuro
"""

from sklearn import preprocessing
from sklearn import manifold
from sklearn.externals import joblib

import numpy as np
import time

start=time.time()

manType = 'Isomap'  #Isomap   LLE   tSNE   SpectralEmbedding
n_comp = 100

X = np.load('v3_feature_list.npy')
X = preprocessing.scale(X)

y=np.ones(1000)
for i in range(9):
    a=np.full(1000,i+2)
    y=np.concatenate((y,a),axis=None)

'''####################################
#   Manifold Learning Algorithms
####################################'''
if manType=='Isomap':
    X_Manifold = manifold.Isomap(n_components=n_comp).fit_transform(X)
elif manType=='LLE':
    X_Manifold = manifold.LocallyLinearEmbedding(n_components=n_comp).fit_transform(X)
elif manType=='tSNE':
    X_Manifold = manifold.TSNE(n_components=n_comp).fit_transform(X)
elif manType=='SpectralEmbedding':
    X_Manifold = manifold.SpectralEmbedding(n_components=n_comp).fit_transform(X)

#np.save('X_'+manType+str(n_comp),X_Manifold)
#joblib.dump(, 'X_'+manType+str(n_comp)+'.pkl')

end = time.time()
print('Total Time taken for DimReduction is :  ', end-start)