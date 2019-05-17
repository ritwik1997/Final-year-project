# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:16:26 2019

@author: rrajpuro
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score 
from statistics import mean
from shutil import copy,rmtree
import pickle
import sys
import os
import time
import numpy as np
import pandas as pd

outputLabel = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
    
def svmFit(X_train,y_train):
    print('\nSVM Classifier')
    clf = SVC(gamma='scale')
    clf.fit(X_train,y_train)
    return clf

def knnFit(X_train,y_train):
    print('\nKNN Classifier')
    clf = KNeighborsClassifier(n_neighbors=10,weights='distance')
    clf.fit(X_train,y_train)
    return clf

def rfcFit(X_train,y_train):
    print('\nRandom Forest Classifier')
    clf = RandomForestClassifier(n_jobs=2,n_estimators=100,max_depth=15, random_state=0)
    clf.fit(X_train,y_train)
    return clf
    
def classifyAll(X_test,y_test,clf):
    y_pred=clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(confusion_matrix(y_test,y_pred))
    
    precision, recall, fscore, support = score(y_test, y_pred)
    print('Accuracy:',accuracy)
    print('precision: {}'.format(mean(precision)))
    print('recall: {}'.format(mean(recall)))
    print('fscore: {}'.format(mean(fscore)))

def classify(path,clf,testImages,X_test):
    n = testImages.where(testImages==path).dropna()
    label = n.index[0]
    h = clf.predict(X_test.loc[label].values.reshape(1,-1))
    return str(outputLabel[int(h[0])-1])

def loadData(manType):
    Xnp = np.load('X_'+manType+'.npy')
    #X_Manifold = np.insert(X_Manifold, 0, np.arange(1,10001,dtype=int), axis=1)
    X = pd.DataFrame(Xnp)
    
    y=np.ones(1000)
    for i in range(9):
        a=np.full(1000,i+2)
        y=np.concatenate((y,a),axis=None)
    y = pd.Series(y)
    
    return X,y

def load(manType):
    
    X_Man, y = loadData(manType)
    
    #orig_stdout = sys.stdout
    #f = open('out_'+manType+'_'+time.strftime("%Y%m%d-%H%M%S")+'.txt', 'w+')
    #sys.stdout = f
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_Man, y, test_size=0.2)        
        
    dirName = 'C:/Final Project/2750'
    testDir = 'C:/Final Project/test'
    path = []
    label = np.empty(2000, dtype = int)
        
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    
    if os.path.exists(testDir):
        rmtree(testDir, ignore_errors=True)
        os.mkdir(testDir)
    else:
        os.mkdir(testDir)
    
    for b in range(2000):
        label[b] = X_test.index[b]
        k = listOfFiles[label[b]]
        copy(k, testDir)
        path.append(os.path.basename(k))
    
    testImages = pd.Series(path, index = label)
    
    return X_train, X_test, y_train, y_test, testImages

def main():
    manType = 'tSNE3'
    X_train, X_test, y_train, y_test, testImages = load(manType)
    sclf = svmFit(X_train,y_train)
    classifyAll(X_test,y_test,sclf)
    neigh = knnFit(X_train,y_train)
    classifyAll(X_test,y_test,neigh)
    fort = rfcFit(X_train,y_train)
    classifyAll(X_test,y_test,fort)

#with open('model_svm'+manType+'.pkl', 'wb') as f:
#    pickle.dump(sclf, f)
#with open('model_knn'+manType+'.pkl', 'wb') as f:
#    pickle.dump(neigh, f)
#with open('model_rfc'+manType+'.pkl', 'wb') as f:
#    pickle.dump(clf, f)
#        