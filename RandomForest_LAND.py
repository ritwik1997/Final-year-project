import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import time
from sklearn.metrics import confusion_matrix
from sklearn.manifold import Isomap
from sklearn.metrics import precision_recall_fscore_support as score
from statistics import mean 
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS 
from sklearn import manifold, datasets, decomposition, discriminant_analysis

#start = time.time()
'''X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k"]
'''

'''class K_Means:
    def __init__(self, k=19, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,   data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification




'''

df = pd.read_csv('output_terrain_300images.csv')
df.drop(['image','image name','size','width','height'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
#print(df.head())
df.fillna(0,inplace=True)

'''def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df

df = handle_non_numerical_data(df) 
#print(df.head())
'''
# add/remove features just to see impact they have.
#df.drop(['ticket','home.dest'], 1, inplace=True)
X = np.array(df.drop(['category'], 1).astype(float))
X = preprocessing.scale(X)
#print(X)
y = np.array(df['categorynumber'])
#X_Isomap = manifold.Isomap(n_components=1000).fit_transform(X)
#X_MDS = manifold.MDS(n_components=500).fit_transform(X)
#X_LLE = manifold.LocallyLinearEmbedding(n_components=1250).fit_transform(X)
#X_tSNE = manifold.TSNE(n_components=3).fit_transform(X)
X_SpectralEmbedding = manifold.SpectralEmbedding(n_components=1000).fit_transform(X)

start=time.time()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_SpectralEmbedding, y, test_size=0.2)


clf = RandomForestClassifier(n_jobs=2,n_estimators=200,max_depth=20,
                              random_state=0)

clf.fit(X_train,y_train)
'''X = np.array(df.drop(['category'], 1).astype(float))
X = preprocessing.scale(X)
#print(X)
y = np.array(df['categorynumber'])


#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X,y)
'''
'''correct = 0
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1


print(correct/len(X))
'''
y_pred=clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(accuracy)
print(confusion_matrix(y_test,y_pred))

end=time.time()
print('Time taken to execute : ' , end-start)


precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(mean(precision)))
print('recall: {}'.format(mean(recall)))
print('fscore: {}'.format(mean(fscore)))