# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:00:40 2019

@author: rrajpuro
"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from matplotlib import pyplot as plt

import numpy as np
import os
import time
import sys

i=0
start = time.time()

model_v3 = InceptionV3(include_top=False, weights='imagenet', input_shape=None, pooling='avg')
#model_v3 = AveragePooling2D((8, 8), strides=(8, 8))(base_model)
#model_v3 = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
#model_v3.summary()

v3_feature_list = np.empty(shape=(10000,2048))

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

dirName = "C:/Final Project/2750"
    
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
print('Completed:\n')
# Print the files
for img_path in listOfFiles:
#    start = time.time()
    img = image.load_img(img_path, target_size=(299, 299))
#    plt.imshow(img)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    v3_feature = model_v3.predict(img_data)
    v3_feature_list[i]=v3_feature
    i+=1
    sys.stdout.write("\r %s %%" % (i*100/10000))
    sys.stdout.flush()
    break
        
#v3_feature_list=np.append(v3_feature_list v3_feature_np)
#v3_feature_list_np = np.array(v3_feature_list)

np.save('v3_feature_list_avgpool',v3_feature_list)

end = time.time()

print('Time taken to execute : ' , end-start)