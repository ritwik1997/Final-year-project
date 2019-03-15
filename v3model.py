from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np

model_v3 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model_v3.summary()

v3_feature_list = []

for idx, dirname in enumerate(subdir):
    # get the directory names, i.e., 'dogs' or 'cats'
    # ...
    
    for i, fname in enumerate(filenames):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        
        img = image.load_img(img_path, target_size=(299, 299))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        v3_feature = model_v3.predict(img_data)
        v3_feature_np = np.array(v3_feature)
        v3_feature_list.append(v3_feature_np.flatten())
        
v3_feature_list_np = np.array(v3_feature_list)

kmeans = KMeans(n_clusters=2, random_state=0).fit(v3_feature_list_np)