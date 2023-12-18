import tensorflow
import keras
from keras_preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50 
from keras.applications import ResNet50
from keras.applications import preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(open('F:\walmart\embeddings.pkl','rb')))
filenames = pickle.load(open('F:\walmart\filenames.pkl','rb'))

img = image.load_img('/content/1528.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

# for file in indices[0][1:6]:
#     temp_img = cv2.imread(filenames[file])
#     cv2_imshow('output',cv2.resize(temp_img,(1024,1024)))
#     cv2.waitKey(0)





