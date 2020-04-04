import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

DIRECTORY = r'C:\\Users\\balajiam\\Documents\\ML Data Analysis\\natural images\\natural_images'
CATEGORIES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
IMG_SIZE = (100, 100)

data = []

for folder in os.listdir(DIRECTORY):
	path = os.path.join(DIRECTORY, folder)
	for img in os.listdir(path):
		try:
			img_path = os.path.join(path, img)
			img_arr = cv2.imread(img_path)
			img_arr = cv2.resize(img_arr, IMG_SIZE)
			img_arr = img_arr/255
			class_num = CATEGORIES.index(folder)
			data.append([img_arr, class_num])
		except Exception as e:
			print(e)

random.shuffle(data)

X = []
y = []

for feature, label in data:
	X.append(feature)
	y.append(label)

X = np.array(X)
y = np.array(y)

import pickle

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))