import cv2
import keras
import numpy as np

model = keras.models.load_model('classification.model')

CATEGORIES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

def img2arr(path):
	arr = cv2.imread(path)
	arr = cv2.resize(arr, (100, 100))
	arr = arr/255
	arr = np.array(arr)
	return arr

predictions = model.predict([img2arr('C:\\Users\\balajiam\\Documents\\ML Data Analysis\\natural images\\natural_images\\motorbike\\motorbike_0000.jpg')])

print(CATEGORIES[predictions.argmax()])