from argparse import ArgumentParser
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import load_model
import keras.applications as apps
import pickle
import cv2
import os

                       
from keras.applications.xception import preprocess_input

parser = ArgumentParser()
parser.add_argument('pathModel', help='path to load model', type=str)
parser.add_argument('pathData', help='path to image (test)', type=str)
args = parser.parse_args()


modelPath = args.pathModel 	#'xception'
imagePath = args.pathData 	#'TEST/TMV/GBN_2.png'

def getTopK(answer: np.array, class_list: list, K: int = 5):
    '''Get top N ordered answers'''
    top_answers = sorted([[i, val] for i, val in enumerate(answer)], key=lambda x: x[1], reverse=True)
    return [(class_list[i], val) for i, val in top_answers[:K]]


with open(modelPath + '.bin', 'rb') as class_file:
    modelName, classes = pickle.load(class_file)
if isinstance(classes, LabelBinarizer):
    classes = classes.classes_
elif isinstance(classes, OneHotEncoder):
    classes = classes.classes
else:
    raise TypeError('Classes object type is not supported ({}).'.format(type(classes).__name__))


#image
print('\nTest image: ' + imagePath  + '\n')

#load model
print('Loading model: ' + modelPath  + '.h5\n')
model = load_model(os.path.abspath(modelPath  + '.h5'))

# setting inputs
image_dim = 299
input_shape = (image_dim, image_dim, 3)

#read and preprocessing the image
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
if img.shape != (image_dim,image_dim):
    img = cv2.resize(img, (image_dim,image_dim))

img_array = np.expand_dims(img, axis=0)
img_array = preprocess_input(img_array)

####################### Prediction
y_pred1 = model.predict(img_array, steps=1)[0]
#pred = np.argmax(y_pred1, axis=1)

#model response
topk = getTopK(y_pred1, classes, 1)
output = '\n'.join('{},{}'.format(*x) for x in topk)
print('\nPredictions:\n'+ output)
