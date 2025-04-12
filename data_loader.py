import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split


def getfile(filePath):
    return filePath.split('\\')[-1]

def InitData(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names = columns)
    data['Center'] = data['Center'].apply(getfile)
    print('Total Images Imported', data.shape[0])
    return data

def balanceDataSet(data, display=True):
    nBin = 21
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.05)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    removeList = []
    for i in range(nBin):
        binDataList = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data['Steering'][j] <= bins[i + 1]:
                binDataList.append(j)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeList.extend(binDataList)
        
    print('Removed Images: ', len(removeList))
    data.drop(data.index[removeList], inplace=True)
    print('Remaining Images: ', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist,width=0.05)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    return data

# Load images and steering angles
def load_images_and_steering(data, img_folder='IMG'):
    images = []
    steerings = []
    for i in range(len(data)):
        img_path = os.path.join(img_folder, data.iloc[i]['Center'])
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            steerings.append(float(data.iloc[i]['Steering']))
    return np.array(images), np.array(steerings)


data = InitData('.')
data = balanceDataSet(data, display=True)

images, steerings = load_images_and_steering(data)
images = images / 255.0  # Normalize

X_train, X_val, y_train, y_val = train_test_split(images, steerings, test_size=0.2, random_state=42)

