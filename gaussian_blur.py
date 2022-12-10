# imports
import random
import cv2
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import pickle

# this code gaussian blurs the images

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()

# print the shapes of the data
print(trainX.shape)
print(trainy.shape)
a = trainX[1]
print(a.shape)


# unpickle method to read dataset
def unpickle(file):

    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


# reading file
file1 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_batch_1"
file2 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_batch_2"
file3 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_batch_3"
file4 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_batch_4"
file5 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_batch_5"
file6 = "C:\\Users\gabri\OneDrive\Documents\GitHub\BlurredImage\cifar-10-batches-py\data_test_batch"

filelist = [file1, file2, file3, file4, file5, file6]


# Cleaning the working directories
pictures_root = "gblurred_pictures/gblur"
for root, dirs, files in os.walk(pictures_root, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
kernelSize = [0, 1, 3, 5, 7, 9]
for ks in kernelSize:
    dirName = pictures_root + "/ks" + str(ks)
    os.mkdir(dirName)
global_img_index = 0


# add for loop over the batches here
for file in filelist:
    blurred_ds = unpickle(file)
    # data -- a 10000x3072 numpy array of uint8s.
    # Each row of the array stores a 32x32 colour image.
    # The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    # The image is stored in row-marnd_nror order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # imgs[b'data']
    for k in range(10000):
        rnd_nr = random.choice(kernelSize)
        img = np.reshape(blurred_ds[b'data'][k], (3, 32, 32))
        img = np.moveaxis(img, 0, -1)
        print("rnd_nr: ", rnd_nr)
        print("index: ", global_img_index)
        if rnd_nr != 0:
            gblur = cv2.GaussianBlur(img, (rnd_nr, rnd_nr), 0)
        else:
            gblur = img
        filename_img = 'gblurred_pictures/img/Image' + \
            str(global_img_index) + '_' + str(rnd_nr)+'.jpg'
        filename = 'gblurred_pictures/gblur/ks' + str(rnd_nr) + '/gblur_Image' + \
            str(global_img_index) + '_' + str(rnd_nr)+'.jpg'

        gblur = np.squeeze(gblur)
        titles = ['Image', 'GaussianBlur ' + str(rnd_nr)]

        if not cv2.imwrite(filename_img, img):
            raise Exception("Could not write image")

        if not cv2.imwrite(filename, gblur):
            raise Exception("Could not write blurred image")
        global_img_index += 1
