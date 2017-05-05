#!/usr/bin/python
# coding=utf-8

from __future__ import print_function
import numpy as np
import time
from scipy import ndimage
import math
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import os
import glob
# from scipy.misc import imread, imresize
import cPickle as pickle
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

from .googlenet import create_googlenet
from sklearn.metrics import log_loss
import h5py
import argparse


# from VGGCAM import train_VGGCAM

def create_submission(predictions, name):
    f = open(name, 'w')
    f.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
    for i in range(1, 5):
        test_id = np.load("test_id_%d.npy" % i)
        predict_small = predictions[i - 1]
        m = len(predict_small)
        for j in range(0, m):
            n = predictions[j].shape[0]
            line = ""
            line += "%s," % test_id[j]
            for k in range(10):
                line += str(predict_small[j][k])
                if (l != 9): line += ","
            line += "\n"
            f.write(line)

    f.close()


def merge_several_folds_mean(data, nfolds):
    res = []
    for i in range(4):
        res.append(np.mean(data[:, i]))
    """
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    """
    return res


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1. / nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    target = []
    index = []
    data = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data)
    target = np.array(target)
    index = np.array(index)
    return data, target, index


def vgg16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.add(Dense(10, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=5e-4, decay=4e-3, momentum=0.9, nesterov=True)
    # adam = Adam()
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def vgg16_256_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    weight_path = 'vgg16_weights.h5'

    f = h5py.File(weight_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        if k == 32:
            weights = []
            weight_small = g['param_0']
            zero = np.zeros((7680, 4096))
            weights.append(np.vstack((weight_small, zero)))
            weights.append(g['param_1'])
        else:
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    model.add(Dense(10, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=5e-4, decay=4e-3, momentum=0.9, nesterov=True)
    adam = Adam()
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def rotate_shift_augmentation(X, angle_range, h_range, w_range):
    # X_aug = np.copy(X)
    size = X.shape[2:]
    for i in range(0, len(X)):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(0, X.shape[1]):
            X[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1, mode='nearest')
            X[i, j] = ndimage.shift(X[i, j], (h_shift, w_shift), order=0, mode='nearest')
    return X


def augment(X, target):
    # X_aug=np.copy(X)
    n = len(X)
    box = np.zeros((3, 60, 60)) - 25.76767
    for i in range(0, n):
        classes = range(10)
        idx, idy = np.random.randint(224 - 60, size=2)
        num = target[i]
        del classes[num]
        rand = random.choice(classes)
        X[i, :, idx:idx + 60, idy:idy + 60] = advsample[rand]
        # X[i,:,idx:idx+60,idy:idy+60]=box
    return X


if __name__ == '__main__':
    color_type_global = 3
    img_rows, img_cols = 224, 224
    target = np.load('input/train_target_mod1.npy')
    train_target = np_utils.to_categorical(target, 10)
    driver_id = np.load('input/driverid_mod1.npy')
    unique_drivers = np.load('unique_drivers.npy')
    advsample = np.load('advsample.npy').transpose((0, 3, 1, 2))
    nb_epoch = 15

    model = create_googlenet('googlenet_weights.h5')
    # model = vgg16_model(img_rows,img_cols, color_type_global)

    weights_path = os.path.join('googlenet_weights_best_semi3.h5')

    train_data = np.load('input/train_data_mod1.npy')
    unique_list_train = ['p012', 'p014', 'p016', 'p021', 'p022', 'p024', 'p035', 'p045', 'p049', 'p050', 'p051', 'p052', 'p072', 'p081', 'p039', 'p056', 'p061', 'p002', 'p041', 'p066', 'p015']
    X_train, y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p026', 'p042', 'p047', 'p064', 'p075']
    X_valid, y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)
    del train_data

    print('Split train: ', len(X_train), len(y_train))
    print('Split valid: ', len(X_valid), len(y_valid))
    print('Train drivers: ', unique_list_train)
    print('Test drivers: ', unique_list_valid)
    callbacks = [
        ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
    ]
    np.save('X_train', X_train)
    del X_train

    for i in range(nb_epoch):
        for j in range(1, 5):
            test_data = np.load("input/test_data_%d.npy" % j)
            pred_target = np.load('testpredall%d.npy' % j)
            pred_target_argmax = np.argmax(pred_target, axis=1)
            n = len(pred_target)
            for i in range(n):
                if np.max(pred_target[i]) > 0.8:
                    x = np.argmax(pred_target[i])
                    pred_target[i] = np_utils.to_categorical(np.expand_dims(np.array(x), axis=0), 10)

            test_data = augment(test_data, pred_target_argmax)
            test_data = rotate_shift_augmentation(test_data, 15, 0.15, 0.15)
            model.fit(test_data, [pred_target, pred_target, pred_target], batch_size=32, nb_epoch=1,
                      verbose=1, callbacks=callbacks, shuffle=True)
            del test_data

        X_aug = np.load('X_train.npy')
        X_aug = augment(X_aug, target)
        X_aug = rotate_shift_augmentation(X_aug, 15, 0.15, 0.15)
        model.fit(X_aug, [y_train, y_train, y_train], batch_size=32, nb_epoch=1,
                  verbose=1, validation_data=(X_valid, [y_valid, y_valid, y_valid]),
                  callbacks=callbacks, shuffle=True)
        del X_aug

    del X_valid

    if os.path.isfile(weights_path):
        model.load_weights(weights_path)

    # Predict
    res = []
    for i in range(1, 5):
        test_data = np.load("input/test_data_%d.npy" % i)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        res.append(test_prediction)
        del test_data
        np.save('predgooglenetadvsysthesis0730', np.array(res))


        # test_res = merge_several_folds_mean(yfull_test, nfolds)
        # np.save('res.npy',np.array(test_res))
        # np.save('res_id.npy',np.array(test_id))
        # test_id = np.load('test_id.npy')
        # create_submission(res, "submission22_epoch1.csv")