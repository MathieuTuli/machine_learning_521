#______ECE521 Assignment 3______
#______DUE: April 6, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit

sess = tf.Session()
PIXELCOUNT = 28*28

def load_data():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        #shuffle and reshape data
        # trainData = np.reshape(trainData, (trainData.shape[0] ,-1))
        # validData = np.reshape(validData, (validData.shape[0],-1))
        # testData = np.reshape(testData, (testData.shape[0],-1))
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def linear_regression():
    #DEFINE PLACEHOLDERS
    X = tf.placeholder(tf.float32, name="data")
    Y = tf.placeholder(tf.float32, name="target")

    #INITIALIZE
    sess.run(tf.global_variables_initializer())

    #for 1.2
    numHiddenUnits = [100, 500, 1000]

    #for 1.3
    dropoutRate = 0.5

    #for part 1.4
    logLearnRateBounds[-7.5, -4.5]
    numLayersBounds = [1, 5]
    numHiddenUnitsBounds = [100, 500]
    wdcBounds = [-9, -6]
    dropoutBoolean = [0, 1]


    #GENERAL VARIABLES
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')
    linear_regression()
