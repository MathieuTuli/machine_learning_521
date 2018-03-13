#______ECE521 Assignment 1______
#______DUE: March 16, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()

def load_data():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        #shuffle and reshape data
        trainData = np.reshape(trainData, (3500,-1))
        validData = np.reshape(validData, (100,-1))
        testData = np.reshape(testData, (145,-1))
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def hypothesis(W,X):
    return tf.matmul(X, W)

def update(W, learning_rate, X, Y, B):
    newW = W - total_loss
    return newW

#not sure what 'b' is for
def total_loss(W, X, Y, b, decay_coeff):
    hypothesis = hypothesis(X, W)
    MSE = tf.matmul((hypothesis + b - Y), X)
    WDL = W * tf.to_float(decay_coeff)
    return MSE + WDL


def linear_regression():
    B = tf.placeholder(tf.int32, name = "B")
    trainX = tf.placeholder(tf.float32, name = "trainX")
    trainY = tf.placeholder(tf.float32, name = "trainY")
    newX = tf.placeholder(tf.float32, name = "newX")
    newY = tf.placeholder(tf.float32, name = "newY")

    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()
    possibleB = [500, 1500, 3500]
    possibleLambda = [0., 0.001, 0.1, 1]
    iterations = 2000
    learning_rates = [0.005, 0.001, 0.0001]
    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')
    linear_regression()
