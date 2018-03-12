#______ECE521 Assignment 1______
#______DUE: March 16, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()

def load_date():
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
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def eucl_dist(X,Z):
    XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
    ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
    #for both...axis2 = D. for axis0 and axis2, there is a corresponding size 1.
    #makes them compatible for broadcasting

    #return the reduced sum accross axis 1. This will sum accros the D dimensional
    #element thus returning the N1xN2 matrix we desire
    return tf.reduce_sum((XExpanded-ZExpanded)**2, 1)

# x = tf.constant([[1,2,1,2,2],[3,4,1,2,2]])
# z = tf.constant([[11,22,1,23,32],[13,14,1,22,12],[2,3,4,5,6]])
# print(sess.run(eucl_dist(x,z)))

def total_loss(W, X, Y, b, decay_coeff):
    return

def linear_regression():
    B = tf.placeholder(tf.int32, name = "B")
    trainX = tf.placeholder(tf.float32, name = "trainX")
    trainY = tf.placeholder(tf.float32, name = "trainY")
    newX = tf.placeholder(tf.float32, name = "newX")
    newY = tf.placeholder(tf.float32, name = "newY")

    possibleB = [500, 1500, 3500]
    possibleLambda = [0., 0.001, 0.1, 1]
    iterations = 2000
    learning_rates = [0.005, 0.001, 0.0001]
    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')
