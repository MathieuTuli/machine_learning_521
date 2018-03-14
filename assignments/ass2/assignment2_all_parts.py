#______ECE521 Assignment 1______
#______DUE: March 16, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()
PIXELCOUNT = 28*28

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
        trainData = np.reshape(trainData, (trainData.shape[0],-1))
        validData = np.reshape(validData, (validData.shape[0],-1))
        testData = np.reshape(testData, (testData.shape[0],-1))
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def calculateCrossEntropyLoss(y, yHat, W, wdc):
    ''' y is the target,
        yHat is the output prediction,
        lambda (hyperparameter) is the weight decay coefficient

        Cross Entropy Loss = Ld + Lw
    '''

    Ld = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = yHat))
    Lw = (wdc / 2) * tf.reduce_sum(tf.square(W))
    crossEntropyLoss = Lw + Ld
    return crossEntropyLoss

def prediction(X, W, b):
    return tf.matmul(X, W) + b

def _1():
    #define placeholders
    trainX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "trainData")
    trainY = tf.placeholder(tf.float64, shape = [None, 1], name = "trainTarget")
    validationX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "validationData")
    validationY = tf.placeholder(tf.float64, shape = [None, 1], name = "validationTarget")
    testX = tf.placeholder(tf.float64, name="testData")
    testY = tf.placeholder(tf.float64, name = "testTarget")
    b = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biases"))
    W = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weights"))

    #Initialize
    sess.run(tf.global_variables_initializer())

    #other variables
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

    numTrainingSamples = trainData.shape[0]
    numValidationSamples = validData.shape[0]
    numTestSamples = testData.shape[0]

    possibleB = [500, 1500, 3500]
    possibleWdc = [0., 0.001, 0.1, 1]
    iterations = 20000
    learning_rates = [0.005, 0.001, 0.0001]

    indices = np.arange(0, numTrainingSamples)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningRate)

    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')
    _1()
