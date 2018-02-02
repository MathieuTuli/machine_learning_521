from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import assignment1_part1 as p1
import assignment1_part3 as p3

sess = tf.Session()

#----------Question: 2---------------------------------------------------------

#randomly generated training and test data sets
def load_data():
    np.random.seed(521)
    Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
    Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
    randIdx = np.arange(100)
    np.random.shuffle(randIdx)
    trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
    validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
    testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def training_responsibilities(distMatrix, K, testPointIndex = None):
    #find the k nearest neighbours (and their indices) in each row, where a row
    #represents a different data point. Take the -ve of the matrix since the
    #smaller the value the closer it is, bigger +ve numbers become smaller -ve
    #numbers
    neighbours, neighboursIndices = tf.nn.top_k(-distMatrix,k = K)

    #get the dimension of possible neighbours. Note distMatrix is N1xN2
    possibleNeighbours = tf.shape(distMatrix)[1]

    #make a vector of size 1xnumTrainingData, to represent possible the possible
    #indicies from distMatrix. Reshape it to make it a N2x1x1 matrix. Need
    #these dimensions for broadcasting purposes
    possibleIndices = tf.range(possibleNeighbours)
    possibleIndices = tf.reshape(possibleIndices, [-1,1,1])

    #next, we want to compare our neighboursIndices matrix with our possibleIndices
    #matrix. So, we expand in the 0 dimension to get a 1xN1xK matrix where K1
    #is the number of rows in distMatrix. This is the dimension necessary for
    #broadcasting
    neighboursIndices = tf.expand_dims(neighboursIndices, 0)

    #finally, we want to compare our possibleIndices with neighboursIndices.
    #broadcasting will take care of dimensions and then we reduce accross the 1
    #axis. Thus our N2x1x1 matrix compares to our 1xN1xK matrix and we get back
    #an N2xN1 matrix when we reduce accross the 1 axis, or K. This matrix returns
    #as True/False values, so we convert those to floating numbers. Then take
    #the transpose to get N1xN2
    kNearest = tf.transpose(tf.reduce_sum(tf.to_float(tf.equal(neighboursIndices, possibleIndices)),2))

    #since we now have a N1xN2 matric with either 0 or 1 as elemental values,
    #we return that matrix divided by K since responsibiliies will have a 1/k
    #weight to them. since kNearest is a N1xN2 matrix, we just return the
    #proper row defined by testPointIndex
    return (kNearest/tf.to_float(K))

def KNN_prediction(targetY, responsibilities):
    #y(x) = r*Y
    return tf.matmul(responsibilities, targetY)

def MSE_loss(targetY, predictedY):
    #as discussed in tut
    return tf.reduce_mean(tf.reduce_mean(tf.square(predictedY - targetY),1))

def run_KNN(trainData, trainTarget, sampleData, sampleTarget, K):
    #compute the euclidean distance between test and training
    testDistance = p1.eucl_dist(sampleData, trainData)

    #get the responsibiliies matrix
    testResponsibilities = training_responsibilities(testDistance, K)

    #compute the KNN prediction
    testPrediction = KNN_prediction(trainTarget, testResponsibilities)

    #compute the MSE
    testMSE = MSE_loss(sampleTarget, testPrediction)

    return testPrediction, testMSE

def solve_KNN():
    #load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()
    #define our placeholders
    K = tf.placeholder(tf.int32, name = "K")
    trainX = tf.placeholder(tf.float32, [None, 1], name = "trainX")
    trainY = tf.placeholder(tf.float32, [None, 1], name = "trainY")
    newX = tf.placeholder(tf.float32, [None, 1], name = "newX")
    newY = tf.placeholder(tf.float32, [None, 1], name = "newY")

    #our input matrix
    X = np.linspace(0.0,11.0, num = 1000)[:,np.newaxis]
    newTarget = np.sin(X) + 0.1 * np.power(X, 2) + 0.5 * np.random.randn(1000 , 1)

    #define possible Ks
    possibleK = [1,3,5,50]

    #compute the MSE loss for each possible k
    trainingError = []
    validationError = []
    testError = []

    plotPrediction = []

    for currK in possibleK:
        trainingErrorPrediction, trainingErrorTemp = sess.run(run_KNN(trainX, trainY, trainX, trainY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, K:currK})
        trainingError.append(trainingErrorTemp)

        validationErrorPrediction, validationErrorTemp = sess.run(run_KNN(trainX, trainY, newX, newY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, newX:validData, \
            newY:validTarget, K:currK})
        validationError.append(validationErrorTemp)

        testErrorPrediction, testErrorTemp = sess.run(run_KNN(trainX, trainY, newX, newY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, newX:testData, \
            newY:testTarget, K:currK})
        testError.append(testErrorTemp)

        # For Q2.2
        plotPredictionTemp, plotMSETemp = sess.run(run_KNN(trainX, trainY, newX, newY, K), \
        feed_dict={trainX:trainData, trainY:trainTarget, \
        newX:X, newY:newTarget, K:currK})
        plotPrediction.append(plotPredictionTemp)

        print("\nwith K = %d, the training MSE loss is %f, "
        "validation MSE loss is %f, and test MSE loss is %f." % (currK, \
        trainingErrorTemp, validationErrorTemp, testErrorTemp))

    #get the index of the minimum validator error and use that to get the
    #corresponding best K
    bestK = possibleK[validationError.index(min(validationError))]
    print('\nBest K: ', bestK, '\n\n')

    plt.figure(currK + 1)
    plt.plot(X, newTarget, '.', label='True Data')
    plt.plot(X, plotPrediction[0], '-', label='K = ' + str(possibleK[0]))
    plt.plot(X, plotPrediction[1], '-', label='K = ' + str(possibleK[1]))
    plt.plot(X, plotPrediction[2], '-', label='K = ' + str(possibleK[2]))
    plt.plot(X, plotPrediction[3], '-', label='K = ' + str(possibleK[3]))
    plt.legend(loc='best', shadow = True, fancybox = True, numpoints = 1)
    plt.title("KNN regression on data1D")
    plt.show()

    return
