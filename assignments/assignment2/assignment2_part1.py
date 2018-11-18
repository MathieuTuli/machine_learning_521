#______ECE521 Assignment 2______
#______DUE: March 16, 2018______

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
        trainData = np.reshape(trainData, (trainData.shape[0] ,-1))
        validData = np.reshape(validData, (validData.shape[0],-1))
        testData = np.reshape(testData, (testData.shape[0],-1))
        return trainData, trainTarget, validData, validTarget, testData, testTarget

def calc_MSE(Y, X, W, b, wdc):
    hypothesis = tf.matmul(X,W) + b
    Ld = tf.reduce_mean(tf.reduce_mean(
                tf.square(hypothesis - Y), 1))

    Lw = (wdc) * tf.reduce_sum(tf.square(W))
    return Lw + Ld

def linear_regression():
    #DEFINE PLACEHOLDERS
    X = tf.placeholder(tf.float32, name="data")
    Y = tf.placeholder(tf.float32, name="target")

    #INITIALIZE
    sess.run(tf.global_variables_initializer())

    #GENERAL VARIABLES
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

    sizeTrainData = trainData.shape[0]
    sizeValidationData = validData.shape[0]
    sizeTestData = testData.shape[0]

    possibleB = [500, 1500, 3500]
    numB = [sizeTrainData // possibleB[0],
                sizeTrainData // possibleB[1],
                    sizeTrainData // possibleB[2]]
    possibleWdc = [0., 0.001, 0.1, 1]
    iterations = 20000
    possibleRates = [0.005, 0.001, 0.0001]

    indices = np.arange(0, sizeTrainData)
    shuffledTrainingData = []
    shuffledTrainingTarget = []


    #-----------------PART 1.1--------------------------------------------------
    print("\n-----PART 1.1-----\n\n")
    #variables specigic to PART 1.1:
    rateLosses = [0.,0.,0.]
    trainAccuracies = [0.,0.,0.]

    fig = plt.figure()
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS PER EPOCH")

    rateIndicator = 0
    for rate in possibleRates:
        #reset W and b for each iterations
        W = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float32))
        b = tf.Variable(tf.truncated_normal([1], stddev = 0.1, dtype = tf.float32))
        sess.run(tf.global_variables_initializer())
        MSELoss = calc_MSE(Y, X, W, b, possibleWdc[0])

        print("Rate:", rate)
        optimizer = tf.train.GradientDescentOptimizer(rate)
        train = optimizer.minimize(MSELoss)


        numEpochs = 0
        lossPerIter = []
        for i in range(iterations):
            batchIndicator = (i % numB[0]) * possibleB[0]

            if not (i % numB[0]):
                np.random.shuffle(indices)
                shuffledTrainingData = trainData[indices]
                shuffledTrainingTarget = trainTarget[indices]
            #since we want to consider epochs, we don't be taking random
            #indices rather, will be grabing batches of size B = 500, sliding
            #over the data set each iteration
            #also shuffle it
            currData, currTarget = shuffledTrainingData[batchIndicator:batchIndicator + possibleB[0]], \
                                shuffledTrainingTarget[batchIndicator:batchIndicator + possibleB[0]]

            sess.run(train, feed_dict = {X: currData, Y: currTarget})

            if ((i+1) % numB[0]) == 0:
                #divide by 2 since we didn't in the MSE function due to
                #matrix complications
                rateLosses[rateIndicator] = (sess.run(MSELoss, feed_dict = {X: currData, Y: currTarget})) / 2
                lossPerIter.append(rateLosses[rateIndicator])

            # print("Epoch:", numEpochs, "| Loss:", lossPerIter[i])
            # numEpochs += 1
        hypothesis = (tf.matmul(X,W) + b)
        classified = tf.to_float(tf.greater(hypothesis, 0.5))
        numCorrect = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(trainTarget), classified)))
        trainAccuracy = sess.run(numCorrect / trainTarget.shape[0], feed_dict = {X: trainData}) * 100
        trainAccuracies[rateIndicator] = trainAccuracy

        #plot
        epochs = np.linspace(0, int(len(lossPerIter)), num = int(len(lossPerIter)))
        if(rateIndicator == 0):
            plt.plot(epochs, lossPerIter, 'r-')
        elif(rateIndicator == 1):
            plt.plot(epochs, lossPerIter, 'g-')
        elif(rateIndicator == 2):
            plt.plot(epochs, lossPerIter, 'b-')

        rateIndicator += 1

    fig.savefig("_1_1.png")

    print("\nThe accuracies per learning rate:" , \
        "Rate:", possibleRates[0], ":", "Acc:", trainAccuracies[0], "|", \
        "Rate:", possibleRates[1], ":", "Acc:", trainAccuracies[1], "|", \
        "Rate:", possibleRates[2], ":", "Acc:", trainAccuracies[2])


    print("\nThe losses per learning rate:" , \
        "Rate:", possibleRates[0], ":", "Loss:", rateLosses[0], "|", \
        "Rate:", possibleRates[1], ":", "Loss:", rateLosses[1], "|", \
        "Rate:", possibleRates[2], ":", "Loss:", rateLosses[2])

    optRate = possibleRates[np.argmin(rateLosses)]
    print("\nOptimal rate:", optRate)

    #-----------------PART 1.2--------------------------------------------------
    print("\n\n\n\n-----PART 1.2-----\n\n")
    batchLosses = [0.,0.,0.]
    batchTime = [0., 0., 0.]
    batchSizeIndicator = 0

    for sizeB in possibleB:
        W = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float32))
        b = tf.Variable(tf.truncated_normal([1], stddev = 0.1, dtype = tf.float32))
        sess.run(tf.global_variables_initializer())
        MSELoss = calc_MSE(Y, X, W, b, possibleWdc[0])

        np.random.shuffle(indices)
        shuffledTrainingData = trainData[indices]
        shuffledTrainingTarget = trainTarget[indices]

        start = timeit.timeit()

        optimizer = tf.train.GradientDescentOptimizer(optRate)
        train = optimizer.minimize(MSELoss)

        for i in range(iterations):
            batchIndicator = (i % numB[batchSizeIndicator]) * sizeB

            if not (i % numB[batchSizeIndicator]):
                np.random.shuffle(indices)
                shuffledTrainingData = trainData[indices]
                shuffledTrainingTarget = trainTarget[indices]

            currData, currTarget = shuffledTrainingData[batchIndicator:batchIndicator + sizeB], \
                                shuffledTrainingTarget[batchIndicator:batchIndicator + sizeB]

            sess.run(train, feed_dict = {X: currData, Y: currTarget})

        end = timeit.timeit()
        batchLosses[batchSizeIndicator] = sess.run(MSELoss, feed_dict = {X: currData, Y: currTarget}) / 2
        batchTime[batchSizeIndicator] = abs(end - start)

        batchSizeIndicator += 1

    print("The losses per batch size:" , \
        "Batch size", possibleB[0], ":", "Loss:", batchLosses[0], "|", \
        "Batch size", possibleB[1], ":", "Loss:", batchLosses[1], "|", \
        "Batch size", possibleB[2], ":", "Loss:", batchLosses[2])

    print("\nBatch times:", \
        "Batch size", possibleB[0], ":", "Time:", batchTime[0], "|", \
        "Batch size", possibleB[1], ":", "Time:", batchTime[1], "|", \
        "Batch size", possibleB[2], ":", "Time:", batchTime[2])

    origTime = batchTime[0]
    batchTime = [batchTime[0] / batchTime[0], \
                    batchTime[1] / batchTime[0], \
                    batchTime[2] / batchTime[0]]
    print("\nBatch times as ratio B = 500:", \
        "Batch size", possibleB[0], ":", "Time:", batchTime[0], "|", \
        "Batch size", possibleB[1], ":", "Time:", batchTime[1], "|", \
        "Batch size", possibleB[2], ":", "Time:", batchTime[2])
    #-----------------PART 1.3--------------------------------------------------
    print("\n\n\n\n-----PART 1.3-----\n\n")
    #variables specigic to PART 1.3:
    wdcLosses = [0.,0.,0.,0.]
    validAccuracies= [0.,0.,0.,0.]

    wdcIndicator = 0
    weightList = []
    biasList = []

    for wdc in possibleWdc:
        W = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float32))
        b = tf.Variable(tf.truncated_normal([1], stddev = 0.5, dtype = tf.float32))
        sess.run(tf.global_variables_initializer())

        MSELoss = calc_MSE(Y, X, W, b, wdc)

        np.random.shuffle(indices)
        shuffledTrainingData = trainData[indices]
        shuffledTrainingTarget = trainTarget[indices]

        optimizer = tf.train.GradientDescentOptimizer(possibleRates[0])
        train = optimizer.minimize(MSELoss)

        for i in range(iterations):
            batchIndicator = (i % numB[0]) * possibleB[0]

            if not (i % numB[0]):
                np.random.shuffle(indices)
                shuffledTrainingData = trainData[indices]
                shuffledTrainingTarget = trainTarget[indices]

            currData, currTarget = shuffledTrainingData[batchIndicator:batchIndicator + possibleB[0]], \
                                shuffledTrainingTarget[batchIndicator:batchIndicator + possibleB[0]]

            sess.run(train, feed_dict = {X: currData, Y: currTarget})

        weightList.append(W)
        biasList.append(b)
        wdcLosses[wdcIndicator] = sess.run(MSELoss, feed_dict = {X: currData, Y: currTarget}) / 2

        #since dealing with class betwen 0 and 1, will use 0.5 as threshold
        hypothesis = (tf.matmul(X,W) + b)
        classified = tf.to_float(tf.greater(hypothesis, 0.5))
        numCorrect = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(validTarget), classified)))
        validAccuracy = sess.run(numCorrect / validTarget.shape[0], feed_dict = {X: validData}) * 100
        validAccuracies[wdcIndicator] = validAccuracy

        wdcIndicator += 1

    optWDC = np.argmax(validAccuracies)
    hypothesis = (tf.matmul(X, weightList[optWDC]) + biasList[optWDC])
    classified = tf.to_float(tf.greater(hypothesis, 0.5))
    numCorrect = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(testTarget), classified)))
    testAccuracy = sess.run(numCorrect / testTarget.shape[0], feed_dict = {X: testData}) * 100

    print("The losses per wdc size:" , \
        "WDC", possibleWdc[0], ":", "Loss:", wdcLosses[0], ":", "VA", validAccuracies[0], "%", "|", \
        "WDC", possibleWdc[1], ":", "Loss:", wdcLosses[1], ":", "VA", validAccuracies[1], "%" "|", \
        "WDC", possibleWdc[2], ":", "Loss:", wdcLosses[2], ":", "VA", validAccuracies[2], "%" "|", \
        "WDC", possibleWdc[3], ":", "Loss:", wdcLosses[3], ":", "VA", validAccuracies[3], "%" )

    print("The test accuracy for WDC", possibleWdc[optWDC], "is: ", testAccuracy)

    #-----------------PART 1.4--------------------------------------------------
    print("\n\n\n\n-----PART 1.4-----\n\n")

    #normal equation w* = (X^T * X + wdcI)^-1 X^T * Y
    wdc = 0
    b = tf.Variable(tf.random_normal([PIXELCOUNT]))
    sess.run(tf.global_variables_initializer())
    trainData = np.insert(trainData, 0, np.ones(PIXELCOUNT), 0)
    trainTarget = np.insert(trainTarget, 0, 1, 0)
    start = timeit.timeit()

    WStar = tf.matmul(
                tf.matrix_inverse(
                tf.matmul(tf.transpose(X), X)),
                tf.matmul(tf.transpose(X), Y))

    # WStar = tf.matrix_solve_ls(X, Y, wdc, fast=True)

    end = timeit.timeit()

    MSELoss = calc_MSE(Y, X, WStar, b, wdc)

    finalMSE = sess.run(MSELoss, feed_dict = {X: trainData, Y: trainTarget}) / 2
    time = abs(end-start) / origTime

    newX = tf.placeholder(tf.float32, name="data")

    hypothesis = (tf.matmul(newX, WStar))
    classified = tf.to_float(tf.greater(hypothesis, 0.5))
    numCorrect = tf.reduce_sum(tf.to_float(tf.equal(tf.to_float(trainTarget), classified)))
    trainAccuracy = sess.run(numCorrect / trainTarget.shape[0], feed_dict = {newX:trainData, X: trainData, Y: trainTarget}) * 100

    print("Final Loss:", finalMSE, "|", \
            "Train Accuracy: ", trainAccuracy, "|", \
            "Computation Time:", time)

    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')
    linear_regression()
