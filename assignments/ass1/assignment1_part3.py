from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import assignment1_part1 as p1
import assignment1_part2 as p2

sess = tf.Session()

#----------Question: 3---------------------------------------------------------

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)

    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
                                    data[rnd_idx[trBatch + validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                            target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
                                            target[rnd_idx[trBatch + validBatch + 1:-1], task]

    return trainData, validData, testData, trainTarget, validTarget, testTarget

def find_neighbours_matrix(trainData, sampleData, K):
    #compute the euclidean distance between test and training
    testDistance = p1.eucl_dist(sampleData, trainData)

    #find the k nearest neighbours (and their indices) in each row, where a row
    #represents a different data point. Take the -ve of the matrix since the
    #smaller the value the closer it is, bigger +ve numbers become smaller -ve
    #numbers
    neighbours, neighboursIndices = tf.nn.top_k(-testDistance,k = K)

    return neighboursIndices

def classification_prediction(trainTarget, sampleTarget, trainData, sampleData, K, tenIdentifier, neighboursIndices):
    #for each row, find the k nearest neighbours, and determine which class
    #this row most seems to resemble. aggregate the results for each row and
    #at the very end, classify based on the majority class that showed up
    #in the aggregate result

    #defined to hold the majority class from each row of the image
    allMajorities = []

    for i in range(neighboursIndices.shape[0]):
        #iteratively grab a new row
        currentRow = tf.gather(neighboursIndices, i)

        #from nearest neighbours target vector, gather the corresponding rows
        possibleClassificationsVec = tf.gather(trainTarget, currentRow)

        #run unique_with_counts to find majority classification
        classes, idx, count = tf.unique_with_counts(possibleClassificationsVec)

        #find the majority class
        majorityClass = tf.gather(classes, tf.argmax(count))

        #append the majority class from the current row
        allMajorities.append(majorityClass)

    # converting from column vector into row vector i.e stacking up all rows into one row
    allMajorities = tf.stack(allMajorities)

    #tenIdentifier = 1 when we want to do the last part of 3.2 display failed
    #image
    if(tenIdentifier == 1):
        #stores a vector where 0 means the image was incorrectly identified
        incorrectIndices = tf.to_float(tf.equal(allMajorities, sampleTarget))

        #we only need one image, so taking argmax will return the first index
        #that was incorrectly classified. Take -ve since want 0 as identifier
        wrongIndex = tf.argmax(-incorrectIndices)

        #also, return the wrong nearest neighbours
        wrongKNN = tf.gather(trainData, neighboursIndices)

        #to avoid dealing with tensors, return the index to be dealt outside
        #the session.
        return wrongIndex

    # find number of unmatching predictions and divide by total number of predictions
    accuracy = tf.reduce_sum(tf.to_float(tf.equal(allMajorities, sampleTarget)))/neighboursIndices.shape[0]

    #return maajority which has predictions of each image per row
    return accuracy*100


def classify(classifyParam):
    #define our placeholders
    K = tf.placeholder(tf.int32, name = "K")
    trainX = tf.placeholder(tf.float32, name = "trainX")
    trainY = tf.placeholder(tf.float32, name = "trainY")
    newX = tf.placeholder(tf.float32, name = "newX")
    newY = tf.placeholder(tf.float32, name = "newY")

    #define possible Ks
    possibleK = [1,5,10,25,50,100,200]

    if classifyParam == 0:
        #load data for name ID
        trainData, validData, testData, trainTarget, validTarget, testTarget = \
            data_segmentation("./data.npy", "./target.npy", 0)
    elif classifyParam == 1:
        #run for gender ID. task = 1
        trainData, validData, testData, trainTarget, validTarget, testTarget = \
            data_segmentation("./data.npy", "./target.npy", 1)

    # print(sess.run(tf.shape(trainData0)),sess.run(tf.shape(trainTarget0)),sess.run(tf.shape(validTarget0)))

    #compute the validation accuracy and test accuracy for each possible k
    validationAccuracy = []
    testAccuracy = []

    for currK in possibleK:

        #validation data
        # return a numpy matrix of closest neighbours indices.
        neighboursIndices = (sess.run(find_neighbours_matrix(trainX, \
        newX, K), feed_dict={trainX:trainData, newX:validData, K:currK}))

        # use this closest neighbours indices to return a predicted classification vector
        validationAccuracyTemp = sess.run(classification_prediction(trainY, newY, trainX, newX, K, 0, neighboursIndices),\
        feed_dict={trainY:trainTarget, newY:validTarget, trainX: trainData, newX: validData, K:currK})
        validationAccuracy.append(validationAccuracyTemp)

        #test data
        # return a numpy matrix of closest neighbours indices.
        neighboursIndices = (sess.run(find_neighbours_matrix(trainX, \
        newX, K), feed_dict={trainX:trainData, newX:testData, K:currK}))

        # use this closest neighbours indices to return a predicted classification vector
        testAccuracyTemp = sess.run(classification_prediction(trainY, newY, trainX, newX, K, 0, neighboursIndices),\
        feed_dict={trainY:trainTarget, newY:testTarget, trainX: trainData, newX: testData, K:currK})
        testAccuracy.append(testAccuracyTemp)

        print("\nwith K = %d, the validation accuracy is %f %% and the"\
        " test accuracy is %f %%" % (currK, validationAccuracyTemp, testAccuracyTemp))

    bestK = possibleK[validationAccuracy.index(max(validationAccuracy))]

    print('\nBest K: ', bestK)

    # use the bestK to find test accuracy
    # return a numpy matrix of closest neighbours indices
    neighboursIndices = (sess.run(find_neighbours_matrix(trainX, \
    newX, K), feed_dict={trainX:trainData, newX:testData, K:bestK}))

    # use this closest neighbours indices to return a predicted classification vector
    testAccuracyTemp = sess.run(classification_prediction(trainY, newY, trainX, newX, K, 0, neighboursIndices),\
    feed_dict={trainY:trainTarget, newY:testTarget, trainX: trainData, newX: testData, K:bestK})

    print("\nWith the best K = %d, the test accuracy is %f %%" % (bestK, testAccuracyTemp))

    #for k = 10, display failure case
    # return a numpy matrix of closest neighbours indices
    neighboursIndices = (sess.run(find_neighbours_matrix(trainX, \
    newX, K), feed_dict={trainX:trainData, newX:testData, K:10}))

    # use this closest neighbours indices to return a predicted classification vector
    wrongIndex = sess.run(classification_prediction(trainY, newY, trainX, newX, K, 1, neighboursIndices),\
    feed_dict={trainY:trainTarget, newY:testTarget, trainX: trainData, newX: testData, K:10})

    reshaped = testData[wrongIndex].reshape(32,32)

    plt.figure(11)
    plt.imshow(reshaped,cmap='gray')

    wrongKNN = sess.run(tf.gather(trainData, neighboursIndices[wrongIndex]))

    # print(wrongKNN.shape)
    for i in range(wrongKNN.shape[0]):
        reshaped = wrongKNN[i].reshape(32,32)
        # print("i: ",i," name/gender: ", trainTarget[neighboursIndices[wrongIndex][i]])
        plt.figure(i)
        plt.imshow(reshaped, cmap='gray')

    plt.show()

    return
