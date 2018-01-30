#______ECE521 Assignment 1______
#______STARTDATE: Jan. 20, 2018______
#______DUE: Feb. 2, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf
# import matplotlib.pylab as plt

#----------Question: 1---------------------------------------------------------

def eucl_dist(X,Z):
    XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
    ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
    #for both...axis2 = D. for axis0 and axis2, there is a corresponding size 1.
    #makes them compatible for broadcasting

    #return the reduced sum accross axis 1. This will sum accros the D dimensional
    #element thus returning the N1xN2 matrix we desire
    return tf.reduce_sum((XExpanded-ZExpanded)**2, 1)

sess = tf.Session()
# x = tf.constant([[1,2,1,2,2],[3,4,1,2,2]])
# z = tf.constant([[11,22,1,23,32],[13,14,1,22,12],[2,3,4,5,6]])
# print(sess.run(eucl_dist(x,z)))

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
    #broadcasting will take care of dimensions and then we reduce accros the 1
    #axis. Thus our N2x1x1 matrix compares to our 1xN1xK matrix and we get back
    #an N2xN1 matrix when we reduce accross the 1 axis, or K. This matrix returns
    #as True/False values, so we convert those to floating numbers. Then take
    #the transpose to get N1xN2
    kNearest = tf.transpose(tf.reduce_sum(tf.to_float(tf.equal(neighboursIndices, possibleIndices)),2))

    #since we now have a N1xN2 matric with either 0 or 1 as elemental values,
    #we return that matrix divided by K since responsibiliies will have a 1/k
    #weight to them. since kNearest is a N1xN2 matrix, we just return the
    #proper row defined by testPointIndex
    if testPointIndex == None:
        return (kNearest/tf.to_float(K))
    else:
        return (kNearest/tf.to_float(K))[testPointIndex]

def KNN_prediction(targetY, responsibilities):
    #y(x) = r*Y
    return tf.matmul(responsibilities, targetY)

def MSE_loss(targetY, predictedY):
    #as discussed in tut
    return tf.reduce_mean(tf.reduce_mean(tf.square(predictedY - targetY),1))

def run_KNN(trainData, trainTarget, sampleData, sampleTarget, K):
    #compute the euclidean distance between test and training
    testDistance = eucl_dist(sampleData, trainData)

    #get the responsibiliies matrix
    testResponsibilities = training_responsibilities(testDistance, K)

    #compute the KNN prediction
    testPrediction = KNN_prediction(trainTarget, testResponsibilities)

    #compute the MSE
    testMSE = MSE_loss(sampleTarget, testPrediction)

    return testMSE

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

    #define possible Ks
    possibleK = [1,3,5,50]

    #compute the MSE loss for each possible k
    trainingError = []
    validationError = []
    testError = []

    for currK in possibleK:
        trainingErrorTemp = sess.run(run_KNN(trainX, trainY, trainX, trainY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, K:currK})
        trainingError.append(trainingErrorTemp)

        validationErrorTemp = sess.run(run_KNN(trainX, trainY, newX, newY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, newX:validData, \
            newY:validTarget, K:currK})
        validationError.append(validationErrorTemp)

        testErrorTemp = sess.run(run_KNN(trainX, trainY, newX, newY, K), \
            feed_dict={trainX:trainData, trainY:trainTarget, newX:testData, \
            newY:testTarget, K:currK})
        testError.append(testErrorTemp)

        print("\n\nwith K = %d, the training MSE loss is %f, "
        "validation MSE loss is %f, and test MSE loss is %f." % (currK, \
        trainingErrorTemp, validationErrorTemp, testErrorTemp))

    #get the index of the minimum validator error and use that to get the
    #corresponding best K
    bestK = possibleK[validationError.index(min(validationError))]
    print('\n\nBest K: ', bestK, '\n\n')

    # plotPrediction, plotMSE = sess.run(run_KNN(trainX, \
    #     trainY, newX, newY, K), feed_dict={trainX:trainData, trainY:trainTarget, \
    #     newX:X, newY:validTarget, K:currK})
    #
    # plt.figure(currK + 1)
    # plot.plot(trainData, trainTarget, '.')
    # plt.plot(X, plotPrediction, '-')
    # plt.title("KNN regression on data1D, where K = %d"%currK)
    # plot.show()

    return


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

def classification_prediction(trainData, trainTarget, sampleData, sampleTarget, K):
    #compute the euclidean distance between test and training
    testDistance = eucl_dist(sampleData, trainData)

    #defined to hold the majority class from each row of the image
    majority = []

    #for each row, find the k nearest neighbours, and determine which class
    #this row most seems to resemble. aggregate the results for each row and
    #at the very end, classify based on the majority class that showed up
    #in the aggregate result
    for i in range(-1):
        #iteratively grab a new row
        currentRow = tf.gather(testDistance, i)

        #find the k nearest neighbours (and their indices) in each row, where a row
        #represents a different data point. Take the -ve of the matrix since the
        #smaller the value the closer it is, bigger +ve numbers become smaller -ve
        #numbers
        neighbours, neighboursIndices = tf.nn.top_k(-currentRow,k = K)

        #from nereast neighbours, gather the corresponding rows
        possibleClassificationsVec = tf.gather(trainTarget, neighboursIndices)

        #run unique_with_counts to find majority classification
        y, idx, count = tf.unique_with_counts(possibleClassificationsVec)

        #append the majority class from the current row
        majority.append(count)

    #return maajority which has predictions of each image per row
    return majority

def classify():
    #define our placeholders
    K = tf.placeholder(tf.int32, name = "K")
    trainX = tf.placeholder(tf.float32, name = "trainX")
    trainY = tf.placeholder(tf.float32, name = "trainY")
    newX = tf.placeholder(tf.float32, name = "newX")
    newY = tf.placeholder(tf.float32, name = "newY")

    #define possible Ks
    possibleK = [1,5,10,15,25,50,100,200]

    #load data
    #run for name ID. task = 0
    trainData0, validData0, testData0, trainTarget0, validTarget0, testTarget0 = \
        data_segmentation("./data.npy", "./target.npy", 0)

    # print(sess.run(tf.shape(trainTarget)),sess.run(tf.shape(validTarget)),sess.run(tf.shape(testTarget)))
    #run for gender ID. task = 1
    trainData1, validData1, testData1, trainTarget1, validTarget1, testTarget1 = \
        data_segmentation("./data.npy", "./target.npy", 1)

    #compute the MSE loss for each possible k
    trainingError = []
    validationError = []
    testError = []

    for currK in range(1):
        classifications = []
        classifications.append(sess.run(classification_prediction(trainX, \
        trainY, newX, newY, K), feed_dict={trainX:trainData0, trainY:trainTarget0, \
        newX:validData0, newY:validTarget0, K:currK}))

        print('\n\nclass: ', classifications)

    return

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n\n\n---------Assignment 1: KNN Regression---------\n\n')

    #part 1
    solve_KNN()

    #part 3
    classify()
