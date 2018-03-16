from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()
PIXELCOUNT = 32*32

def load_data():
    data = np.load("./data.npy")/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load("./target.npy")

    np.random.seed(521)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)

    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))

    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
                                    data[rnd_idx[trBatch + validBatch+1:-1],:]

    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], 0], \
                                            target[rnd_idx[trBatch+1:trBatch + validBatch], 0],\
                                            target[rnd_idx[trBatch + validBatch + 1:-1], 0]

    return trainData, trainTarget, validData, validTarget, testData, testTarget


def calculateCrossEntropyLoss(y, yHat, W, wdc, num_classes):
    ''' y is the target,
        yHat is the output prediction,
        lambda (hyperparameter) is the weight decay coefficient

        Cross Entropy Loss = Ld + Lw
    ''' 

    Ld = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(indices = tf.cast(y, tf.int32), depth = num_classes, on_value = 1, off_value = 0, axis = -1), logits = yHat))
    Lw = (wdc / 2) * tf.reduce_sum(tf.square(W))
    crossEntropyLoss = Lw + Ld
    return crossEntropyLoss

def prediction(X, W, b):
    return tf.matmul(X, W) + b

# Note: PIXELCOUNT is 28x28
def logisticRegression():

    # Variable definitions
    batchSize = 300
    numIterations = 5000
    learningRate = 0.005

    # Load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()
    num_classes = len(np.unique(trainTarget))

    # trainData = trainData.reshape(trainData.shape[0], PIXELCOUNT)
    # validData = validData.reshape(validData.shape[0], PIXELCOUNT)
    # testData = testData.reshape(testData.shape[0], PIXELCOUNT)

    # Extract number of samples for each type of data
    numTrainingSamples = trainData.shape[0]
    numValidationSamples = validData.shape[0]
    numTestSamples = testData.shape[0]

    # i.e numBatches = 747//300 = 2
    numBatches = numTrainingSamples // batchSize
    numEpochs = numIterations // numBatches

    # Placeholders
    trainX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "trainData")
    trainY = tf.placeholder(tf.float64, shape = [None], name = "trainTarget")

    validationX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "validationData")
    validationY = tf.placeholder(tf.float64, shape = [None], name = "validationTarget")

    testX = tf.placeholder(tf.float64, name="testData")
    testY = tf.placeholder(tf.float64, name = "testTarget")
    
    learningRateArray = [0.005, 0.001, 0.0001]
    wdcArray = [0.05, 0.01, 0.001]
    combinations = list()
    for learningRate in learningRateArray:
        for wdc in wdcArray:
            combinations.append([learningRate,wdc])
    learningRate_wdc_ErrorsArray = []

    trainingLossPerLearningRatePerEpoch = []
    validationLossPerLearningRatePerEpoch = []

    trainingAccuracyPerLearningRatePerEpoch = []
    validationAccuraryPerLearningRatePerEpoch = []
    testAccuracyPerLearningRatePerEpoch = []

    # Create an array of indices from 0 to 3499
    # Useful for random selection of incides for a batch
    indices = np.arange(0, numTrainingSamples)
    shuffledTrainingData = []
    shuffledTrainingTarget = []

    ''' For each learning rate, calculate the training error over 
        5000 iterations and pick the learning rate with the minimum error
    '''
    for element in combinations:
        learningRate = element[0]
        wdc = element[1]
        print(learningRate, wdc)

        # Array variables for training
        tempTrainingLossArray = []
        tempValidationLossArray = []

        trainingLossArray = []
        validationLossArray = []
        
        trainingAccuracyArray = []
        validationAccuraryArray = []
        testAccuracyArray = []

        b = tf.Variable(tf.truncated_normal(shape = [num_classes], stddev = 0.1, dtype = tf.float64, name = "biases"))
        W = tf.Variable(tf.truncated_normal([PIXELCOUNT, num_classes], stddev = 0.5, dtype = tf.float64, name = "weights"))

        # Training data prediction and loss
        trainYHat = prediction(trainX, W, b)
        trainingLoss = calculateCrossEntropyLoss(trainY, trainYHat, W, wdc, num_classes)

        # Setup gradient descent optimizer and oprimize the training loss
        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
        train = optimizer.minimize(trainingLoss)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # At this point we have our weight vector W
        # Use W to calculate validation prediction and loss 
        # This data is useful for tuning our learning rate
        validationYHat = prediction(validationX, W, b)
        validationLoss = calculateCrossEntropyLoss(validationY, validationYHat, W, wdc, num_classes)

        # Calculate classification accuracy
        # Use a threshold of 0.5, values above are classified as 1
        # Values below are classified as 0
        outputVector = tf.nn.softmax(tf.matmul(testX, W) + b)
        ClassificationVector = tf.cast(tf.argmax(outputVector, 1), tf.float64)
        correctClassificationVector = tf.cast(tf.equal(ClassificationVector, testY), tf.float64)
        numCorrectClassified = tf.reduce_sum(correctClassificationVector)
        classificationAccuracy = ( tf.cast(numCorrectClassified, tf.float64) / tf.cast(tf.shape(correctClassificationVector)[0], tf.float64)) * 100

        for i in range(numIterations):

            ''' One iteration is an SGD update through
                the entire pass of one mini-batch
            '''

            # Shuffle indices once every numBatches iterations i.e once every 7 iterations
            if not (i % numBatches): 
                np.random.shuffle(indices)
                shuffledTrainingData = trainData[indices]
                shuffledTrainingTarget = trainTarget[indices]

            startBatchIndex = (i % numBatches) * batchSize
            endBatchIndex = startBatchIndex + batchSize

            # Obtain a batch of training data from start index to end index
            batchData = shuffledTrainingData[startBatchIndex : endBatchIndex]
            batchTarget = shuffledTrainingTarget[startBatchIndex : endBatchIndex]

            [tempTrain, tempTrainingLoss, tempValidationLoss] = sess.run([train, trainingLoss, validationLoss], feed_dict={trainX: batchData, trainY: batchTarget, validationX: validData , validationY: validTarget})
            # This is the error per iteration i.e per batch
            tempTrainingLossArray.append(tempTrainingLoss)
            tempValidationLossArray.append(tempValidationLoss)

            # i.e at every epoch find the training loss, validation loss, training and classification accuracy
            if ((i+1) % numBatches) == 0:
                trainingLossArray.append(tempTrainingLoss)
                validationLossArray.append(tempValidationLoss)

                tempTrainingAccuracy = sess.run(classificationAccuracy, feed_dict = {testX: trainData, testY: trainTarget})
                tempValidationAccuracy = sess.run(classificationAccuracy, feed_dict = {testX: validData, testY: validTarget})
                tempTestAccuracy = sess.run(classificationAccuracy, feed_dict = {testX: testData, testY: testTarget})
                
                trainingAccuracyArray.append(tempTrainingAccuracy)
                validationAccuraryArray.append(tempValidationAccuracy)
                testAccuracyArray.append(tempTestAccuracy)

        learningRate_wdc_ErrorsArray.append(min(tempValidationLossArray))

        trainingLossPerLearningRatePerEpoch.append(trainingLossArray)
        validationLossPerLearningRatePerEpoch.append(validationLossArray)

        trainingAccuracyPerLearningRatePerEpoch.append(trainingAccuracyArray)
        validationAccuraryPerLearningRatePerEpoch.append(validationAccuraryArray)
        testAccuracyPerLearningRatePerEpoch.append(testAccuracyArray)
    # Out of for loop

    # Calculate best learning rate based on validation errors
    bestLearningRateIndex = learningRate_wdc_ErrorsArray.index(min(learningRate_wdc_ErrorsArray))
    bestLearningRate_wdc = combinations[bestLearningRateIndex]

    # Using the best learning rate, find the arrays of best
    # training loss, validation loss, training accuracy, validation accuracy and test accuracy
    bestTrainingLossPerEpoch = trainingLossPerLearningRatePerEpoch[bestLearningRateIndex]
    bestValidationLossPerEpoch = validationLossPerLearningRatePerEpoch[bestLearningRateIndex]
    bestTrainingAccuracyPerEpoch = trainingAccuracyPerLearningRatePerEpoch[bestLearningRateIndex]
    bestValidationAccuracyPerEpoch = validationAccuraryPerLearningRatePerEpoch[bestLearningRateIndex]
    bestTestAccuracyPerEpoch = testAccuracyPerLearningRatePerEpoch[bestLearningRateIndex]

    epochs = np.linspace(0, numEpochs, num = numEpochs)

    # Plot loss vs number of epochs
    figure = plt.figure()
    axes = plt.gca()
    plt.plot(epochs, bestTrainingLossPerEpoch, "b-", label = 'Training Loss')
    plt.plot(epochs, bestValidationLossPerEpoch, "r-", label = 'Validation Loss')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.title("Best Training and Validation Loss vs Number of Epochs")    
    plt.show()

    # # Plot classification accuracy vs number of epochs
    figure = plt.figure()
    axes = plt.gca()
    plt.plot(epochs, bestTrainingAccuracyPerEpoch, "b-", label = 'Training Acc')
    plt.plot(epochs, bestValidationAccuracyPerEpoch, "r-", label = 'Validation Acc')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.title("Best Training and Validation Accuracy vs Number of Epochs")    
    plt.show()

    # Calculate best test accuracy obtained: 90.32
    bestTestAccuracy = max(bestTestAccuracyPerEpoch)
    print("Best [learning rate, wdc] is ", bestLearningRate_wdc, "and best test accuracy for this rate is", bestTestAccuracy)

    
if __name__ == '__main__':
    print('\n\n\n---------Assignment 2.1.1---------\n\n')
    logisticRegression()
