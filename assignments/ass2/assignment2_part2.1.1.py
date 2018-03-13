from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()
PIXELCOUNT = 28*28

# Load notMNIST data
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

# Note: PIXELCOUNT is 28x28
def logisticRegression():

    # Variable definitions
    batchSize = 500
    numIterations = 5000
    wdc = 0.01
    learningRate = 0.005

    # Load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()

    trainData = trainData.reshape(trainData.shape[0], PIXELCOUNT)
    validData = validData.reshape(validData.shape[0], PIXELCOUNT)
    testData = testData.reshape(testData.shape[0], PIXELCOUNT)

    # Extract number of samples for each type of data
    numTrainingSamples = trainData.shape[0]
    numValidationSamples = validData.shape[0]
    numTestSamples = testData.shape[0]

    # i.e numBatches = 3500/500 = 7
    numBatches = numTrainingSamples // batchSize
    numEpochs = numIterations // numBatches

    # Placeholders
    trainX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "trainData")
    trainY = tf.placeholder(tf.float64, shape = [None, 1], name = "trainTarget")
    
    validationX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "validationData")
    validationY = tf.placeholder(tf.float64, shape = [None, 1], name = "validationTarget")

    testX = tf.placeholder(tf.float64, name="testData")
    testY = tf.placeholder(tf.float64, name = "testTarget")
    
    b = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biases"))
    W = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weights"))

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Training data prediction and loss
    trainYHat = prediction(trainX, W, b)
    trainingLoss = calculateCrossEntropyLoss(trainY, trainYHat, W, wdc)

    # Setup gradient descent optimizer and oprimize the training loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningRate)
    train = optimizer.minimize(trainingLoss)

    # At this point we have our weight vector W
    # Use W to calculate validation prediction and loss 
    # This data is useful for tuning our learning rate
    validationYHat = prediction(validationX, W, b)
    validationLoss = calculateCrossEntropyLoss(validationY, validationYHat, W, wdc)

    # Calculate classification accuracy
    # Use a threshold of 0.5, values above are classified as 1
    # Values below are classified as 0
    outputVector = tf.sigmoid(tf.matmul(testX, W) + b)
    booleanClassificationVector = tf.greater(outputVector, 0.5)
    integerClassificationVector = tf.cast(booleanClassificationVector, tf.float64)
    correctClassificationVector = tf.cast(tf.equal(integerClassificationVector, testY), tf.float64)
    numCorrectClassified = tf.reduce_sum(correctClassificationVector)
    classificationAccuracy = ( tf.cast(numCorrectClassified, tf.float64) / tf.cast(tf.shape(correctClassificationVector)[0], tf.float64)) * 100
    
    learningRateArray = [0.005, 0.001, 0.0001]
    learningRateErrorsArray = []

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
    for rate in learningRateArray:
        learningRate = rate

        # Array variables for training
        tempTrainingLossArray = []
        tempValidationLossArray = []

        trainingLossArray = []
        validationLossArray = []
        
        trainingAccuracyArray = []
        validationAccuraryArray = []
        testAccuracyArray = []

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

        learningRateErrorsArray.append(sum(tempValidationLossArray))

        trainingLossPerLearningRatePerEpoch.append(trainingLossArray)
        validationLossPerLearningRatePerEpoch.append(validationLossArray)

        trainingAccuracyPerLearningRatePerEpoch.append(trainingAccuracyArray)
        validationAccuraryPerLearningRatePerEpoch.append(validationAccuraryArray)
        testAccuracyPerLearningRatePerEpoch.append(testAccuracyArray)
    # Out of for loop

    # Calculate best learning rate based on validation errors
    bestLearningRateIndex = learningRateErrorsArray.index(min(learningRateErrorsArray))
    bestLearningRate = learningRateArray[bestLearningRateIndex]

    # Using the best learning rate, find the arrays of best
    # training loss, validation loss, training accuracy, and validation accuracy
    bestTrainingLossPerEpoch = trainingLossPerLearningRatePerEpoch[bestLearningRateIndex]
    bestValidationLossPerEpoch = validationLossPerLearningRatePerEpoch[bestLearningRateIndex]
    bestTrainingAccuracyPerEpoch = trainingAccuracyPerLearningRatePerEpoch[bestLearningRateIndex]
    bestValidationAccuracyPerEpoch =validationAccuraryPerLearningRatePerEpoch[bestLearningRateIndex]

    epochs = np.linspace(0, numEpochs, num = numEpochs)

    # Plot loss vs number of epochs
    figure = plt.figure()
    axes = plt.gca()
    # axes.set_ylim([0,0.5])
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

    # Calculate best test accuracy obtained
    tempTestAccuracyMax = [max(arr) for arr in testAccuracyPerLearningRatePerEpoch]
    bestTestAccuracy = max(tempTestAccuracyMax)

    
if __name__ == '__main__':
    print('\n\n\n---------Assignment 2.1.1---------\n\n')
    logisticRegression()