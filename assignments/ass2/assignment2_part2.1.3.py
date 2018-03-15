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

def calculateMSE(y, yHat):
    return tf.reduce_mean(tf.square(yHat - y)) / 2

def prediction(X, W, b):
    return tf.matmul(X, W) + b

# Note: PIXELCOUNT is 28x28
def logisticRegression():

    # Variable definitions
    batchSize = 500
    numIterations = 5000
    wdc = 0.0
    learningRate = 0.001

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
    dataX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "trainData")
    targetY = tf.placeholder(tf.float64, shape = [None, 1], name = "trainTarget")
    
    bLinearRegression = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biasesLinearRegression"))
    WLinearRegression = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weightsLinearRegression"))

    bLogisticRegression = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biasesLogisticRegression"))
    WLogisticRegression = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weightsLogisticRegression"))

    # Linear Regression: Prediction and MSE
    yHatLinearRegression = prediction(dataX, WLinearRegression, bLinearRegression)
    MSE = calculateMSE(targetY, yHatLinearRegression)

    # Linear Regression: Optimization of weight vector
    optimizerAdamLinearRegression = tf.train.AdamOptimizer(learning_rate = learningRate)
    trainAdamLinearRegression = optimizerAdamLinearRegression.minimize(MSE)

    # Logistic Regression: Prediction and Cross Entropy Error
    yHatLogisticRegression = prediction(dataX, WLogisticRegression, bLogisticRegression)
    crossEntropyLoss = calculateCrossEntropyLoss(targetY, yHatLogisticRegression, WLogisticRegression, wdc)

    # Linear Regression: Optimization of weight vector
    optimizerAdamLogisticRegression = tf.train.AdamOptimizer(learning_rate = learningRate)
    trainAdamLogisticRegression = optimizerAdamLogisticRegression.minimize(crossEntropyLoss)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Calculate classification accuracy for LINEAR REGRESSION
    # Use a threshold of 0.5
    outputVectorLinearRegression = tf.matmul(dataX, WLinearRegression) + bLinearRegression
    classificationVectorLinearRegression = tf.cast(tf.greater(yHatLinearRegression, 0.5), tf.float64)
    correctClassificationVectorLinearRegression = tf.cast(tf.equal(classificationVectorLinearRegression, targetY), tf.float64)
    numCorrectClassifiedLinearRegression = tf.reduce_sum(correctClassificationVectorLinearRegression)
    classificationAccuracyLinearRegression = ( tf.cast(numCorrectClassifiedLinearRegression, tf.float64) / tf.cast(tf.shape(correctClassificationVectorLinearRegression)[0], tf.float64)) * 100
    
    # Calculate classification accuracy for LINEAR REGRESSION
    # Use a threshold of 0.5
    outputVectorLogisticRegression = tf.matmul(dataX, WLogisticRegression) + bLogisticRegression
    classificationVectorLogisticRegression = tf.cast(tf.greater(yHatLogisticRegression, 0.5), tf.float64)
    correctClassificationVectorLogisticRegression = tf.cast(tf.equal(classificationVectorLogisticRegression, targetY), tf.float64)
    numCorrectClassifiedLogisticRegression = tf.reduce_sum(correctClassificationVectorLogisticRegression)
    classificationAccuracyLogisticRegression = ( tf.cast(numCorrectClassifiedLogisticRegression, tf.float64) / tf.cast(tf.shape(correctClassificationVectorLogisticRegression)[0], tf.float64)) * 100

    # Array variables 
    trainingLossPerEpochLinearRegression = []
    trainingLossPerEpochLogisticRegression = []

    trainingAccuracyPerEpochLinearRegression = []
    validationAccuraryPerEpochLinearRegression = []
    testAccuracyPerEpochLinearRegression = []

    trainingAccuracyPerEpochLogisticRegression = []
    validationAccuraryPerEpochLogisticRegression = []
    testAccuracyPerEpochLogisticRegression = []

    # Create an array of indices from 0 to 3499
    # Useful for random selection of incides for a batch
    indices = np.arange(0, numTrainingSamples)
    shuffledTrainingData = []
    shuffledTrainingTarget = []

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

        sess.run(trainAdamLinearRegression, feed_dict = {dataX: batchData, targetY: batchTarget})
        sess.run(trainAdamLogisticRegression, feed_dict = {dataX: batchData, targetY: batchTarget})

        tempTrainingLossLinearRegression = sess.run(MSE, feed_dict={dataX: batchData, targetY: batchTarget})
        tempTrainingLossLogisticRegression = sess.run(crossEntropyLoss, feed_dict={dataX: batchData, targetY: batchTarget})

        tempTrainingAccuracyLinearRegression = sess.run(classificationAccuracyLinearRegression, feed_dict={dataX: batchData, targetY: batchTarget})
        tempValidationAccuracyLinearRegression = sess.run(classificationAccuracyLinearRegression, feed_dict={dataX: validData, targetY: validTarget})
        tempTestAccuracyLinearRegression = sess.run(classificationAccuracyLinearRegression, feed_dict={dataX: testData, targetY: testTarget})

        tempTrainingAccuracyLogisticRegression = sess.run(classificationAccuracyLogisticRegression, feed_dict={dataX: batchData, targetY: batchTarget})
        tempValidationAccuracyLogisticRegression = sess.run(classificationAccuracyLogisticRegression, feed_dict={dataX: validData, targetY: validTarget})
        tempTestAccuracyLogisticRegression = sess.run(classificationAccuracyLogisticRegression, feed_dict={dataX: testData, targetY: testTarget})

        # i.e at every epoch find the training, validation, and classification accuracy
        if ((i+1) % numBatches) == 0:
            trainingLossPerEpochLinearRegression.append(tempTrainingLossLinearRegression)
            trainingLossPerEpochLogisticRegression.append(tempTrainingLossLogisticRegression)

            trainingAccuracyPerEpochLinearRegression.append(tempTrainingAccuracyLinearRegression)
            validationAccuraryPerEpochLinearRegression.append(tempValidationAccuracyLinearRegression)
            testAccuracyPerEpochLinearRegression.append(tempTestAccuracyLinearRegression)

            trainingAccuracyPerEpochLogisticRegression.append(tempTrainingAccuracyLogisticRegression)
            validationAccuraryPerEpochLogisticRegression.append(tempValidationAccuracyLogisticRegression)
            testAccuracyPerEpochLogisticRegression.append(tempTestAccuracyLogisticRegression)

    print(max(trainingAccuracyPerEpochLinearRegression), max(validationAccuraryPerEpochLinearRegression), max(testAccuracyPerEpochLinearRegression))
    print(max(trainingAccuracyPerEpochLogisticRegression), max(validationAccuraryPerEpochLogisticRegression), max(testAccuracyPerEpochLogisticRegression))

    epochs = np.linspace(0, numEpochs, num = numEpochs)

    # DON'T HAVE TO PLOT THIS BUT HAVE IT JUST IN CASE
    # Linear Regression: Training, Validation and Test Accuracy vs Number of Epochs
    # figure = plt.figure()
    # axes = plt.gca()
    # plt.plot(epochs, trainingAccuracyPerEpochLinearRegression, "b-", label = 'Training Acc')
    # plt.plot(epochs, validationAccuraryPerEpochLinearRegression, "r-", label = 'Validation Acc')
    # plt.plot(epochs, testAccuracyPerEpochLinearRegression, "g-", label = 'Test Acc')
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best', shadow = True, fancybox = True)
    # plt.title("Linear Regression: Training, Validation and Test Accuracy vs Number of Epochs")    
    # plt.show()

    # Training Loss for Linear/Logistic Regression vs Num of Epochs
    figure = plt.figure()
    axes = plt.gca()
    plt.plot(epochs, trainingLossPerEpochLinearRegression, "b-", label = 'Linear Regression Loss')
    plt.plot(epochs, trainingLossPerEpochLogisticRegression, "r-", label = 'Logistic Regression Loss')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.title("Training Loss for Linear/Logistic Regression vs Num of Epochs")    
    plt.show()

    # Training Accuracy for Linear/Logistic Regression vs Num of Epochs
    figure = plt.figure()
    axes = plt.gca()
    plt.plot(epochs, trainingAccuracyPerEpochLinearRegression, "b-", label = 'Linear Regression Accuracy')
    plt.plot(epochs, trainingAccuracyPerEpochLogisticRegression, "r-", label = 'Logistic Regression Accuracy')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.title("Training Accuracy for Linear/Logistic Regression vs Num of Epochs")    
    plt.show()

    # Plotting dummy target = 0 for 100 points in prediction interval [0, 1]
    dummyTargetY = tf.zeros(100, tf.float64)
    dummyYHat = np.linspace(0.0, 1.0, num = 100)

    dummyMSEPointWise = tf.square(dummyYHat - dummyTargetY)
    dummyCrossEntropyErrorPointWise = tf.nn.sigmoid_cross_entropy_with_logits(logits = dummyTargetY, labels = dummyYHat)

    MSE = sess.run(dummyMSEPointWise)
    CE = sess.run(dummyCrossEntropyErrorPointWise)

    figure = plt.figure()
    axes = plt.gca()
    plt.plot(dummyYHat, MSE, "r-", label= "Mean-Squared Loss")
    plt.plot(dummyYHat, CE, "b-", label= "Cross Entropy Loss")
    plt.xlabel("Prediction yhat")
    plt.ylabel("Loss")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.show()

    
if __name__ == '__main__':
    print('\n\n\n---------Assignment 2.1.1---------\n\n')
    logisticRegression()