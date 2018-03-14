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
    learningRate = 0.001

    # Load and reshape data
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

    # Placeholders and variables
    trainX = tf.placeholder(tf.float64, shape = [None, PIXELCOUNT], name = "trainData")
    trainY = tf.placeholder(tf.float64, shape = [None, 1], name = "trainTarget")
    
    bSGD = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biasesSGD"))
    bAdam = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.1, dtype = tf.float64, name = "biasesAdam"))
    
    WSGD = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weightsSGD"))
    WAdam = tf.Variable(tf.truncated_normal([PIXELCOUNT, 1], stddev = 0.5, dtype = tf.float64, name = "weightsAdam"))

    # SGD: Training data prediction and loss
    trainYHatSGD = prediction(trainX, WSGD, bSGD)
    trainingLossSGD = calculateCrossEntropyLoss(trainY, trainYHatSGD, WSGD, wdc)

    # Setup gradient descent optimizer and oprimize the training loss
    optimizerSGD = tf.train.GradientDescentOptimizer(learning_rate = learningRate)
    trainSGD = optimizerSGD.minimize(trainingLossSGD)

    # Adam: Training data prediction and loss
    trainYHatAdam = prediction(trainX, WAdam, bAdam)
    trainingLossAdam = calculateCrossEntropyLoss(trainY, trainYHatAdam, WAdam, wdc)    

    # Setup adam optimizer and oprimize the training loss
    optimizerAdam = tf.train.AdamOptimizer(learning_rate = learningRate)
    trainAdam = optimizerAdam.minimize(trainingLossAdam)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create an array of indices from 0 to 3499
    # Useful for random selection of incides for a batch
    indices = np.arange(0, numTrainingSamples)
    shuffledTrainingData = []
    shuffledTrainingTarget = []

    # Array variables for training
    tempTrainingLossArraySGD = []
    tempTrainingLossArrayAdam = []

    # Array of losses
    trainingLossPerEpochSGD = []
    trainingLossPerEpochAdam = []

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

        [tempTrainSGD, tempTrainAdam, tempTrainingLossSGD, tempTrainingLossAdam] = sess.run([trainSGD, trainAdam, trainingLossSGD, trainingLossAdam], feed_dict={trainX: batchData, trainY: batchTarget})
        
        # This is the error per iteration i.e per batch
        tempTrainingLossArraySGD.append(tempTrainingLossSGD)
        tempTrainingLossArrayAdam.append(tempTrainingLossAdam)

        # i.e at every epoch find the training loss for SGD and for Adam
        if ((i+1) % numBatches) == 0:
            trainingLossPerEpochSGD.append(tempTrainingLossSGD)
            trainingLossPerEpochAdam.append(tempTrainingLossAdam)

    epochs = np.linspace(0, numEpochs, num = numEpochs)

    figure = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,5])
    plt.plot(epochs, trainingLossPerEpochSGD, "b-", label = 'Training Loss SGD')
    plt.plot(epochs, trainingLossPerEpochAdam, "r-", label = 'Training Loss Adam')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best', shadow = True, fancybox = True)
    plt.title("Best Training Loss vs Number of Epochs for SGD and Adam")    
    plt.show()
    
if __name__ == '__main__':
    print('\n\n\n---------Assignment 2.1.2---------\n\n')
    logisticRegression()