import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()

def loadData():
	with np.load("notMNIST.npz") as data:
		Data, Target = data ["images"], data["labels"]
		np.random.seed(521)
		randIndx = np.arange(len(Data))
		np.random.shuffle(randIndx)
		Data = Data[randIndx]/255.
		Target = Target[randIndx]
		trainData, trainTarget = Data[:15000], Target[:15000]
		validData, validTarget = Data[15000:16000], Target[15000:16000]
		testData, testTarget = Data[16000:], Target[16000:]

		return trainData, trainTarget, validData, validTarget, testData, testTarget

# Function returns weighted sum of inputs to the hidden layer
# Takes two arguments: the input tensor and the number of the hidden units
def build_layer(inputTensor, numHiddenUnits):

	# Fetch number of inputs and create shape for weight vector
	numInputs = inputTensor.get_shape().as_list()[-1]
	weightVectorShape = [numInputs, numHiddenUnits]

	# Variable declaration for Weights and Biases
	# W = tf.get_variable(dtype = tf.float64, shape = weightVectorShape, initializer = tf.contrib.layers.xavier_initializer(), name = "Weights")
	W = tf.Variable(tf.random_normal(weightVectorShape, stddev = 3.0 / (numInputs + numHiddenUnits), dtype = tf.float64, name = "Weights"))
	zerosTensor = tf.zeros(dtype = tf.float64, shape = [numHiddenUnits])
	b = tf.Variable(zerosTensor, name = "Biases")

	# Output of the function
	weightedSum = tf.matmul(inputTensor, W) + b
	return weightedSum

def calculateCrossEntropyLoss(y, yHat, wdc):
	''' y is the target,
	    yHat is the output prediction,
	    wdc (lambda) is the weight decay coefficient
	    WHidden/Woutput are the weights for hidden/output later respectively
	    Cross Entropy Loss = Ld + Lw
	''' 

	WHidden1 = tf.get_default_graph().get_tensor_by_name("hiddenLayer1/Weights:0")
	WHidden2 = tf.get_default_graph().get_tensor_by_name("hiddenLayer2/Weights:0")
	WOutput = tf.get_default_graph().get_tensor_by_name("outputLayer/Weights:0")

	Ld = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(indices = tf.cast(y, tf.int32), depth = 10, on_value = 1.0, off_value = 0.0, axis = -1), logits = yHat))
	Lw = (wdc / 2) * tf.reduce_sum(tf.square(WHidden)) * tf.reduce_sum(tf.square(WOutput))
	loss = Lw + Ld
	return loss

def neuralNetwork():
	# load notMNIST data and targets
	trainData, trainTarget, validData, validTarget, testData, testTarget = loadData()

	# Reshape data to have a width of 28x28 i.e image size
	# Cast target shape into int
	trainData = trainData.reshape(trainData.shape[0], 28*28)
	trainTarget = trainTarget.reshape(trainTarget.shape[0]).astype(int)

	validData = validData.reshape(validData.shape[0], 28*28)
	validTarget = validTarget.reshape(validTarget.shape[0]).astype(int)

	testData = testData.reshape(testData.shape[0], 28*28)
	testTarget = testTarget.reshape(testTarget.shape[0]).astype(int)

	# Parameter Declarations
	numClasses = 10
	wdc = 3e-4
	learningRate = 0.0001
	numHiddenUnits = 500

	batchSize = 500
	numIterations = 6000

	numTrainingSamples = trainData.shape[0]
	numBatches = numTrainingSamples // batchSize
	numEpochs = numIterations // numBatches

	# x0 is input to first layer
	x0 = tf.placeholder(dtype = tf.float64, shape = [None, 28*28], name = "Data")
	y0 = tf.placeholder(dtype = tf.int32, shape = [None], name = "Target")

	# x1 is output of first layer (and input to second later)
	with tf.variable_scope("hiddenLayer1"):
		x1 = tf.nn.relu(build_layer(x0, numHiddenUnits))

	# x2 is output of second layer (and input to output later)
	with tf.variable_scope("hiddenLayer2"):
		x2 = tf.nn.relu(build_layer(x1, numHiddenUnits))

	# sOut is the weighted sum of the output layer which will be fed into softmax
	with tf.variable_scope("outputLayer"):
		sOut = build_layer(x2, numClasses)

	# Adam Optimizer
	crossEntropyLoss = calculateCrossEntropyLoss(y0, sOut, wdc)
	optimizerAdam = tf.train.AdamOptimizer(learning_rate = learningRate)
	trainAdam = optimizerAdam.minimize(crossEntropyLoss)

	# Send sOut into softmax and get output of the last/output later
	# This is nothing by the final classification of the image
	x3 = tf.nn.softmax(sOut)
	classification = tf.cast(tf.equal(tf.argmax(x3, 1), tf.cast(y0, tf.int64)), tf.float32)
	classificationAccuracy = tf.reduce_mean(classification) * 100

	sess.run(tf.global_variables_initializer())
	trainSaver = tf.train.Saver()

	# Arrays for recording data
	trainingLoss = []
	validationLoss = []
	testLoss = []

	trainingAccuracy = []
	validationAccuracy = []
	testAccuracy = []

	# Create an array of indices from 0 to numTrainingSamples
	# Useful for random selection of incides for a batch
	indices = np.arange(0, numTrainingSamples)
	shuffledTrainingData = []
	shuffledTrainingTarget = []

	# numIterations is 6000
	# batchSize is 500
	# numBatches is 30 (stays constant)
	# numEpochs is 6000/30 = 200
	for i in range(numIterations):

		# Shuffle indices once every numBatches (30) iterations
		if not (i % numBatches): 
			print(i)
			np.random.shuffle(indices)
			shuffledTrainingData = trainData[indices]
			shuffledTrainingTarget = trainTarget[indices]

		startBatchIndex = (i % numBatches) * batchSize
		endBatchIndex = startBatchIndex + batchSize

		# Obtain a batch of training data from start index to end index
		batchData = shuffledTrainingData[startBatchIndex : endBatchIndex]
		batchTarget = shuffledTrainingTarget[startBatchIndex : endBatchIndex]

		sess.run(trainAdam, feed_dict={x0: batchData, y0: batchTarget})

		if not startBatchIndex:
			trainingLoss.append(sess.run(crossEntropyLoss, feed_dict = {x0: batchData, y0: batchTarget}))
			validationLoss.append(sess.run(crossEntropyLoss, feed_dict = {x0: validData, y0: validTarget}))
			testLoss.append(sess.run(crossEntropyLoss, feed_dict = {x0: testData, y0: testTarget}))

			trainingAccuracy.append(sess.run(classificationAccuracy, feed_dict = {x0: batchData, y0: batchTarget}))
			validationAccuracy.append(sess.run(classificationAccuracy, feed_dict = {x0: validData, y0: validTarget}))
			testAccuracy.append(sess.run(classificationAccuracy, feed_dict = {x0: testData, y0: testTarget}))

	print(validationLoss[-1])
	print(validationAccuracy[-1])

	# Plotting
	epochs = np.linspace(0, numEpochs, num = numEpochs)

    # Plot loss vs number of epochs
	figure = plt.figure()
	axes = plt.gca()
	plt.plot(epochs, trainingLoss, "r-", label = 'Training Loss')
	plt.plot(epochs, validationLoss, "g-", label = 'Validation Loss')
	plt.xlabel("Number of epochs")
	plt.ylabel("Loss")
	plt.legend(loc='best', shadow = True, fancybox = True)
	plt.title("Training and Validation Loss vs Number of Epochs")    
	plt.show()

	# Plot loss vs number of epochs
	figure = plt.figure()
	axes = plt.gca()
	plt.plot(epochs, trainingAccuracy, "r-", label = 'Training Accuracy')
	plt.plot(epochs, validationAccuracy, "g-", label = 'Validation Accuracy')
	plt.xlabel("Number of epochs")
	plt.ylabel("Loss")
	plt.legend(loc='best', shadow = True, fancybox = True)
	plt.title("Training and Validation Accuracy vs Number of Epochs")    
	plt.show()

if __name__ == '__main__':
    print('\n\n\n---------Assignment 3---------\n\n')
    neuralNetwork()
