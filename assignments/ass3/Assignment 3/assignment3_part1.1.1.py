import tensorflow as tf
import numpy as np

# Function returns weighted sum of inputs to the hidden layer
# Takes two arguments: the input tensor and the number of the hidden units
def build_layer(inputTensor, numHiddenUnits):

	# Fetch number of inputs and create shape for weight vector
	numInputs = inputTensor.get_shape().as_list()[-1]
	weightVectorShape = [numInputs, numHiddenUnits]

	# Variable declaration for Weights and Biases
	W = tf.Variable(tf.random_normal(weightVectorShape, stddev = 3.0 / (numInputs + numHiddenUnits), dtype = tf.float64, name = "Weights"))
	zerosTensor = tf.zeros(dtype = tf.float64, shape = [numHiddenUnits])
	b = tf.Variable(zerosTensor, name = "Biases")

	# Output of the function
	weightedSum = tf.matmul(inputTensor, W) + b
	return weightedSum
