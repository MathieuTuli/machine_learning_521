#______AUTHOR: MATHIEU TULI______
#______DATE: Jan. 20, 2018______
#______DUE: Feb. 2, 2018______

from __future__ import print_function
import numpy as np
import tensorflow as tf

#----------Question: 1---------------------------------------------------------

def eucl_dist(X,Z):
    XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
    ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
    #for both...dim1 = D. for dim0 and dim2, there is a corresponding 1.
    #makes them compatible for broadcasting

    #------------------Some Testing------------------
    # print(sess.run(tf.shape(XExpanded)))
    # print(sess.run(XExpanded))
    # print("\n\n")
    #
    # print(sess.run(tf.shape(ZExpanded)))
    # print(sess.run(ZExpanded))
    # print("\n\n")
    #
    # print(sess.run((XExpanded-ZExpanded)**2))
    # print("\n\n")


    #return element-byelement subtracted square accross axis1; or for D.
    return tf.reduce_sum((XExpanded-ZExpanded)**2, 1)

sess = tf.Session()
x = tf.constant([[1,2,1,2,2],[3,4,1,2,2]])
z = tf.constant([[11,22,1,23,32],[13,14,1,22,12],[2,3,4,5,9]])
print(sess.run(eucl_dist(x,z)))

#----------Question: 2---------------------------------------------------------
#randomly generated training and test data sets
np.random.seed(521)
Data = np.linspace(1.0, 10.0, num=100) [:, np. newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data, 2) +  0.5 * np.random.randn(100, 1)
randIdx = np.arange(100)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

def training_responsibilities(distMatrix,k):
    constant = tf.constant([1/k])
    neighbours = tf.nn.top_k(distMatrix,k)
    return
