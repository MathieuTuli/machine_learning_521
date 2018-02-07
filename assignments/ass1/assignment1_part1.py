from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sess = tf.Session()

#----------Question: 1---------------------------------------------------------

def eucl_dist(X,Z):
    XExpanded = tf.expand_dims(X,2) # shape [N1, D, 1]
    ZExpanded = tf.expand_dims(tf.transpose(Z),0) # shape [1, D, N2]
    #for both...axis2 = D. for axis0 and axis2, there is a corresponding size 1.
    #makes them compatible for broadcasting

    #return the reduced sum accross axis 1. This will sum accros the D dimensional
    #element thus returning the N1xN2 matrix we desire
    return tf.reduce_sum((XExpanded-ZExpanded)**2, 1)
