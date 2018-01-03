from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct
import jieba
import re
import util
 
a = tf.Variable([[ ]])
b = tf.Variable([[1,2], [3,4]])

c = tf.concat([a,b], 0)
c = tf.reshape(c, [-1, 2, 2])
print(c)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(c))