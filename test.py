from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct

dic = [[1,2,2,3], [1,2,3,3]]
dic = np.array(dic)
z = np.array( [[1,2,2,3], [1,2,3,3]])
print(np.concatenate((dic, z), axis=0))




