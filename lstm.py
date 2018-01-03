# -*- coding:utf-8 -*-  
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct
import util

'''
	This is the first version to use lstm and crf to sovle ner problem
'''


#-----------------------------hyper parameters------------------------
batch_size = 99
max_sequence_length = util.max_sequence_length #default 110
word_embedding = []
word_embedding_size = 0
hidden_size = 120
num_layers = 10
num_class = 5
epoch_size = 1000
check_size = 100
learning_rate = 0.1
forget_bias = 0.0
input_keep_prob = 1.0
output_keep_prob = 1.0
#--------------------------------------------------
init = 0
sess = []
correct_prediction = []
train_op = 0
accuracy = 0
tf_input = 0
tf_target = 0
#--------------------------------------------------
(data, data_ner, word_embedding) = util.loadChineseData()
(batch_size, max_sequence_num, train_set, train_lb, test_set, test_lb) = util.DivideDataSet(data, data_ner)
word_embedding.insert(0, [0 for i in range(len(word_embedding[0]))])
word_embedding_size = len(word_embedding )
word_embedding_dim = len(word_embedding[0])

#def mynet ():
tf_input = tf.placeholder(dtype=tf.int32, shape=[max_sequence_num, max_sequence_length])
tf_target = tf.placeholder(dtype=tf.int32, shape=[max_sequence_num, max_sequence_length])
print('fuck')
#class_output = tf.nn.embedding_lookup(np.eye(5,5), tf_target)
map_nertype = [[0 if j != i else 1 for j in range(5)] for i in range(5)]
word_embedding = np.array(word_embedding, dtype=np.float64)
map_nertype = np.array(map_nertype, dtype=np.float64)
cell_input = tf.nn.embedding_lookup(word_embedding, tf_input)
class_output = tf.nn.embedding_lookup(map_nertype, tf_target)

cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True)
cell_bk = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True, reuse=True)
#cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
#cell_bk = tf.nn.rnn_cell.DropoutWrapper(cell_bk, input_keep_prob = input_keep_prob, output_keep_prob=output_keep_prob)

#cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw]*num_layers, state_is_tuple=True)
#cell_bk = tf.nn.rnn_cell.MultiRNNCell([cell_bk]*num_layers, state_is_tuple=True)


initial_state_fw = cell_fw.zero_state(max_sequence_num, dtype=tf.float64)
initial_state_bk = cell_bk.zero_state(max_sequence_num, dtype=tf.float64)
#inputs_list = [tf.squeeze(s, squeeze_dims=1) for s in tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)]  
tf_tmp = 0
s = tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)
for i in range(len(s)):
	if i == 0:
		tf_tmp = s[i]
	else:
		tf_tmp = tf.concat([tf_tmp, s[i]], 0)
inputs_list = tf.reshape(tf_tmp, [-1, max_sequence_num, word_embedding_dim])
print (inputs_list)
# st = tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)
# for s in st:
# 	inputs_list.append(s)

outputs, state_fw, state_bw = \
			tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bk, inputs_list, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bk)  

output = tf.reshape((1, outputs), [-1, hidden_size])
W = tf.get_variable('W', [hidden_size, num_class], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
b = tf.get_variable('b', [num_class], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
logits = tf.matmul(output, W) + b
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(class_output, [-1, num_class]))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(tf_target,1), tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, max))

init = tf.global_variables_initializer()
sess = tf.Session()

#if __name__ == '__main__':

#--------------------Preservation----------------------- 
# (data, data_ner, word_embedding) = util.loadChineseData()
# (batch_size, train_set, train_lb, test_set, test_lb) = util.DivideDataSet(data, data_ner)
# word_embedding.insert(0, [0 for i in range(len(word_embedding[0]))])
word_embedding_size = len(word_embedding )
word_embedding_dim = len(word_embedding[0])
print(word_embedding_size, word_embedding_dim)



#----------------------Run session--------------------------

train_size = len(train_set)
test_size = len(test_set)

sess.run(init)
for i in range(epoch_size):
	sess.run(train_op, feed_dict={tf_input: train_set[i%train_size], tf_target: train_lb[i%train_size]})
	pass
	if i % check_size == 0:
		print ("epoch_size:", i, sess.run([accuracy], feed_dict={tf_input: test_set[i%test_size], tf_target: test_lb[i%test_size]}))


