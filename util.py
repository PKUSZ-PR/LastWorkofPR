# -*- coding:utf-8 -*-  
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import struct
import jieba
import re
import chardet
import time

map_NER = {'LOC':1, 'PER':2, 'ORG':3, 'MISC':4}
max_sequence_length = 110
max_sequence_num = 210

def get_key(dic, val):
	res =  [k for k,v in dic.items() if v == val]
	return '' if (len(res) == 0) else res[0]

def en_get_key(dic, val):
	val = val.lower()
	res = [] 
	for k, v in dic.items():
		if isinstance(v, int) or isinstance(v, float):
			pass
		else:
			if v.lower() == val:
				res.append(k)
	return '' if (len(res) == 0) else res[0]

def en_find_key(dic, val):
	val = val.lower()
	return dic[val] if (val in dic.keys()) else ''

def ch_find_key(dic, val):
	return dic[val] if (val in dic.keys()) else ''

def readChinsesWordEmbedding(filename):
	dic = {}
	dics = {}
	tup = []
	i = 1; 
	with open(filename, 'r', encoding='utf-8',errors='ignore') as fp:
		lp = fp.readline()
		for ln in fp: 
			ln = ln.strip('\n')
			#ln.decode('utf-8').encode('gbk')
			ln = ln[0:-1]
			s = re.split(' ', ln)

			s[0] = s[0].replace('\n', '')
			dic[i] = s[0].replace(' ', '') 
			dics[s[0].replace(' ', '')] = i
			#print(dic[i])
			tmp_tup = []
			for j in range(1, len(s)): 
				tmp_tup.append(float(s[j]))
			i += 1
			tup.append(tmp_tup) 
	#print(dics, dic)
	return (dic, dics, tup)
	

def readEnglishWordEmbedding(filename):
	dic = {}
	dics = {}
	tup = []
	i = 1;
	with open(filename, 'r', encoding='utf-8', errors='ignore') as fp:
		for ln in fp: 
			ln = ln.replace('\n','')
			s = re.split(' ', ln)
			s[0] = s[0].replace('\n', '')
			dic[i] = s[0].replace(' ', '') 
			dics[s[0].replace(' ', '')] = i
			tmp_tup = []
			for j in range(1, len(s)):
				tmp_tup.append(float(s[j]))
			i += 1
			tup.append(tmp_tup)
	return (dic, dics, tup)
	
def splitChineseSequence(filename, dic, dics):
	#cutlist = ["，", ",", "！", "……", "!", "？", "?", "；", ";" ]
	#cutlist = r'\，\,\;\；.\。\?\？\!\！' 
	cutlist = r'\,|\，|\.|\。|\！|\!|\?|\？|\:|\：'
	fp = open(filename, 'r', encoding='utf-8', errors='ignore') 
	strAll = fp.read().replace('\n','')
	strAll = strAll.replace('\t', '')
	col = re.split(cutlist, strAll)
	res = []
	for st in col: 
		lins = re.split(' ', st) 
		reslist = [0 for i in range(max_sequence_length)]

		pro_lins = []
		for x in lins:
			if x != '':
				pro_lins.append(x) 
		i = -1
		for vob in pro_lins:
			# if(len(pro_lins) > 58):
			# 	print(len(pro_lins))
			i = i + 1
			idx = ch_find_key(dics, vob)
 
			#idx = get_key(dic, vob)
			if idx == '':
				continue
			reslist[i] = idx
			
		res.append(reslist)
 
	fp.close()
	return res

def splitEnglishiSeqence(filename, dic, dics): 
	cutlist = r'\.|\?|\,|\…|\:|\;'
	fp = open(filename, 'r', encoding='utf-8', errors='ignore')
	strAll = fp.read().replace('\n', '')
	strAll = strAll.replace('\t', '')
	col = re.split(cutlist, strAll)
	res = []
	for st in col:
		lins = re.split(' ', st)
		reslist = [0 for i in range(max_sequence_length)]
		i = -1
		pro_lins = []
		for x in lins:
			if x != '':
				pro_lins.append(x)
		for vob in pro_lins:
			i += 1
			idx = en_get_key(dic, vob)
			if idx == '':
				continue
			reslist[i] = idx
		res.append(reslist)
	fp.close()
	return res

def mapChineseNER(filename, dic, article, dics): 
	res = [[0 for j in range(max_sequence_length)] for i in range(len(article))]
	map_file = {}
	with open(filename, 'r', encoding='utf-8', errors='ignore') as fp:
		for st in fp:
			st = st.replace('\n', '')
			st = re.split('\t', st)

			#idx = en_find_key(dics, st[-1])
			idx = get_key(dic, st[-1].replace(' ', '')) 
			if idx == '':
				idx = 0
			ydx = map_NER[st[1]]
			map_file[idx] = ydx 
		for i in range(len(article)):
			for j in range(max_sequence_length):
				if article[i][j] != 0:
					idx = article[i][j]
					# print ('idx', idx, dic[idx])
					if idx in map_file:
						ydx = map_file[idx]
					else:
						ydx = 0
					res[i][j] = ydx 
	return res

def mapEnglishNER(filename, dic, article, dics): 
	res = [[0 for j in range(max_sequence_length)] for i in range(len(article))]
	map_file = {}
	with open(filename, 'r', encoding='utf-8', errors='ignore') as fp:
		for st in fp:
			st = st.replace('\n', '')
			st = re.split('\t', st)
			word_ner = re.split(' ', st[1])
			idx = en_find_key(dics, st[-1])  
			if idx == '':
				idx = 0
			
			ydx = map_NER[word_ner[0]]
			map_file[idx] = ydx

		for i in range(len(article)):
			for j in range(max_sequence_length):
				if article[i][j] != 0:
					idx = article[i][j]
					# print ('idx', idx, dic[idx])
					if idx in map_file:
						ydx = map_file[idx]
					else:
						ydx = 0
					article[i][j] = ydx
 
	return res

def loadChineseData():
	path = os.getcwd()
	word_path = path + "\\" + "Chinese_Embedding.txt" 
	dic, dics, word_embedding = readChinsesWordEmbedding(word_path)
	ans = 0
	data = []
	data_ner = []
	datasetpath = path + "\\" + "chinese_data"
	filenameList = os.listdir(datasetpath)
	filenameList.sort(reverse=True) 
	for fn in filenameList:
		if fn.find("-3.txt") > 0:
			fn = datasetpath + "\\" + fn 
			ans = splitChineseSequence(fn, dic, dics)
			data.append(ans)
 
		elif fn.find("-1.txt") > 0:
			fn = datasetpath + "\\" + fn 
			v = mapChineseNER(fn, dic, ans, dics)
			data_ner.append(v)

		else:
			pass
	return (data, data_ner, word_embedding)


def loadEnglishiData():
	path = os.getcwd()
	word_path = os.path.join(path, "glove.6B.100d.txt")
	print('word_path', word_path)
	dic, dics, word_embedding = readEnglishWordEmbedding(word_path)

	

	data = []
	data_ner = []
	datasetpath = os.path.join(path, "english_data") 
	filenameList = os.listdir(datasetpath)
	filenameList.sort(reverse=True)
	for fn in filenameList:
		if fn.find("txt") > 0:
			fn = os.path.join(datasetpath, fn)
			ans = splitEnglishiSeqence(fn, dic, dics) 
		elif fn.find("ann") > 0:
			fn = os.path.join(datasetpath, fn)
			data_ner.append(mapEnglishNER(fn, dic, ans, dics))
		else:
			pass
	return (data, data_ner, word_embedding)



def DivideDataSet(dataset, nerdata):
	batch_size = len(dataset)
	print ('batch', batch_size)
	train_set = []
	test_set = []
	train_lb = []
	test_lb = []
	max_sequence_num = 210
	n = random.randint(3, 7)
	#dataset = np.array(dataset)
	#nerdata = np.array(nerdata)
	for i in range(batch_size):
		batch = dataset[i]
		batch_ner = nerdata[i ]
		l = max_sequence_num - len(batch)
		#batch = np.reshape(batch, (-1, max_sequence_length))
		#batch_ner = np.reshape(batch_ner, (-1, max_sequence_length))
		
		# if max_sequence_num < len(batch):
		# 	max_sequence_num = len(batch)
		if l == 0:
			continue
		for j in range(l):
			batch.append([0 for k in range(max_sequence_length)])
			batch_ner.append([0 for k in range(max_sequence_length)])
 
		
		batch = np.reshape(batch, (-1, max_sequence_length))
		batch_ner = np.reshape(batch_ner, (-1, max_sequence_length))	
		if i % n == 0:
			test_set.append(batch)
			test_lb.append(batch_ner)
		else:
			train_set.append(batch)
			train_lb.append(batch_ner)
	# for i in range(len(test_set)):
	# 	size =max_sequence_num - len(test_set[0])
	# 	for j in range(size):
	# 		#test_set[i].append([0 for k in range(max_sequence_length)])	
	# 		#test_lb[i].append([0 for k in range(max_sequence_length)])	
	# 		test_set[i] = np.row_stack(test_set[i], [0 for k in range(max_sequence_length)])
	# 		test_lb[i] = np.row_stack(test_set[i], [0 for k in range(max_sequence_length)])

	return (batch_size, max_sequence_num, train_set, train_lb, test_set, test_lb)


#--------------------Test functions-----------------

if __name__ == '__main__':
	print ("util.py main funcion")
	#(data, data_ner, word_embedding) = loadChineseData()  
	#print (data)
	(data, data_ner, word_embedding) = loadChineseData()
	(batch_size, max_sequence_num, train_set, train_lb, test_set, test_lb) = DivideDataSet(data, data_ner)
 
	for i in range(len(train_set)):
	 	print ('i:', i,  len(train_set[i]), len(train_lb[i][1]))


