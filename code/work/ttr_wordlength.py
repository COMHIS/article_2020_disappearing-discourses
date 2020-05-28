import os
from tqdm import tqdm
from random import shuffle
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def get_ttr(list_files):
	'''
	Will read a list of files
	and return TTR
	
	## adapt to other type of iterables

	'''
	types_tokens = dict()
	for file in tqdm(files):
		with open(f) as f:
			text = f.read()
			text = text.lower()
		for w in text.split():
			try:
				types_tokens[w] += 1
			except KeyError:
				types_tokens[w] = 1

	types = len(types_tokens.keys())
	tokens = sum(list(types_tokens.values()))
	TTR = types/tokens

	return TTR

def length(list_files):
#def length():
	'''
	Gets a list of files
	- Returns a distribution based on word length
	- Returns a distrubtion based on number of words
	'''
	length_doc = [] ## that's the document length
	length_word = []
	for file in list_files:
		with open(file) as f:
			text = f.read()
			text = text.split()
		length_doc.append(len(text))
		for w in text:
			length_word.append(len(w))
	
	#length_doc = [10,10,15,20,10,1,1,1,1,1,1,1] ## testing purpose, ignore
	#length_word = [5,10,12,12,12,5,10,1000,10]
	mean_len_doc = np.mean(length_doc)
	std_len_doc = np.std(length_doc)
	print("Mean of doc length",mean_len_doc, "("+str(std_len_doc)+")")


	mean_len_word = np.mean(length_word)
	std_len_word = np.std(length_word)
	print("Mean of word length",mean_len_word, "("+str(std_len_word)+")")

	plt.hist(length_doc, bins=int(sqrt(len(length_doc))))
	plt.title("Distribution of # of words per doc")
	plt.show()

	plt.hist(length_word, bins=int(sqrt(len(length_word))))
	plt.title("Distribution of # of characters per word")
	plt.show()


def corpus_shuffler(list_1, list_2):
	'''
	This will take two lists of files 
	will compare and randomly harmonise lengths (based on amount of files)
	so that TTR is comparable
	'''
	if len(list_1) > len(list_2):
		random.shuffle(list_1)
		list_1 = list_1[:len(list_2)]
	else:
		random.shuffle(list_2)
		list_2 = list_2[:len(list_1)]

	TTR1 = get_ttr(list_1)
	TTR2 = get_ttr(list_2)

	return TTR1, TTR2

if __name__ == '__main__':

	TTRs = corpus_shuffler(list_1,list_2)
	print("TTRs are",TTRs)
	length(list_1, list_2)
