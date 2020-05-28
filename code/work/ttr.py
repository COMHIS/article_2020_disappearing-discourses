import os
from tqdm import tqdm
from random import shuffle
import sys

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
