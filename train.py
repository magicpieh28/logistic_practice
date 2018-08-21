from path import  Path
import os
from collections import Counter
import numpy as np
from random import shuffle


positive = Path('/Users/jungwon-c/Documents/ML Logistic/data/books/positive.review')
negative = Path('/Users/jungwon-c/Documents/ML Logistic/data/books/negative.review')


def token_freq(path:Path):
	with open(path) as data:
		lines = []
		for line in data:
			sentence = []
			for word in line.strip().split():
				separate = word.find(':')
				token = word[:separate]
				freq = int(word[separate+1:])
				sentence.append((token, freq))
			lines.append(sentence)
			# print(sentence)
		return lines

def count_freq(path:Path, counter:Counter):
	lines = token_freq(path)
	for line in lines:
		for token, freq in line:
		# print(f'token => {token}\nfreq => {freq}')
			counter[token] += freq
		# for token, freq in :
		# 	print(f'{token}, {freq}')
		# counter[token] = freq+1
	# print(counter)
	return counter

def make_vocabulary(vocab_size:int, counter:Counter):
	vocabulary = {}
	# print(f'original counter => {counter}')
	count_freq(positive, counter)
	# print(f'only pos word => {counter}')
	count_freq(negative, counter)
	# print(f'add neg word => {counter}')
	for index, counted_token in enumerate(counter.most_common(vocab_size)):
		vocabulary[index+1] = counted_token
	# print(vocabulary)
	return vocabulary

def make_vector(path:Path, vocab_size:int, target:float):
	vocabulary = make_vocabulary(vocab_size, counter)
	lines = token_freq(path)
	vocab_values = [value[0] for value in vocabulary.values()]
	# print(vocab_values)
	for line in lines:
		vector = [0] * len(line)
		for index, pair in enumerate(line):
			# print(f'index => {index}\npair => {pair}')
			if pair[0] in vocab_values:
				vector[index] = 1
				# print(vector)
		yield vector, target
		# print(vector, target)

def return_with_target(vocab_size:int):
	pos_vector, pos_target = zip(*make_vector(positive, vocab_size, 1.0))
	# print(f'pos_vector => {pos_vector}\npos_target => {pos_target}')
	neg_vector, neg_target = zip(*make_vector(negative, vocab_size, 0.0))
	# print(f'neg_vector => {neg_vector}\nneg_target => {neg_target}')
	dataset = list(zip(pos_vector+neg_vector, pos_target+neg_target))
	shuffle(dataset)
	data, target = zip(*dataset)
	print(f'data => {data}\ntarget => {target}')
	return data, target

def iteration(data, target, batch_size:int):
	for sample_num in range(0, len(data)+1, batch_size):
		yield data[sample_num * batch_size : (sample_num+1) * batch_size],\
			target[sample_num * batch_size : (sample_num+1) * batch_size]



if __name__ == '__main__':
	counter = Counter()
	# token_freq(positive)
	# count_freq(positive, counter)
	# make_vocabulary(2000, counter)
	# make_vector(negative, 2000)
	return_with_target(200)