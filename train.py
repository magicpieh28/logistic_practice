from pathlib import Path
from collections import Counter
from random import shuffle
from typing import Dict, List, Tuple


dir = Path('/Users/jungwon-c/Documents/ML Logistic/data')
positive = dir/'books'/'positive.review'
negative = dir/'books'/'negative.review'

UNK = '<UNK>' # unknown 단어가 나왔을 때를 위한 포석


def token_freq(path:Path) -> List:
	with path.open(mode='r', encoding='utf-8') as data:
		lines = []
		for line in data:
			sentence = []
			for word in line.strip().split():
				separate = word.find(':')
				token = word[:separate]
				freq = int(word[separate+1:])
				sentence.append((token, freq))
			lines.append(sentence)
			# print(f'sentence => {sentence}')
			# print(f'zip(*sentence) => {list(zip(*sentence))}')
		return lines

def count_freq(path:Path, counter:Counter):
	lines = token_freq(path)
	for line in lines:
		for token, freq in line:
		# print(f'token => {token}\nfreq => {freq}')
			counter[token] += freq
	# print(f'counter => {counter}\ntype => {type(counter)}')
	return counter

def make_vocabulary(vocab_size:int, counter:Counter) -> Dict:
	vocabulary = {}
	# print(f'original counter => {counter}')
	count_freq(positive, counter)
	# print(f'only pos word => {counter}')
	count_freq(negative, counter)
	# print(f'add neg word => {counter}')
	for index, (token, _) in enumerate(counter.most_common(vocab_size)): # .most_common()은 튜플을 묶은 리스트를 반환한다
		vocabulary[token] = index # 거꾸로 하면 보기 편하지만 make_BOW_vector에서 key error일으킴
	vocabulary[len(vocabulary)] = UNK
	# print(vocabulary)
	return vocabulary

def make_BOW_vector(vocabulary:dict, sentence:list) -> List:
	vector = [0] * len(vocabulary)
	for (token, _) in sentence:
		# print(token)
		if token in vocabulary.keys():
			index = vocabulary[token]
		else:
			index = len(vocabulary)-1
		vector[index] = 1
	# print(vector)
	return vector

def make_data(path:Path, vocab_size:int, target:float) -> List:
	vocabulary = make_vocabulary(vocab_size, counter)
	data = []
	for sentence in token_freq(path):
		sentence_vector = make_BOW_vector(vocabulary, sentence)
		data.append((sentence_vector, target))
	# print(data)
	return data

def return_with_target(vocab_size:int) -> Tuple:
	pos_data = make_data(positive, vocab_size, 1.0)
	# print(f'pos => {type(pos_data)}')
	neg_data = make_data(negative, vocab_size, 0.0)
	dataset = pos_data + neg_data
	shuffle(dataset)
	data, targets = zip(*dataset)
	print(f'data => {data}\ntargets => {targets}')
	return data, targets

def iteration(data, target, batch_size:int):
	for sample_num in range(0, len(data)+1, batch_size): # 0 부터 data 길이까지 batch_size만큼씩 불러오면서 반복
		yield data[sample_num * batch_size : (sample_num+1) * batch_size],\
			target[sample_num * batch_size : (sample_num+1) * batch_size]



if __name__ == '__main__':
	counter = Counter()
	# token_freq(positive)
	# count_freq(positive, counter)
	# make_vocabulary(200, counter)
	# make_BOW_vector(make_vocabulary(200, counter), token_freq(positive))
	# make_data(positive, 200, 1.0)
	return_with_target(200)