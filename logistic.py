import numpy as np
from train import return_with_target, iteration

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

class Classifier(object):
	def __init__(self, input_size):
		self.input_size = input_size
		self.W = np.random.random(input_size)
		self.b = np.array(0.)

	def forward(self, input_data): #sigmoid
		z = input_data @ self.W + self.b
		return sigmoid(z)

	def backward(self, input_data, output, target): #backward = back propagation
		# 로지스틱 회귀의 최적화는 backward를 계산하는 것이며 이의 값은 광배법을 위한 값이 된다.
		# print(f'input_data => {input_data}\noutput => {output}\ntarget => {target}')
		# targets.shape = > (2000,), input.shape = > (32, 1001), output.shape = > (32,)
		grad = (target - output) @ input_data / input_data.shape[0]
		# print(f'grad => {grad}')
		error_mean = np.mean(output - target)
		# print(f'error_mean => {error_mean}')
		return grad, error_mean

def cross_entropy(output, target):
	e = np.finfo(np.float32).eps
	cross = -(target * np.log(output  + e)) + (1-target) * np.log(1-output + e)
	# 곱하기 * 를 사용하면 sum을 사용해야 하지만 다 내적으로 곱해버리면 sum은 필요없음
	# x 내적 y = sum(x_i * y_i) 그렇지! 직접 행렬로 적어서 보면 아는 것을...
	return float(np.mean(cross))

def accuracy(output, target):
	y = np.array([y_i > 0.5 for y_i in output], dtype=np.int32)
	t = np.array([target], dtype=np.int32)
	return float(np.mean(y == t))

def train(vocab_size:int = 1000, epoch_num:int = 100, batch_size:int = 8, step_size:float = 0.05):
	data, targets = return_with_target(vocab_size)
	# print(f'data from train => {type(data)}\ntargets from train => {type(targets)}')
	data = np.array(data, dtype=np.float32)
	targets = np.array(targets, dtype=np.float32)
	# data.shape => (2000, 1001), targets.shape => (2000,)
	logistic = Classifier(vocab_size+1) # UNK 를 추가했으니까

	for epoch in range(epoch_num):
		for input_data, t in iteration(data, targets, batch_size):
			output = logistic.forward(input_data)
			grad_W, grad_b = logistic.backward(input_data, output, t)

			logistic.W -= step_size * grad_W
			logistic.b -= step_size * grad_b

			# print(output)

		outputs = logistic.forward(data)
		acc_score = accuracy(outputs, targets)
		loss_score = cross_entropy(outputs, targets)
		print(f'epoch => {epoch+1} / accuracy => {acc_score:.2f} / loss => {loss_score:.2f}')



if __name__ == '__main__':
	train()