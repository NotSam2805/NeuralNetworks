import networks as n
import numpy as np


dataset_size = 30000 #MAX=60000
file = 'SGD_4.txt'

print('Initialising network...')
network = n.N_Network([128,64,32], 28 * 28, 10)

import mnist_data as mnist

(test_data, test_labels) = mnist.normalised_test_data()
(train_data, train_labels) = mnist.normalised_training_data()

print(f'Using dataset size: {dataset_size}')
(dataset_X, dataset_y) = (train_data[:dataset_size], train_labels[:dataset_size])

train_outputs = mnist.expected_outputs(dataset_y)
print('Dataset formatting complete\n')

import os.path
if os.path.exists(file):
    print(f'Loading network from {file}...')
    network.from_string(n.read_from_file(file))

print('Network loaded\n')
print(f'First input data: {dataset_X[0]}')
print(f'Weights length: {len(network.weights)}')
print(f'Biases length: {len(network.biases)}')
print(f'1st layer biases: {network.biases[0]}')
print('1st layer test:')
dot = np.dot(network.weights[0], dataset_X[0])
zs = []
for d,bias in zip(dot,network.biases[0]):
    zs.append(d + bias)
z = np.array(zs)
a = n.sigmoid(z)
print(f'Dot: {dot}, Z: {z}, Activation: {a}')
print('Network output:')
print(network.feedforward(dataset_X[0]))
print('Output Layer:')
print(network.output_layer())
print('Correct output:')
print(train_outputs[0])

n.save_to_file(network.to_string(), file)
print(f'Network saved to {file}')