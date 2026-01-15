import networks as n

# SOTF_1,SGD_1.txt and SGD_2.txt are networks of [16,16] layers
# SGD_3.txt is [64,64,32,32]
# SGD_4.txt is [128,64,32]
#
# For ideal MNIST structure (as in 'ReLu_Softmax_test1.txt'),
# use n.N_Network([128,64,32], 28 * 28, 10, [n.relu, n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime, n.relu_prime])
# note that the lists of functions (and the derivatives) is important, the other parameters matter less
# 
# for 'ReLu_Softmax_best_1.txt' use n.N_Network([128,64], 28 * 28, 10, [n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime])
dataset_size = 60000 #MAX=60000
file = 'ReLu_Softmax_3.json'
epochs = 10
batch_size = 256
learn_rate = 0.001
eval = True
test_size = 1000 #MAX=10000

print('Initialising network...',end='\r')
network = n.N_Network([128,128,64], 28 * 28, 10, [n.relu, n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime, n.relu_prime])
print('Network initialised          ')

import mnist_data as mnist

(test_data, test_labels) = ([],[])
if(eval):
    print('Normalising test data...', end='\r')
    (test_data, test_labels) = mnist.normalised_test_data()
print('Normalising training data...', end='\r')
(train_data, train_labels) = mnist.normalised_training_data()
print('Data normalised              ')

print(f'Using dataset size: {dataset_size}')
(dataset_X, dataset_y) = (train_data[:dataset_size], train_labels[:dataset_size])

train_outputs = mnist.expected_outputs(dataset_y)
print('Dataset formatting complete\n')

import os.path
if os.path.exists(file):
    print(f'Loading network from {file}...',end='\r')
    network.load_json(file)
    print(f'Network loaded from {file}                   \n')

print('Running SGD training')
dataset = mnist.convert_to_list_of_tuples(dataset_X, train_outputs)

print(f'Using {epochs} epochs, batch sizes of {batch_size}, {int(dataset_size/batch_size)} batches, learn rate = {learn_rate}, evaluate epoch = {eval}')

network = network.train(dataset, epochs, batch_size, learn_rate, eval_epoch=eval, X=test_data[:test_size], y=test_labels[:test_size])

network.save_json(file)
print(f'Model saved to {file}')