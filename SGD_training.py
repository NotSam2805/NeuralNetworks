import networks as n

dataset_size = 60000 #MAX=60000
file = input("Model File: ")
epochs = int(input("Epochs: "))
batch_size = int(input("Batch size: "))
learn_rate = 0.001
eval = False
if (input("Test each epoch (Y/N): ") == "Y"):
    eval = True
test_size = 0
if (eval):
    test_size = int(input("Test size (max=10000): ")) #MAX=10000

print('Initialising network...',end='\r')
network = n.N_Network(
    [256,128,64,32], 28 * 28, 10,
    [n.relu, n.relu, n.relu, n.relu, n.softmax],
    [n.relu_prime, n.relu_prime, n.relu_prime, n.relu_prime]
)
print('Network initialised          ')

import os.path
if os.path.exists(file):
    print(f'Loading network from {file}...',end='\r')
    network.load_json(file)
    print(f'Network loaded from {file}                   \n')

import mnist_data as mnist

(test_data, test_labels) = ([],[])
if(eval):
    (test_data, test_labels) = mnist.normalised_test_data()
(train_data, train_labels) = mnist.normalised_training_data()

print(f'Using dataset size: {dataset_size}')
(dataset_X, dataset_y) = (train_data[:dataset_size], train_labels[:dataset_size])

train_outputs = mnist.expected_outputs(dataset_y)
print('Dataset formatting complete\n')

print('Running SGD training')
dataset = mnist.convert_to_list_of_tuples(dataset_X, train_outputs)

print(f'Using {epochs} epochs, batch sizes of {batch_size}, {int(dataset_size/batch_size)} batches, learn rate = {learn_rate}, evaluate epoch = {eval}')

network = network.train(dataset, epochs, batch_size, learn_rate, eval_epoch=eval, X=test_data[:test_size], y=test_labels[:test_size])

network.save_json(file)
print(f'Model saved to {file}')