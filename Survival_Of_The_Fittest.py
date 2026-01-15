import networks as n

def survival_of_the_fittest(generations, generation_size, mutation_effect, network_template, data, expected_outputs):
    import copy

    print('Loading initial network')
    best = network_template

    for i in range(generations):
        print('='*80)
        print(f'Starting Generation {i}')
        networks = []
        networks.append(best)
        print('-'*40)
        for j in range(1, generation_size):
            print(f'Loading Neural Network {j}')
            network = copy.deepcopy(best)
            network.mutate(mutation_effect)
            networks.append(network)

        print('-'*40)

        costs = []
        for j in range(generation_size):
            print(f'Evaluating Network {j}...')
            costs.append(networks[j].average_cost(data, expected_outputs))
            print(f'Network {j} evaluated, cost: {costs[j]}\n')
        index = costs.index(min(costs))
        best = networks[index]

        print(f'Generation {i}, best cost: {costs[index]}, from Network {index}')
        print('='*80)
    
    return best

dataset_size = 30000 #MAX=60000
file = 'SOTF_1.txt'

print('Initialising network...')
network = n.N_Network([16,16], 28 * 28, 10)

import mnist_data as mnist

print('Normalising test data...')
(test_data, test_labels) = mnist.normalised_test_data()
print('Normalising training data...')
(train_data, train_labels) = mnist.normalised_training_data()
print('Data normalised')

print(f'Using dataset size: {dataset_size}')
(dataset_X, dataset_y) = (train_data[:dataset_size], train_labels[:dataset_size])

train_outputs = mnist.expected_outputs(dataset_y)
print('Dataset formatting complete\n')

import os.path
if os.path.exists(file):
    print(f'Loading network from {file}...')
    network.from_string(n.read_from_file(file))

print('Network loaded\n')


print('Running Survival Of The Fittest')
generations = 25
generation_size = 25
mutation_effect = 0.1
print(f'{generations} generations, {generation_size} per generation, {mutation_effect} mutation')
network = survival_of_the_fittest(generations, generation_size, mutation_effect, network, dataset_X, train_outputs)

n.save_to_file(network.to_string(), file)
print(f'Network saved to {file}')