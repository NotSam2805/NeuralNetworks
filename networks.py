import random as rnd
import numpy as np
import json
import sys

def sigmoid(x):
  return np.divide(1,np.add(np.exp(-x),1))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return (sig * (1 - sig))

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class N_Network:
    
    def __init__(self, layer_sizes, n_inputs, n_outputs, activation_functions = [], activation_derivatives = []):
        self.sizes = [n_inputs] # First layer is the input layer
        for size in layer_sizes: self.sizes.append(size) # Add each hidden layer
        self.sizes.append(n_outputs) # Add the output layer

        if activation_functions == []:
            for i in range(len(self.sizes) - 1):
                activation_functions.append(sigmoid)
                activation_derivatives.append(sigmoid_prime)
            activation_derivatives = activation_derivatives[:-1]
        elif len(activation_functions) != (len(self.sizes) - 1):
            raise Exception('Activation functions in incorrect format')
        
        self.activation_functions = activation_functions
        self.activation_derivatives = activation_derivatives

        self.biases = [np.random.uniform(-1.0, 1.0, (y, 1)) for y in self.sizes[1:]] # Initalise with random bias
        self.weights = [np.random.uniform(-1.0, 1.0, (y, x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])] # Initalise with random weights

        self.activations = []

    def feedforward(self, input):
        activation = input
        self.activations = [activation]
        zs = []
        layer_count = 0
        for b,w in zip(self.biases, self.weights):
            dot = np.dot(w, activation)
            t_zs = []
            for d,bias in zip(dot, b):
                t_zs.append(d + bias)
            z = np.array(t_zs)
            zs.append(z)

            activation = self.activation_functions[layer_count](z)
            self.activations.append(activation)
            layer_count += 1
        return activation
    
    def backprop(self, x, y):
        #  return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x,
        #  nabla_b and nabla_w are layer-by-layer lists of numpy arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        self.activations = [activation]
        zs = []
        layer_count = 0
        for b,w in zip(self.biases, self.weights):
            dot = np.dot(w, activation)
            t_zs = []
            for d,bias in zip(dot, b):
                t_zs.append(d + bias)
            z = np.array(t_zs)
            zs.append(z)

            activation = self.activation_functions[layer_count](z)
            self.activations.append(activation)
            layer_count += 1
        
        #self.feedforward(x)
        np_activations = np.array(self.activations, dtype=object)

        #backward pass
        cost_delta = self.cost_derivative(np_activations[-1], y)
        delta = cost_delta
        if self.activation_functions[-1] == sigmoid:
            sp = sigmoid_prime(zs[-1])
            delta = cost_delta * sp
        #print(f'Delta: {delta}')
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(np_activations[-2]))

        for i in range(2, len(np_activations)):
            z = zs[-i]
            sp = self.activation_derivatives[1-i](z)
            layer_count += 1
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp

            nabla_b[-i] = delta
            try:
                nabla_w[-i] = np.dot(delta, np.transpose(np_activations[-i-1]))
            except:
                actv = []
                for a in np_activations[-i-1]:
                    actv.append([a])
                n_actv = np.array(actv)
                nabla_w[-i] = np.dot(delta, np.transpose(n_actv))
        
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, learn_rate):
        # Update the network's weights and biases by applying
        # gradient descent using backpropagation to a single mini batch.

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w-(learn_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learn_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def train(self, training_data, epochs, mini_batch_size, learn_rate = 0.1, eval_epoch = False, X = False, y = False):
        # Train the neural network using mini-batch stochastic gradient descent.
        # Evaluating each epoch is useful for keeping track of progress, but slows down training considerably
        print('='*100)
        n = len(training_data)
        for i in range(epochs):
            print(f'Starting epoch {i + 1}')
            print()
            rnd.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            print('Mini batches generated')
            print()
            count = 1
            printProgressBar(0, len(mini_batches), 'Batch progress:', 'Complete')
            for batch in mini_batches:
                printProgressBar(count, len(mini_batches), 'Batch progress:', 'Complete')
                count += 1
                self.update_mini_batch(batch, learn_rate)
            
            if(eval_epoch):
                print()
                print('Calculating accuracy...',end='\r')
                print(f'Accuracy: {self.accuracy(X,y) * 100}%           ')
            
            print()
            print('Epoch complete')
            print('='*100)
        return self
    
    def cost_derivative(self, output_activations, y):
        cost_delta = []
        for output, t_y in zip(output_activations, y):
            cost_delta.append(output - t_y)
        return np.array(cost_delta)

    def output_layer(self):
        return self.activations[-1]
    
    def average_cost(self, X, y):
        cost_sum = 0.0
        count = 0.0
        for i in range(len(X)):
            self.feedforward(X[i])
            cost_sum += cost(self.output_layer(), y[i])
            count += 1.0
        return (cost_sum / count)
    
    def predict(self,input):
        output = self.feedforward(input)
        values = []
        for value in output:
            values.append(value[0])
        confidence = max(values)
        return (values.index(confidence), confidence)

    def accuracy(self, X, y, show=False):
        correct = 0
        for i in range(len(X)):
            predicted = self.predict(X[i])
            actual = y[i]
            if show:
                print(f'Predicted: {predicted[0]}, confidence: {predicted[1]}\nActual: {actual}')
            if predicted[0] == actual:
                correct += 1
        return correct / len(X)
    
    def mutate(self, effect):
        self.weights = [ws * (np.random.uniform(-1.0, 1.0 , ws.shape) * effect) for ws in self.weights]
        self.biases = [bs * (np.random.uniform(-1.0, 1.0 , bs.shape) * effect) for bs in self.biases]

    def to_string(self):
        string = ''
        for weights in self.weights:
            for ws in weights:
                for w in ws:
                    string += f'{w},'
                string = string[:-1]
                string += '|'
            string = string[:-1]
            string += '\n'
        
        string += 'Bias:\n'

        for biases in self.biases:
            for bs in biases:
                for b in bs:
                    string += f'{b},'
                string = string[:-1]
                string += '|'
            string = string[:-1]
            string += '\n'

        return string[:-1]
    
    def from_string(self, lines):
        # Load weights and biases from list of strings called lines
        # Format for lines described in to_string
        in_bias = False
        biases = []
        weights = []
        for line in lines:
            if in_bias:
                bs = line.split('|')
                t_biases = []
                for b in bs:
                    t_biases.append([float(b)])
                biases.append(np.array(t_biases))
            elif line[:5] != 'Bias:':
                t_weights = line.split('|')
                layer_weights = []
                for t_ws in t_weights:
                    ws = t_ws.split(',')
                    t_weight = []
                    for w in ws:
                        t_weight.append(float(w))
                    layer_weights.append(np.array(t_weight))
                weights.append(np.array(layer_weights))

            if line[:5] == 'Bias:':
                in_bias = True
        
        self.weights = weights
        self.biases = biases
    
    def save_json(self, filename):
        """
        Save network architecture, weights, and biases to a JSON file.
        """
        string_functions = [func.__name__ for func in self.activation_functions]
        string_derivatives = [func.__name__ for func in self.activation_derivatives]

        data = {
            "sizes": self.sizes,
            "activations" : string_functions,
            "derivatives" : string_derivatives,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }

        if filename[-5:] != '.json':
            filename += '.json'

        with open(filename, "w") as f:
            json.dump(data, f)

    def load_json(self, filename, activation_functions = [], activation_derivatives = []):
        """
        Load network architecture, weights, and biases from a JSON file.
        """
        if filename[-5:] != '.json':
            filename += '.json'

        with open(filename, "r") as f:
            data = json.load(f)

        self.sizes = data["sizes"]

        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]

        string_functions = ''
        functions = []
        derivatives = []
        for func in data["activations"]:
            try:
                functions.append(getattr(sys.modules[__name__], func))
            except:
                found = False
                for act in activation_functions:
                    if func == act.__name__:
                        functions.append(act)
                        found = True
                        break
                if not found:
                    string_functions += f'{func}, '
        
        for func in data["derivatives"]:
            try:
                derivatives.append(getattr(sys.modules[__name__], func))
            except:
                found = False
                for act in activation_derivatives:
                    if func == act.__name__:
                        derivatives.append(act)
                        found = True
                        break
                if not found:
                    string_functions += f'{func}, '

        if string_functions != '':
            string_functions = string_functions[:-2]
            raise Exception(f'Could not find functions for {string_functions}, provide these functions in the activation_functions and activation_derivatives parameters')
        
        self.activation_functions = functions
        self.activation_derivatives = derivatives

def cost(output, expected_output):
    cost = 0.0
    for i in range(len(output)):
        cost += (output[i] - expected_output[i]) ** 2.0
    return cost

def save_to_file(data, file):
    with open(file, 'w') as f:
        f.write(data)

def read_from_file(file):
    with open(file,'r') as f:
        return f.readlines()