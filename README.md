#Neural Networks
Some stuff that I've been making to help me learn about the workings of Neural Networks.
Used MNIST digit images.

In some places I used "normalised" data, this refers to data in 1D arrays with values between 0 and 1. For example a single image from the MNIST dataset is a 2D array of shape (28,28) with values from 0 to 255, this normalised becomes an array of shape (784) with all values divided by 255.

##File formatting for reference
SOTF_1,SGD_1.txt and SGD_2.txt are networks of [16,16] layers
SGD_3.txt is [64,64,32,32]
SGD_4.txt is [128,64,32]

For ideal MNIST structure (as in 'ReLu_Softmax_test1.txt'), use n.N_Network([128,64,32], 28 * 28, 10, [n.relu, n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime, n.relu_prime])
note that the lists of functions (and the derivatives) is important, the other parameters matter less 
for 'ReLu_Softmax_best_1.txt' use n.N_Network([128,64], 28 * 28, 10, [n.relu, n.relu, n.softmax], [n.relu_prime, n.relu_prime])

Moved to using JSON for ease, dont use .txt anymore. (the models saved to .txt's arent good anyway)
