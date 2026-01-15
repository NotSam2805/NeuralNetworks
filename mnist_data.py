print('Loading MNIST data...', end='\r')
from keras.datasets import mnist
import numpy as np


(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('MNIST data loaded    ')

def normalise_values(data):
    data = data/255.0
    return data

def denormalise_values(data):
    data = np.array(data)
    data = data * 255.0
    return data

def training_data():
    return (train_X, train_y)

def test_data():
    return (test_X, test_y)

def normalise_data_shape(dataset):
    X = []
    for a in range(dataset.shape[0]):
        X.append([])
        for b in range(dataset[a].shape[0]):
            for c in range(dataset[a][b].size):
                X[a].append(dataset[a][b][c])
    return X

def denormalise_data_shape(dataset):
    data = np.array(dataset)
    return data.reshape((data.shape[0],28,28))

def denormalise_data(dataset):
    dataset = denormalise_data_shape(dataset)
    return denormalise_values(dataset)

def normalise_data(dataset):
    dataset = normalise_values(dataset)
    return normalise_data_shape(dataset)

def normalised_training_data():
    print('Normalising training data...', end='\r')
    X = normalise_data(train_X)
    print('Data normalised              ')
    return (X, train_y)

def normalised_test_data():
    print('Normalising test data...', end='\r')
    X = normalise_data(test_X)
    print('Data normalised              ')
    return (X, test_y)

def label_to_output_layer(label):
    output = []
    for i in range(10):
        if(i == label):
            output.append(1.0)
        else:
            output.append(0.0)
    return output

def expected_outputs(labels):
    expected_outputs = []
    for label in labels:
        expected_outputs.append(label_to_output_layer(label))
    return expected_outputs

def convert_to_list_of_tuples(X,y):
    tuples = []
    for i in range(len(X)):
        t = (X[i], y[i])
        tuples.append(t)
    return tuples

########################################
#print('X_train: ' + str(train_X.shape))
#print('Y_train: ' + str(train_y.shape))
#print('X_test:  '  + str(test_X.shape))
#print('Y_test:  '  + str(test_y.shape))

#from matplotlib import pyplot
#for i in range(9):  
#    pyplot.subplot(330 + 1 + i)
#    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#pyplot.show()
#######################
##########################################