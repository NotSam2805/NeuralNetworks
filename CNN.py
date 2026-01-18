import Convolution as c
import numpy as np
import networks as n

class c_layer:

    def __init__(self, layer_size, pool_size = (1,1), channels = 1, activation = n.relu, kernels = None):
        self.size = layer_size
        self.activation = activation
        self.pool_size = pool_size
        self.channels = channels
        self.biases = np.random.uniform(-1,1,(layer_size[0]))

        if(kernels == None):
            if channels > 1:
                self.kernels = [c.random_kernel((channels,layer_size[1],layer_size[2])) for i in range(layer_size[0])]
            else:
                self.kernels = [c.random_kernel((layer_size[1],layer_size[2])) for i in range(layer_size[0])]
        elif(len(kernels) != layer_size[0]):
            raise Exception('kernel initialisation error')
        else:
            self.kernels = kernels
    
    def feedforward(self, x):
        outputs = []
        for kernel, bias in zip(self.kernels, self.biases):
            if self.channels > 1:
                convolved = c.convolve_image_channels(x,kernel,self.channels)
            else:
                convolved = c.convolve_image(x, kernel)
            activated = self.activation(np.add(convolved, bias))
            pooled = c.max_pool_image(activated, self.pool_size)
            outputs.append(pooled)
        
        return np.array(outputs)

class CNN:
    def __init__(self, c_layers, fc_network):
        self.c_layers = c_layers
        self.fc_network = fc_network

    def feedforward(self, x):
        out = x
        for layer in self.c_layers:
            out = layer.feedforward(out)
        
        norm = normalise(out,1)

        fc_out = self.fc_network.feedforward(norm)

        return fc_out
    
    def predict(self, input):
        output = self.feedforward(input)
        values = []
        for value in output:
            values.append(value[0])
        confidence = max(values)
        return (values.index(confidence), confidence)

def normalise(data, factor=255.0):
    size = data.size
    
    output = np.reshape(data,(size,1))
    output = np.divide(output, factor)

    return output

def test():
    layer1 = c_layer((2,5,5), (2,2), activation=n.relu)
    layer2 = c_layer((4,3,3), (2,2), channels=2, activation=n.sigmoid)
    fc_layer = n.N_Network([], 7*7*4, 10, activation_functions=[n.softmax])

    net = CNN([layer1,layer2],fc_layer)

    import mnist_data as mnist

    X = mnist.test_X
    image = X[0]
    predict = net.predict(image)

    print(predict)
    c.show_image(image)

    #for image in out:
    #    if (len(image.shape) > 2):
    #        for channel in image:
    #            c.show_image(channel)
    #    else:
    #        c.show_image(image)


test()