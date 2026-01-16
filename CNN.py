import Convolution as c
import numpy as np
import networks as n

class c_layer:

    def __init__(self, layer_size, pool_size = (1,1), channels = 1, activation = n.relu, kernels = []):
        self.size = layer_size
        self.activation = activation
        self.pool_size = pool_size
        self.channels = channels
        self.biases = np.random.uniform(-1,1,(layer_size[0]))

        if(kernels == []):
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
        
def normalise(data, factor=255.0):
    size = 1
    for s in data.shape:
        size = size * s
    
    output = np.reshape(data,(size))
    output = np.divide(output, factor)

    return output

def test():
    layer1 = c_layer((2,5,5), (2,2), activation=n.relu)
    layer2 = c_layer((4,3,3), (2,2), channels=2, activation=n.sigmoid)
    fc_layer = n.N_Network([], 7*7*4, 10, activation_functions=[n.softmax])

    import mnist_data as mnist

    X = mnist.test_X

    out = layer1.feedforward(X[0])
    out = layer2.feedforward(out)
    norm = normalise(out)
    predicted = fc_layer.predict(norm)

    print(predicted)

    #for image in out:
    #    if (len(image.shape) > 2):
    #        for channel in image:
    #            c.show_image(channel)
    #    else:
    #        c.show_image(image)


test()