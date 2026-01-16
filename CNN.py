import Convolution as c
import numpy as np
import networks as n

class c_layer:

    def __init__(self, size, activation = n.relu, kernels = []):
        self.size = size
        self.activation = activation
        self.biases = np.random.uniform(-1,1,(size[0]))

        if(kernels == []):
            self.kernels = [c.random_kernel(size[1],size[2]) for i in range(size[0])]
        elif(len(kernels) != size[0]):
            raise Exception('kernel initialisation error')
        else:
            self.kernels = kernels
    
    def feedforward(self, x):
        outputs = []
        for kernel, bias in zip(self.kernels, self.biases):
            convolved = c.convolve_image(x, kernel)
            outputs.append(self.activation(np.add(convolved,bias)))
        
        return np.array(outputs)

def test():
    layer = c_layer((2,5,5), activation=n.relu)

    import mnist_data as mnist

    X = mnist.test_X

    out = layer.feedforward(X[0])

    for image in out:
        c.show_image(image)

test()