import Convolution as c
import numpy as np

class c_layer:

    def __init__(self, size, kernels = []):
        self.size = size

        if(kernels == []):
            self.kernels = [np.zeros((size[1],size[2])) for i in range(size[0])]
        elif(len(kernels) != size[0]):
            print('kernel initialisation error')
        else:
            self.kernels = kernels
    
    def feedforward(self, x):
        outputs = []
        for kernel in self.kernels:
            outputs.append(c.convolve_image(x,kernel))
        
        return np.array(outputs)

def test():
    kernels = [c.horizontal_edge_kernel(3,3), c.vertical_edge_kernel(3,3), c.sharpen_kernel]
    layer = c_layer((3,3,3), kernels)

    import mnist_data as mnist

    X = mnist.test_X

    out = layer.feedforward(X[0])

    for image in out:
        c.show_image(image)

test()