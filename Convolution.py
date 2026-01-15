import numpy as np
from matplotlib import pyplot
import mnist_data as mnist

def show_image(image, normalised=False):
    pyplot.clf()
    if normalised:
        image = np.multiply(image, 255.0)
        image = image.reshape((28,28))
    if np.max(image) <= 1.0:
        image = np.multiply(image, 255.0)
    if np.min(image) < 0.0:
        image = np.multiply(image, 0.5)
        image = np.add(image, 255.0/2.0)
    pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
    pyplot.autoscale()
    pyplot.show()

def convolve_image(image, kernel, normalised=False):
    if(normalised):
        #image = np.multiply(image, 255.0)
        image = image.reshape((28,28))
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros(image.shape)

    for i in range(0,image_height):
        max_height = i + kernel_height
        for j in range(0,image_width):
            max_width = j + kernel_width
            selection = image[i:max_height, j:max_width]
            if selection.shape != kernel.shape:
                k = kernel[0:selection.shape[0], 0:selection.shape[1]]
                output[i,j] = np.sum(np.multiply(selection, k))
            else:
                output[i,j] = np.sum(np.multiply(selection, kernel))
    
    return output

def uniform_kernel(width, height):
    n = width * height
    value = 1.0 / n
    return np.full((height, width), value)

def vertical_edge_kernel(width, height):
    kernel = np.zeros((width, height))
    mid = int(width/2)
    for i in range(height):
        for j in range(width):
            value = (j/mid) - 1
            value = value / height
            kernel[i,j] = value
    return kernel

def horizontal_edge_kernel(width,height):
    kernel = np.zeros((width, height))
    mid = int(height/2)
    for i in range(height):
        for j in range(width):
            value = (i/mid) - 1
            value = value / width
            kernel[i,j] = value
    return kernel

X, y = mnist.test_data()
image = X[0]

show_image(image)

k = horizontal_edge_kernel(3,3)
convolved = convolve_image(image, k)
show_image(convolved)