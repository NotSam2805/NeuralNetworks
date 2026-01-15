import numpy as np
from matplotlib import pyplot
import math

def denormalise_image(image, factor = 255.0):
    image = np.multiply(image, factor)
    size = math.sqrt(image.size)
    return image.reshape((size,size))

def normalise_image(image, factor = 255.0):
    size = image.shape[0] * image.shape[1]
    image = image.reshape((size))
    return np.divide(image, factor)

def show_image(image, normalised=False):
    pyplot.clf()
    if normalised:
        image = denormalise_image(image)
    if np.min(image) < 0.0:
        image = np.multiply(image, 0.5)
        image = np.add(image, 255.0/2.0)
    pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
    pyplot.autoscale()
    pyplot.show()

def convolve_image(image, kernel, normalised=False):
    if(normalised):
        image = denormalise_image(image)
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
    
    if (normalised):
        return normalise_image(output)

    return output

def max_pool_image(image, pooled_size, normalised = False):
    if(normalised):
        image = denormalise_image(image)
    
    output_width, output_height = pooled_size
    output = np.zeros(pooled_size)
    factor_x = int(image.shape[0] / output_width)
    factor_y = int(image.shape[1] / output_height)

    for i in range(output_width):
        min_x = i * factor_x
        max_x = min_x + factor_x
        for j in range(output_height):
            min_y = j * factor_y
            max_y = min_y + factor_y
            selection = image[min_x:max_x, min_y:max_y]
            output[i,j] = np.max(selection)
    
    if (normalised):
        return normalise_image(output)

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

def average_image(images):
    output = np.zeros(images[0].shape)
    height = output.shape[1]
    width = output.shape[0]

    count = 0.0
    for image in images:
        count += 1.0
        for i in range(width):
            for j in range(height):
                output[i,j] += image[i,j]
    
    np.divide(output, count)
    return output

import mnist_data as mnist

sharpen_kernel = np.array([[0.00,0.00,-0.5,0.00,0.00],
                           [0.00,-0.5,-0.5,-0.5,0.00],
                           [-0.5,-0.5,6.00,-0.5,-0.5],
                           [0.00,-0.5,-0.5,-0.5,0.00],
                           [0.00,0.00,-0.5,0.00,0.00]])

X, y = mnist.test_data()
image = X[1]
show_image(image)

sharpened = convolve_image(image,sharpen_kernel)
show_image(sharpened)

k = vertical_edge_kernel(3,3)
convolved = convolve_image(sharpened, k)
show_image(convolved)

pooled = max_pool_image(convolved, (14,14))
show_image(pooled)