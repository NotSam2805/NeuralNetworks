import numpy as np
from matplotlib import pyplot
import math

sharpen_kernel = np.array([[0.00,0.00,-0.5,0.00,0.00],
                           [0.00,-0.5,-0.5,-0.5,0.00],
                           [-0.5,-0.5,6.00,-0.5,-0.5],
                           [0.00,-0.5,-0.5,-0.5,0.00],
                           [0.00,0.00,-0.5,0.00,0.00]])

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

    for i in range(image_height):
        min_height = max(0,i - int(kernel_height/2))
        max_height = min(image_height -  1, i + int(kernel_height/2) + 1)
        for j in range(image_width):
            min_width = max(0,j - int(kernel_width/2))
            max_width = min(image_width - 1,j + int(kernel_width/2) + 1)
            #print(f'Height: {min_height} - {max_height}, width: {min_width} - {max_width}')
            selection = image[min_width:max_width,min_height:max_height]
            #print(selection.shape)
            if selection.shape != kernel.shape:
                k = kernel[0:selection.shape[0], 0:selection.shape[1]]
                output[j,i] = np.sum(np.multiply(selection, k))
            else:
                output[j,i] = np.sum(np.multiply(selection, kernel))
    
    if (normalised):
        return normalise_image(output)

    return output

def max_pool_image(image, pooling_size, stride = None, normalised = False):
    if(normalised):
        image = denormalise_image(image)
    
    if (stride == None):
        stride = pooling_size[0]

    output_width = int(image.shape[0] / pooling_size[0])
    output_height = int(image.shape[1] / pooling_size[1])
    output = np.zeros((output_width, output_height))

    for i in range(0,output_height):
        min_y = i * stride
        max_y = min_y + stride
        for j in range(0,output_width):
            min_x = j * stride
            max_x = min_x + stride
            selection = image[min_x:max_x, min_y:max_y]
            output[j,i] = np.max(selection)
    
    if (normalised):
        return normalise_image(output)

    return output

def uniform_kernel(width, height):
    n = width * height
    value = 1.0 / n
    return np.full((width, height), value)

def vertical_edge_kernel(width, height):
    kernel = np.zeros((width, height))
    mid = int(width/2)
    for i in range(height):
        for j in range(width):
            value = (j/mid) - 1
            value = value / height
            kernel[i,j] = value
    return kernel

def horizontal_edge_kernel(width, height):
    kernel = np.zeros((width, height))
    mid = int(height/2)
    for i in range(height):
        for j in range(width):
            value = (i/mid) - 1
            value = value / width
            kernel[i,j] = value
    return kernel

def random_kernel(width, height):
    return np.random.uniform(-1,1,(width,height))

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

def test():
    import mnist_data as mnist

    X, y = mnist.test_data()
    image = X[0]
    show_image(image)

    sharpened = convolve_image(image,sharpen_kernel)
    show_image(sharpened)

    k = vertical_edge_kernel(3,3)
    vert_edges = convolve_image(sharpened, k)
    show_image(vert_edges)

    k = horizontal_edge_kernel(3,3)
    hori_edges = convolve_image(sharpened, k)
    show_image(hori_edges)

    pooled = max_pool_image(vert_edges, (2,2))
    show_image(pooled)