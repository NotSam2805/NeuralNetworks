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

def convolve_valid(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Valid 2D cross-correlation (commonly called convolution in CNNs).
    image:  (H, W)
    kernel: (K, K)
    stride: int >= 1
    returns: (H_out, W_out) where
        H_out = (H - K)//stride + 1
        W_out = (W - K)//stride + 1
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("convolve_valid expects 2D arrays: image (H,W), kernel (K,K)")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    H, W = image.shape
    K1, K2 = kernel.shape
    if K1 != K2:
        raise ValueError("kernel must be square (K,K)")
    K = K1

    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError("kernel is larger than image for valid convolution")

    out = np.zeros((H_out, W_out), dtype=float)

    # Cross-correlation (no kernel flip) — standard in CNNs
    for i_out, i in enumerate(range(0, H - K + 1, stride)):
        for j_out, j in enumerate(range(0, W - K + 1, stride)):
            patch = image[i:i+K, j:j+K]
            out[i_out, j_out] = np.sum(patch * kernel)

    return out

def convolve_valid_channels(image: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    image:  (C, H, W)
    kernel: (C, K, K)
    returns: (H_out, W_out)
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim != 3 or kernel.ndim != 3:
        raise ValueError("expects image (C,H,W) and kernel (C,K,K)")
    if image.shape[0] != kernel.shape[0]:
        raise ValueError("image and kernel must have same channel count")

    out = None
    for c in range(image.shape[0]):
        conv_c = convolve_valid(image[c], kernel[c], stride=stride)
        out = conv_c if out is None else (out + conv_c)
    return out

def convolve_image(image, kernel, stride = 1, normalised=False):
    if(normalised):
        image = denormalise_image(image)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height//stride + 1
    output_width = image_width//stride + 1

    output = np.zeros((output_height, output_width))
    #print(output.shape)

    for i in range(0, image_height, stride):
        min_height = max(0,i - int(kernel_height/2))
        max_height = min(image_height -  1, i + int(kernel_height/2) + 1)
        for j in range(0, image_width, stride):
            min_width = max(0,j - int(kernel_width/2))
            max_width = min(image_width - 1,j + int(kernel_width/2) + 1)
            #print(f'Height: {min_height} - {max_height}, width: {min_width} - {max_width}')
            selection = image[min_height:max_height, min_width:max_width]
            #print(selection.shape)
            if selection.shape != kernel.shape:
                k = kernel[0:selection.shape[0], 0:selection.shape[1]]
                output[int(i/stride), int(j/stride)] = np.sum(np.multiply(selection, k))
            else:
                output[int(i/stride), int(j/stride)] = np.sum(np.multiply(selection, kernel))
    
    if (normalised):
        return normalise_image(output)

    return output

def convolve_image_channels(image, kernel, channels, normalised=False):
    output = np.zeros(image.shape[1:])
    for channel in range(channels):
        convolved = convolve_image(image[channel], kernel[channel], normalised=normalised)
        output = np.add(output, convolved)
    return output

def pool_image(image, pooling_size, pool_func, stride = None, normalised = False):
    if(normalised):
        image = denormalise_image(image)
    
    if (stride == None):
        stride = pooling_size[0]

    output_width = int(image.shape[0] / stride)
    output_height = int(image.shape[1] / stride)
    output = np.zeros((output_width, output_height))

    for i in range(0,output_height):
        min_y = i * stride
        max_y = min_y + pooling_size[0]
        for j in range(0,output_width):
            min_x = j * stride
            max_x = min_x + pooling_size[1]
            selection = image[min_y:max_y, min_x:max_x]
            output[i,j] = pool_func(selection)
    
    if (normalised):
        return normalise_image(output)

    return output

def max_pool_image(image, pooling_size, stride = None, normalised = False):
    return pool_image(image, pooling_size, np.max, stride, normalised)

def min_pool_image(image, pooling_size, stride = None, normalised = False):
    return pool_image(image, pooling_size, np.min, stride, normalised)

def average_pool_image(image, pooling_size, stride = None, normalised = False):
    return pool_image(image, pooling_size, np.average, stride, normalised)

def uniform_kernel(width, height, channels = 1):
    n = width * height * channels
    value = 1.0 / n
    if channels > 1:
        return np.full((channels,width,height), value)
    return np.full((width,height), value)

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

def random_kernel(size):
    return np.random.uniform(-1,1,size)

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

    k = vertical_edge_kernel(3,3)
    vert_edge = convolve_valid(image, k)
    show_image(vert_edge)

test()