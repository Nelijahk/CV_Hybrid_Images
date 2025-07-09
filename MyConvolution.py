import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd) :
    type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """

    if len(kernel.shape) != 2:
        raise ValueError("ERROR: wrong input: kernel is not correct shape")

    kheight, kwidth = kernel.shape              # getting kernel dimensions
    pad_y, pad_x = kheight // 2, kwidth // 2    # padding sizes - half of the kernel size along each dimension - to ensure that output is the same size as input
    flipped_kernel = np.flip(kernel)            # flipping the kernel for convolution

    rows, cols, channels, padded_image, convolved_image = 0, 0, 1, [], []
    if len(image.shape) == 2:
        rows, cols = image.shape                # Grayscale image

        # padding the image with zeros to maintain output size and handle border pixels
        padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), 'constant')

        convolved_image = np.zeros((rows, cols))  # initialization of the output image with zeros
    elif len(image.shape) == 3:
        rows, cols, channels = image.shape      # Colourful image
        padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
        convolved_image = np.zeros((rows, cols, channels))
    else:
        raise ValueError("ERROR: wrong input: image is not correct shape")

    for i in range(pad_y, rows + pad_y):      # iterating over rows
        for j in range(pad_x, cols + pad_x):  # iterating over columns
            if channels == 1:
                window = padded_image[i - pad_y:i + pad_y + 1, j - pad_x:j + pad_x + 1]  # current region of convolution from the padded image
                convolved_image[i - pad_y, j - pad_x] = np.sum(window * flipped_kernel)  # sum of element-wise multiplication
            else:
                for c in range(channels):                                                # process each channel separately
                    window = padded_image[i - pad_y:i + pad_y + 1, j - pad_x:j + pad_x + 1, c]
                    convolved_image[i - pad_y, j - pad_x, c] = np.sum(window * flipped_kernel)
    return convolved_image
