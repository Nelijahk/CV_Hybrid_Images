import math
import numpy as np
from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
        by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
        a high-pass image created by subtracting highImage from highImage convolved with
        a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    # gaussian kernels with different standard deviations (sigma values)
    low_kernel = makeGaussianKernel(lowSigma)
    high_kernel = makeGaussianKernel(highSigma)

    low_pass = convolve(lowImage, low_kernel)                                   # the low-pass filtered image

    blurred_high = convolve(highImage, high_kernel)                             # blurred version for high-pas image
    high_pass = highImage.astype(np.float64) - blurred_high                     # subtraction of blurred version

    hybrid_image = low_pass + high_pass                                         # combination of the low-pass and high-pass images
    return np.clip(hybrid_image, 0, 255).astype(np.uint8)                       # clipping values to the 0-255 range


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = math.floor(8 * sigma + 1)                                            # kernel size based on the given formula
    if size % 2 == 0:
        size += 1                                                               # transformation in odd number

    kernel = np.zeros((size, size), dtype=np.float64)                           # empty kernel
    center = size // 2                                                          # center of the kernel
    sum_total = 0.0                                                             # for normalization

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center                                       # relative position from the center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))            # gaussian function
            sum_total += kernel[i, j]

    # normalization of the kernel so that the sum of all elements is 1
    return kernel / sum_total
