import cv2
from PIL import Image, ImageFilter
import numpy as np

def apply_low_pass_filter(image, kernel_size):
    """
    Applies a low pass filter (Box Blur) to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the filter.
    
    Returns:
    PIL.Image: The filtered image.
    """
    return image.filter(ImageFilter.BoxBlur(kernel_size))

def apply_hpf(image, kernel_size):
    """
    Applies a high pass filter to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the Gaussian blur.
    
    Returns:
    PIL.Image: The filtered image.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    hpf_image = cv2.subtract(gray_image, blurred_image)
    return Image.fromarray(hpf_image)

def apply_mean_filter(image, kernel_size):
    """
    Applies a mean filter to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the filter.
    
    Returns:
    PIL.Image: The filtered image.
    """
    mean_image = cv2.blur(np.array(image), (kernel_size, kernel_size))
    return Image.fromarray(mean_image)

def apply_median_filter(image, kernel_size):
    """
    Applies a median filter to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the filter.
    
    Returns:
    PIL.Image: The filtered image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    median_image = cv2.medianBlur(np.array(image), kernel_size)
    return Image.fromarray(median_image)

def apply_roberts_edge_detector(image, _):
    """
    Applies the Roberts edge detection algorithm to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The edge-detected image.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    roberts_image = cv2.Canny(gray_image, 100, 200)
    return Image.fromarray(roberts_image)

def apply_prewitt_edge_detector(image, _):
    """
    Applies the Prewitt edge detection algorithm to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The edge-detected image.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(gray_image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    return Image.fromarray(prewitt_edges)

def apply_sobel_edge_detector(image, _):
    """
    Applies the Sobel edge detection algorithm to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The edge-detected image.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_image = cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(sobel_image)
