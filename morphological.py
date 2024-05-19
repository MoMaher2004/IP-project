import cv2
import numpy as np
from PIL import Image

def apply_erosion(image, kernel_size):
    """
    Applies erosion to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the erosion.
    
    Returns:
    PIL.Image: The eroded image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion_image = cv2.erode(np.array(image), kernel, iterations=1)
    return Image.fromarray(erosion_image)

def apply_dilation(image, kernel_size):
    """
    Applies dilation to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the dilation.
    
    Returns:
    PIL.Image: The dilated image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation_image = cv2.dilate(np.array(image), kernel, iterations=1)
    return Image.fromarray(dilation_image)

def apply_open(image, kernel_size):
    """
    Applies the open morphological operation to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the operation.
    
    Returns:
    PIL.Image: The image after opening.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    open_image = cv2.morphologyEx(np.array(image), cv2.MORPH_OPEN, kernel)
    return Image.fromarray(open_image)

def apply_close(image, kernel_size):
    """
    Applies the close morphological operation to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    kernel_size (int): The kernel size for the operation.
    
    Returns:
    PIL.Image: The image after closing.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    close_image = cv2.morphologyEx(np.array(image), cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(close_image)
