import cv2
import numpy as np
from PIL import Image

def apply_hough_circle_transform(image, _):
    """
    Applies Hough Circle Transform to detect circles in the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The image with detected circles highlighted.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        hough_image = np.array(image).copy()
        for i in circles[0, :]:
            cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
        return Image.fromarray(hough_image)
    return image

def apply_region_split_merge_segmentation(image, _):
    """
    Applies region split and merge segmentation to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The segmented image.
    """
    region_split_merge_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    height, width = region_split_merge_image.shape

    def region_growing(image, seed):
        visited = set()
        stack = [seed]
        while stack:
            x, y = stack.pop()
            if (x, y) not in visited:
                visited.add((x, y))
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                            if abs(int(image[ny, nx]) - int(image[y, x])) < 10:
                                stack.append((nx, ny))
        return visited

    seeds = [(int(width / 2), int(height / 2))]
    for seed in seeds:
        region = region_growing(region_split_merge_image, seed)
        for (x, y) in region:
            region_split_merge_image[y, x] = 255

    return Image.fromarray(region_split_merge_image)

def apply_thresholding_segmentation(image, _):
    """
    Applies thresholding segmentation to the image.
    
    Parameters:
    image (PIL.Image): The input image.
    _ (int): Unused parameter.
    
    Returns:
    PIL.Image: The segmented image.
    """
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(threshold_image)
