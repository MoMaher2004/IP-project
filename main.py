# Made By:
#   Mohamed Ahmed Mohamed Maher - Level 1
#   Yousef Fathy Ibrahim Shalaby - Level 1

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from edge_detection import apply_low_pass_filter, apply_mean_filter, apply_median_filter
from edge_detection import apply_hpf, apply_roberts_edge_detector, apply_prewitt_edge_detector, apply_sobel_edge_detector
from morphological import apply_erosion, apply_dilation, apply_open, apply_close
from others import apply_hough_circle_transform, apply_region_split_merge_segmentation, apply_thresholding_segmentation

# Function to upload an image and display it
def upload_image():
    """
    Opens a file dialog for the user to select an image file, 
    resizes it to fit in the window, and returns the image object.
    """
    file_path = filedialog.askopenfilename()  # Open file dialog to choose an image
    if file_path:
        original_image = Image.open(file_path)  # Open the chosen image
        original_image.thumbnail((600, 400))  # Resize image to fit in the window
        return original_image
    return None

# Function to apply the selected filter to the image
def apply_filter(filter_name, original_image, slider_value):
    """
    Applies the selected filter to the given image.
    
    Parameters:
    filter_name (str): The name of the filter to apply.
    original_image (PIL.Image): The original image object.
    slider_value (int): The value from the slider, used as a parameter for some filters.
    
    Returns:
    PIL.Image: The filtered image.
    """
    filter_map = {
        "Low Pass Filter": apply_low_pass_filter,
        "High Pass Filter": apply_hpf,
        "Mean Filter": apply_mean_filter,
        "Median Filter": apply_median_filter,
        "Robert Filter": apply_roberts_edge_detector,
        "Prewitt Filter": apply_prewitt_edge_detector,
        "Sobel Filter": apply_sobel_edge_detector,
        "Erosion": apply_erosion,
        "Dilation": apply_dilation,
        "Open": apply_open,
        "Close": apply_close,
        "Hough Transform for Circle": apply_hough_circle_transform,
        "Region Split Merge Segmentation": apply_region_split_merge_segmentation,
        "Thresholding Segmentation": apply_thresholding_segmentation,
    }
    
    if filter_name in filter_map:
        return filter_map[filter_name](original_image, slider_value)  # Apply the selected filter
    return original_image  # Return the original image if filter is not found

# Function to update the displayed image
def update_image(image_label, image):
    """
    Updates the image displayed in the GUI.
    
    Parameters:
    image_label (tk.Label): The label widget to display the image.
    image (PIL.Image): The image to display.
    """
    photo = ImageTk.PhotoImage(image)  # Convert image to PhotoImage
    image_label.config(image=photo)  # Update label with the new image
    image_label.image = photo  # Keep a reference to avoid garbage collection

# Main function to create the GUI
def main():
    """
    Creates and runs the main application GUI.
    """
    root = tk.Tk()  # Create the main window
    root.title("Filters")  # Set the title of the window
    root.geometry("800x600")  # Set the size of the window
    
    # Add label with names and levels at the top
    info_label = tk.Label(root, text="Mohamed Ahmed Mohamed Maher - level 1\nYousef Fathy Ibrahim Shalaby - level 1", font=("Arial", 14), pady=10)
    info_label.pack()
    
    # Function to handle the upload button click
    def on_upload():
        nonlocal original_image  # Use the nonlocal variable
        original_image = upload_image()  # Call the upload_image function
        if original_image:
            update_image(image_label, original_image)  # Update the image display
    
    # Function to apply the selected filter
    def apply_selected_filter():
        selected_filter = filter_var.get()  # Get the selected filter from the dropdown
        if original_image:
            filtered_image = apply_filter(selected_filter, original_image, slider.get())  # Apply the filter
            update_image(image_label, filtered_image)  # Update the image display
    
    # Create and place the upload button
    upload_button = tk.Button(root, text="Upload", command=on_upload, bg="blue", fg="white")
    upload_button.pack(pady=20)
    
    # Create and place the label to display the image
    image_label = tk.Label(root)
    image_label.pack()
    
    # Create and place the frame for filter selection
    filters_frame = tk.Frame(root)
    filters_frame.pack(pady=20)
    
    original_image = None  # Initialize the original image variable
    
    # List of available filters
    filters = [
        "Low Pass Filter",
        "High Pass Filter",
        "Mean Filter",
        "Median Filter",
        "Robert Filter",
        "Prewitt Filter",
        "Sobel Filter",
        "Erosion",
        "Dilation",
        "Open",
        "Close",
        "Hough Transform for Circle",
        "Region Split Merge Segmentation",
        "Thresholding Segmentation",
    ]
    
    # Create and place the frame for the kernel size slider
    slider_frame = tk.Frame(root)
    slider_frame.pack(pady=20)
    
    # Create and place the label for the slider
    slider_label = tk.Label(slider_frame, text="Kernel Size:")
    slider_label.grid(row=0, column=0, padx=10)
    
    # Create and place the slider
    slider = tk.Scale(slider_frame, from_=1, to=20, orient="horizontal", length=200)
    slider.set(5)
    slider.grid(row=0, column=1)
    
    # Create and place the dropdown for filter selection
    filter_var = tk.StringVar(root)
    filter_var.set(filters[0])  # Set default value to the first filter
    
    filter_dropdown = tk.OptionMenu(filters_frame, filter_var, *filters)
    filter_dropdown.pack(side="left", padx=10)
    
    # Create and place the button to apply the selected filter
    apply_button = tk.Button(filters_frame, text="Apply Filter", command=apply_selected_filter)
    apply_button.pack(side="left", padx=10)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
