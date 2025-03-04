import cv2
import glob
import os
import numpy as np

images = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

# Retrieve a list of image file paths with .jpg extension
image_files = glob.glob(os.path.join(images, "*.jpg"))

# Check if any images are found
if image_files:
    # Select the first image from the list
    image_path = image_files[0]
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is not None:
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
        
        # Display the image
        cv2.imshow("HSV Image", hsv_image)
        
        # Wait for a key press and close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to load image at {image_path}")
else:
    print("No images found in the specified directory.")
