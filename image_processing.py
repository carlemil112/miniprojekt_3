import cv2
import glob
import os
import numpy as np

# Directory containing images
images_dir = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

# Retrieve a list of image file paths with .jpg extension
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

# Check if any images are found
if not image_files:
    print("No images found in the specified directory.")
    exit()

# Select the first image from the list
image_path = image_files[0]  # Change index if needed

# Read the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Dictionary of color spaces
color_spaces = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "LAB": cv2.COLOR_BGR2Lab,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "Grayscale": cv2.COLOR_BGR2GRAY
}

def modify_channel(img, channel_index, value):
    """ Modify a specific channel in an image """
    img = img.astype(np.float32)
    img[:, :, channel_index] = np.clip(img[:, :, channel_index] + value, 0, 255)
    return img.astype(np.uint8)

def process_image(color_space):
    """ Convert image to the selected color space """
    if color_space in color_spaces:
        converted_img = cv2.cvtColor(image, color_spaces[color_space])
    else:
        converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return converted_img

def interactive_editor(color_space):
    converted = process_image(color_space)

    if len(converted.shape) == 2:  # Grayscale case (single channel)
        cv2.imshow(f"{color_space} Image", converted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    modified = converted.copy()
    
    def update(val):
        nonlocal modified
        for i in range(3):
            modified[:, :, i] = modify_channel(converted, i, cv2.getTrackbarPos(f'Channel {i}', 'Editor'))
        cv2.imshow("Editor", modified)
        cv2.waitKey(1)  # Prevents freezing

    cv2.namedWindow("Editor", cv2.WINDOW_GUI_NORMAL)  # Allow resizing
    cv2.setWindowProperty("Editor", cv2.WND_PROP_TOPMOST, 1)  # Keep on top
    
    for i in range(3):
        cv2.createTrackbar(f'Channel {i}', 'Editor', 0, 100, update)
    
    update(0)

    # Add a loop to keep the window active and responsive
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC key to close
            break

    cv2.destroyAllWindows()