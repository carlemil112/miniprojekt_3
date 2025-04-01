import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the reference crown image
crown_img = cv2.imread('Reference Tiles/reference_crown.jpg')

# Convert the image to RGB
reference_image = cv2.cvtColor(crown_img, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)

# Create test image by adding Scale Invariance and Rotational Invariance
# Instead of pyrDown, use resize to scale down the image by a smaller factor
scale_factor = 0.75  # Adjust this factor to control the scale
test_image = cv2.resize(reference_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Rotate the image
num_rows, num_cols = test_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# Display training and test images
fx, plots = plt.subplots(1, 2, figsize=(20,10))

# Reference Image
plots[0].set_title("Reference Image")
plots[0].imshow(reference_image)

# Test Image
plots[1].set_title("Test Image")
plots[1].imshow(test_image)

# Show the plot
plt.show()

# Use SIFT to detect keypoints and compute descriptors
sift = cv2.SIFT_create()

train_keypoints, train_descriptor = sift.detectAndCompute(gray_image, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_gray, None)

# BFMatcher with default parameters
bf = cv2.BFMatcher()

# Find the best matches between train and test image descriptors
matches = bf.knnMatch(train_descriptor, test_descriptor, k=2)

# Apply ratio test to keep good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the matches
matched_image = cv2.drawMatches(reference_image, train_keypoints, test_image, test_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched keypoints
plt.figure(figsize=(20,10))
plt.title('Matched Keypoints')
plt.imshow(matched_image)
plt.show()

# Print the number of good matches
print(f"Number of good matches: {len(good_matches)}")
