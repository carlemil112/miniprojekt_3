import color_classification as cc
from color_classification import Tile_Classifier
from neighbour_detection import NeighbourDetection
from template_matching import CrownDetector
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Initialize classifier instance
classifier = Tile_Classifier()

# Load a single image to test
images = classifier.load_images(r"miniprojekt_3\Cropped and perspective corrected boards")
image = images[0]  # Let's visualize features from the first loaded image

# Split into tiles and select the first tile
tiles = classifier.get_tiles(image)
first_tile = tiles[0][0]  # First tile from top-left

# Compute histogram and texture features
histogram = classifier.get_histogram(first_tile, bins=32)
texture_features = classifier.get_texture_features(first_tile)

# Visualization of histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(histogram)), histogram, color='blue')
plt.title('Hue Histogram of First Tile')
plt.xlabel('Bin')
plt.ylabel('Frequency')

# Visualization of texture (Sobel magnitude)
gray = cv.cvtColor(first_tile, cv.COLOR_BGR2GRAY)
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Texture Visualization (Sobel Magnitude)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()

# Load labels from CSV
label_map = classifier.load_labels(r"miniprojekt_3\labels_uden_kroner.csv")  # Adjust path if needed

# Extract features and labels across all tiles
color_features, texture_features, labels = classifier.collect_features_and_label(
    classifier.images, label_map, bins=32
)

# Combine color and texture features
combined_features = np.hstack((color_features, texture_features))

# Reduce dimensions using PCA for before-LDA plot
pca = PCA(n_components=2)
features_pca = pca.fit_transform(combined_features)

# Apply LDA to reduce dimensions and improve class separation
lda = LinearDiscriminantAnalysis(n_components=2)
features_lda = lda.fit_transform(combined_features, labels)

# Plot PCA (before LDA) and LDA result side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Apply LDA to reduce dimensions and improve class separation
lda = LinearDiscriminantAnalysis(n_components=2)
features_lda = lda.fit_transform(combined_features, labels)

# Encode string labels into integers
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Apply LDA using numeric labels
lda = LinearDiscriminantAnalysis(n_components=2)
features_lda = lda.fit_transform(combined_features, numeric_labels)


# BEFORE (PCA)
scatter = axs[0].scatter(features_pca[:, 0], features_pca[:, 1], c=numeric_labels, cmap='tab10', s=10)
axs[0].set_title('Before LDA (PCA Visualization)')
axs[0].set_xlabel('PCA Component 1')
axs[0].set_ylabel('PCA Component 2')

# AFTER (LDA)
scatter = axs[1].scatter(features_lda[:, 0], features_lda[:, 1], c=numeric_labels, cmap='tab10', s=10)
axs[1].set_title('After LDA')
axs[1].set_xlabel('LDA Component 1')
axs[1].set_ylabel('LDA Component 2')

# Optional: replace auto-generated legend with actual class names
handles, _ = scatter.legend_elements()
class_labels = label_encoder.classes_
axs[1].legend(handles, class_labels, title="Terrain Types", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()