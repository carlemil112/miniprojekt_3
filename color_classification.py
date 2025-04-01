import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Load images from directory.
def load_images(image_path):
    image_files = sorted(os.listdir(image_path), key=lambda f: int(os.path.splitext(f)[0]))
    images = []
    for f in image_files:
        img = cv.imread(os.path.join(image_path, f))
        if img is not None:
            images.append(img)
        else:
            print(f"⚠️ Skipped loading {f}")
    return images

#Split image into 5x5 grid of 100x100px tiles.
def get_tiles(image):
    tiles = []
    for y in range(5):  # rows
        tiles.append([])
        for x in range(5):  # columns
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

#Get normalized hue histogram from a tile.
def get_histogram(tile, bins):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hist = cv.calcHist([hue], [0], None, [bins], [0, 180])
    hist = cv.normalize(hist, hist).flatten()
    return hist


def x_images(images):
    print(f"Loaded {len(images)} images.")


def x_histograms(all_histograms):
    print(f"Total histograms generated: {len(all_histograms)}")

#Load CSV labels into a dictionary.
def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df['key'] = list(zip(df['Image'], df['row'], df['column']))
    label_map = dict(zip(df['key'], df['TrueLabel']))
    return label_map

#Generate histograms and match with labels.
def collect_hist_and_label(images, bins, label_map):
    histograms = []
    labels = []

    label_grid = np.empty((len(images), 5, 5), dtype=object)  # Store tile labels per image

    for img_index, img in enumerate(images, start=1):  # Start at 1 to match CSV
        tiles = get_tiles(img)

        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                hist = get_histogram(tile, bins=bins)
                histograms.append(hist)

                key = (img_index, row_idx, col_idx)
                label = label_map.get(key, "unknown")
                labels.append(label)

                # Store label in the label grid
                label_grid[img_index - 1, row_idx, col_idx] = label

    return histograms, labels, label_grid


def apply_lda(X_train, X_test, y_train, n_components=2):
    """
    Applies Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Parameters:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Testing data features.
        y_train (np.ndarray): Training labels (required for LDA).
        n_components (int): Number of components to reduce to (max C-1).

    Returns:
        X_train_lda (np.ndarray): LDA-transformed training data.
        X_test_lda (np.ndarray): LDA-transformed test data.
        lda (LinearDiscriminantAnalysis): The fitted LDA model.
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda


# Main entry point
def main():
    image_path = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"
    label_path = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\labels_uden_kroner.csv"

    images = load_images(image_path)
    x_images(images)

    bins = 30
    label_map = load_labels(label_path)
    all_histograms, all_labels, label_grid = collect_hist_and_label(images, bins, label_map)
    x_histograms(all_histograms)

    print(f"Total labels collected: {len(all_labels)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(all_histograms, all_labels, test_size=0.2, random_state=42)

    X_train_lda, X_test_lda, lda_model = apply_lda(X_train, X_test, y_train, n_components=6)

if __name__ == "__main__":
    main()



#knn klassifisering med median. Kig på kanter. Til krone: Temple matching, hawk og sift. 