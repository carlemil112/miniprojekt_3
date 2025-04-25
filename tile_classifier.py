import cv2 as cv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Tile_Classifier:
        
    def __init__(self):
        self.images = None
        self.color_histograms = None
        self.texture_features = None
        self.labels = None
        self.combined_features = None
        self.lda_model = None
        self.knn_model = None
        self.y_true = None
        self.y_pred = None
        self.classes = None

    # Load images from directory
    def load_images(self, image_path):
        image_files = sorted(os.listdir(image_path), key=lambda f: int(os.path.splitext(f)[0]))
        self.images = []
        for f in image_files:
            img = cv.imread(os.path.join(image_path, f))
            if img is not None:
                self.images.append(img)
            else:
                print(f"Skipped loading {f}")
        return self.images

    # Split image into 5x5 grid of 100x100px tiles
    def get_tiles(self, image):
        tiles = []
        for y in range(5):  # rows
            row = []
            for x in range(5):  # columns
                row.append(image[y*100:(y+1)*100, x*100:(x+1)*100])
            tiles.append(row)
        return tiles

    # Get normalized hue histogram from a tile
    def get_histogram(self, tile, bins):
        hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        hist = cv.calcHist([hue], [0], None, [bins], [0, 180])
        hist = cv.normalize(hist, hist).flatten()
        return hist

    # Extract texture features using Sobel filter
    def get_texture_features(self, tile):
        # Convert to grayscale for texture analysis
        gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
        
        # Apply Sobel filter in x and y directions
        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize the magnitude
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        # Calculate statistics as texture features
        features = [
            np.mean(magnitude),           # Mean gradient strength
            np.std(magnitude),            # Standard deviation
            np.percentile(magnitude, 75), # 75th percentile
            np.percentile(magnitude, 90), # 90th percentile
            np.max(magnitude),            # Maximum gradient
            np.sum(magnitude > 0.25),     # Count of strong edges
        ]
        
        return features

    # Load CSV labels into a dictionary
    def load_labels(self, csv_path):
        df = pd.read_csv(csv_path)
        df['key'] = list(zip(df['Image'], df['row'], df['column']))
        label_map = dict(zip(df['key'], df['TrueLabel']))
        return label_map

    # Generate histograms, texture features and match with labels
    def collect_features_and_label(self, images, label_map, bins=32):
        color_histograms = []
        texture_features = []
        labels = []
        
        for img_index, img in enumerate(images, start=1):  # Start at 1 to match CSV
            tiles = self.get_tiles(img)

            for row_idx, row in enumerate(tiles):
                for col_idx, tile in enumerate(row):
                    # Get color features
                    hist = self.get_histogram(tile, bins=bins)
                    color_histograms.append(hist)
                    
                    # Get texture features
                    texture = self.get_texture_features(tile)
                    texture_features.append(texture)

                    key = (img_index, row_idx, col_idx)
                    label = label_map.get(key, "unknown")
                    labels.append(label)

        return color_histograms, texture_features, labels

    def split_data(self, X, y, test_size=0.2, val_size=0.25, random_state=42):
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    #slet senere
    #def save_to_csv(self, X, y, filename):
        """Saves feature data and labels to a CSV file."""
        #data = pd.DataFrame(X)
        #data['label'] = y
        #data.to_csv(filename, index=False)
        #print(f"Data saved to {filename}")

    def lda(self, X_train, X_val, X_test, y_train):
        lda = LinearDiscriminantAnalysis(n_components=7)
        lda.fit(X_train, y_train)

        X_train_lda = lda.transform(X_train)
        X_val_lda = lda.transform(X_val)
        X_test_lda = lda.transform(X_test)

        return lda, X_train_lda, X_val_lda, X_test_lda

    def knn(self, X_train_lda, X_val_lda, X_test_lda, y_train, y_val, y_test, n_neighbors=5):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_lda, y_train)
        
        y_val_pred = knn.predict(X_val_lda)

        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
        print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

        return knn
    
    def run_pipeline(self, image_path, label_path, bins=32, n_neighbors=5):
        self.load_images(image_path)
        label_map = self.load_labels(label_path)
        
        # Collect both color and texture features
        color_histograms, texture_features, all_labels = self.collect_features_and_label(self.images, label_map, bins=bins)
        
        print(f"Total features generated: {len(color_histograms)}")
        print(f"Total labels collected: {len(all_labels)}")
        
        # Combine color and texture features
        self.combined_features = []
        for i in range(len(color_histograms)):
            combined = np.concatenate([color_histograms[i], texture_features[i]])
            self.combined_features.append(combined)
        
        print(f"Combined feature length: {len(self.combined_features[0])}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(self.combined_features, all_labels)
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        self.lda_model, X_train_lda, X_val_lda, X_test_lda = self.lda(X_train, X_val, X_test, y_train)
        self.knn_model = self.knn(X_train_lda, X_val_lda, X_test_lda, y_train, y_val, y_test, n_neighbors=5)

        self.y_pred = self.knn_model.predict(X_test_lda)
        self.y_true = y_test
        self.classes = sorted(list(set(all_labels)))


    def test_on_csv(self, test_csv_path, label_column='label'):
        # Load test data
        df = pd.read_csv(test_csv_path)
        X_test = df.drop(columns=[label_column]).values
        y_test = df[label_column].values

        # Transform features using trained LDA model
        X_test_lda = self.lda_model.transform(X_test)

        # Predict using trained KNN model
        y_pred = self.knn_model.predict(X_test_lda)

        # Evaluation
        print("Test Accuracy:", accuracy_score(y_test, y_pred))
        print("Test Classification Report:\n", classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y_test)))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix on Test Set")
        plt.tight_layout()
        plt.show()
        



