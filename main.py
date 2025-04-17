import color_classification as cc
from color_classification import Tile_Classifier
from neighbour_detection import NeighbourDetection
from template_matching import CrownDetector
import cv2
import numpy as np


# Classify all tiles in the image using histogram + texture + LDA + KNN,
# and count crowns using the CrownDetector
def classify_image_to_grid(tc, crown_detector, image, bins=32):
    tiles = tc.get_tiles(image)
    label_grid = []
    crown_grid = []

    for row in tiles:
        row_labels = []
        row_crowns = []
        for tile in row:
            # Feature extraction
            hist = tc.get_histogram(tile, bins)
            texture = tc.get_texture_features(tile)
            combined = np.concatenate((hist, texture))

            # LDA + KNN classification
            lda_feat = tc.lda_model.transform([combined])
            label = tc.knn_model.predict(lda_feat)[0]

            # Crown detection
            hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            crowns = len(crown_detector.detect_crowns(hsv_tile))

            row_labels.append(label)
            row_crowns.append(crowns)

        label_grid.append(row_labels)
        crown_grid.append(row_crowns)

    return label_grid, crown_grid


def main():
    image_path = r"Cropped and perspective corrected boards"
    label_path = r"labels_uden_kroner.csv"

    # Train the classifier
    classifier = Tile_Classifier()
    classifier.run_pipeline(image_path, label_path)

    # Load a single test image
    img = cv2.imread(r"Cropped and perspective corrected boards\2.jpg")
    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")

    # Crown detector setup
    crown_templates = [
        r"Reference_tiles\reference_crown_small1.jpg",
        r"Reference_tiles\reference_crown2.jpg"
    ]
    crown_detector = CrownDetector(crown_templates)

    # Classify tiles and count crowns
    tile_grid, crown_grid = classify_image_to_grid(classifier, crown_detector, img)

    # Initialize neighbour detection
    nd = NeighbourDetection(tile_grid, crown_grid)

    # Print tile classification and neighbour info
    for i in range(5):
        for j in range(5):
            print(f"\nTile ({i},{j}) = {tile_grid[i][j]}")
            print("Naboer:", nd.get_neighbours(i, j))
            print("Matchende naboer:", nd.count_matching_neighbours(i, j))
            print("Koordinater for matchende naboer:", nd.get_matching_neighbour_coords(i, j))

    # Score calculation
    all_regions = nd.find_all_regions()
    total_score = sum(region['score'] for region in all_regions)

    print(f"\nTotal board score: {total_score}")
    for idx, region in enumerate(all_regions):
        print(f"Region {idx + 1}:")
        print(f"Type: {region['tile_type']}")
        print(f"Tiles: {region['tiles']}")
        print(f"Score: {region['score']}\n")


if __name__ == "__main__":
    main()
