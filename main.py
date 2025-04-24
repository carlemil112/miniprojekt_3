import cv2
import numpy as np
import random
import os

from tile_classifier import Tile_Classifier
from crown_detection import CrownDetector
from neighbour_detection import NeighbourDetection
from score_calculation import point_1_74  # Sørg for at filen eksisterer og importen er korrekt


def classify_image_to_grid(tc, crown_detector, image, bins=32):
    tiles = tc.get_tiles(image)
    label_grid = []
    crown_grid = []

    for row in tiles:
        row_labels = []
        row_crowns = []
        for tile in row:
            hist = tc.get_histogram(tile, bins)
            texture = tc.get_texture_features(tile)
            combined = np.concatenate((hist, texture))

            lda_feat = tc.lda_model.transform([combined])
            label = tc.knn_model.predict(lda_feat)[0]

            hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            crowns = len(crown_detector.detect_crowns(hsv_tile))

            row_labels.append(label)
            row_crowns.append(crowns)

        label_grid.append(row_labels)
        crown_grid.append(row_crowns)

    return label_grid, crown_grid


def visualize_classification(image, tile_grid, crown_grid, regions):
    tile_height, tile_width = image.shape[0] // 5, image.shape[1] // 5

    region_colors = [tuple(random.randint(50, 255) for _ in range(3)) for _ in regions]

    for idx, region in enumerate(regions):
        color = region_colors[idx]
        for (i, j) in region['tiles']:
            y1, y2 = i * tile_height, (i + 1) * tile_height
            x1, x2 = j * tile_width, (j + 1) * tile_width
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    for i in range(5):
        for j in range(5):
            y1, y2 = i * tile_height, (i + 1) * tile_height
            x1, x2 = j * tile_width, (j + 1) * tile_width
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
            if crown_grid[i][j] > 0:
                cv2.putText(image, f"{crown_grid[i][j]}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Classified Board", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("classified_output.jpg", image)  # Uncomment to save output


def run_score_comparison_test(image_path, classifier, crown_detector):
    total = len(point_1_74)
    correct = 0
    total_error = 0

    print("\n--- Running Full Score Comparison Test on 74 Boards ---")

    for board_id, true_score in point_1_74.items():
        img_path = os.path.join(image_path, f"{board_id}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image {board_id}.jpg not found, skipping.")
            continue

        tile_grid, crown_grid = classify_image_to_grid(classifier, crown_detector, img)
        nd = NeighbourDetection(tile_grid, crown_grid)
        all_regions = nd.find_all_regions()
        predicted_score = sum(region['score'] for region in all_regions)

        print(f"Board {board_id}: True = {true_score}, Predicted = {predicted_score}")
        if predicted_score == true_score:
            correct += 1
        total_error += abs(true_score - predicted_score)

    accuracy = correct / total * 100
    avg_error = total_error / total

    print(f"\nScore Accuracy: {accuracy:.2f}%")
    print(f"Average Score Error: {avg_error:.2f}")


def main():
    image_path = r"Cropped and perspective corrected boards"
    label_path = r"labels_uden_kroner.csv"

    classifier = Tile_Classifier()
    classifier.run_pipeline(image_path, label_path)

    img_path = os.path.join(image_path, "1.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    crown_templates = [
        r"Reference_tiles\reference_crown_small1.jpg",
        r"Reference_tiles\reference_crown2.jpg"
    ]
    crown_detector = CrownDetector(crown_templates)

    tile_grid, crown_grid = classify_image_to_grid(classifier, crown_detector, img)
    nd = NeighbourDetection(tile_grid, crown_grid)

    for i in range(5):
        for j in range(5):
            print(f"\nTile ({i},{j}) = {tile_grid[i][j]}")
            print("Naboer:", nd.get_neighbours(i, j))
            print("Matchende naboer:", nd.count_matching_neighbours(i, j))
            print("Koordinater for matchende naboer:", nd.get_matching_neighbour_coords(i, j))

    all_regions = nd.find_all_regions()
    total_score = sum(region['score'] for region in all_regions)

    print(f"\nTotal board score: {total_score}")
    for idx, region in enumerate(all_regions):
        print(f"Region {idx + 1}:")
        print(f"Type: {region['tile_type']}")
        print(f"Tiles: {region['tiles']}")
        print(f"Score: {region['score']}\n")

    visualize_classification(img, tile_grid, crown_grid, all_regions)
    run_score_comparison_test(image_path, classifier, crown_detector)


if __name__ == "__main__":
    main()




