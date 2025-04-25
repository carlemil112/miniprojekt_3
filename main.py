import cv2
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


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


# Visualize the scores for each region on the original image, only colored with an outline
def visualize_scores(image, regions):
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    tile_height, tile_width = vis_image.shape[0] // 5, vis_image.shape[1] // 5

    for region in regions:
        color = tuple(random.randint(50, 255) for _ in range(3))

        # Draw outline for each tile
        for (i, j) in region['tiles']:
            y1, y2 = i * tile_height, (i + 1) * tile_height
            x1, x2 = j * tile_width, (j + 1) * tile_width
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw the score on the first tile in the region
        first_i, first_j = region['tiles'][0]
        y1, y2 = first_i * tile_height, (first_i + 1) * tile_height
        x1, x2 = first_j * tile_width, (first_j + 1) * tile_width
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.putText(vis_image, f"{region['score']}", (center_x - 10, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Write the complete score on the bottom left corner of the image
        cv2.putText(vis_image, f"Total Score: {sum(region['score'] for region in regions)}", (10, vis_image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        

    cv2.imshow("Score Visualization", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Function to load the true crown labels from the CSV file
def load_true_crown_labels_from_csv(label_file):
    # Load the CSV file
    df = pd.read_csv(label_file)
    
    # We want to extract the 'image_id', 'x', 'y', and 'crowns' columns
    crowns_data = df[['image_id', 'x', 'y', 'crowns']]
    
    # Create a dictionary to store the labels
    labels = {}
    
    # Iterate through each row in the data
    for _, row in crowns_data.iterrows():
        image_id = row['image_id']
        x, y, crowns = row['x'], row['y'], row['crowns']
        
        # If image_id is not already in the dictionary, add it
        if image_id not in labels:
            labels[image_id] = []
        
        # Append the (x, y, crowns) information to the list for the corresponding image_id
        labels[image_id].append((x, y, crowns))
    
    return labels

def run_crown_detection_accuracy_test(image_path, classifier, crown_detector, label_file):
    # Load true crown labels from the CSV file
    true_crowns = load_true_crown_labels_from_csv(label_file)

    
    

def evaluate_crown_detection(true_crowns, predicted_crowns):

    all_true = []
    all_predicted = []

    # Iterate over each image and compare true crowns to predicted ones
    for image_id in true_crowns:
        true_data = true_crowns[image_id]
        predicted_data = predicted_crowns.get(image_id, [])

        # Map the crowns to the list (we assume both true and predicted data are aligned)
        for (x, y, true_crowns_count), (_, _, predicted_crowns_count) in zip(true_data, predicted_data):
            all_true.append(true_crowns_count)
            all_predicted.append(predicted_crowns_count)

    # Compute precision, recall, and F1 score
    precision = precision_score(all_true, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_true, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_predicted, average='macro', zero_division=0)

    return precision, recall, f1







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


# Confusion matrix for the classification of the tiles

def visualize_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()










def main():
    image_path = r"miniprojekt_3\Cropped and perspective corrected boards"
    label_path = r"miniprojekt_3\labels_uden_kroner.csv"

    classifier = Tile_Classifier()
    classifier.run_pipeline(image_path, label_path)

    classifier.test_on_csv(r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\X_test.csv")


    img_path = os.path.join(image_path, "1.jpg")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    crown_templates = [
        r"Reference_tiles\reference_crown_small1.jpg",
        r"Reference_tiles\reference_crown2.jpg",
        r"Reference_tiles\reference_crown.jpg",
        r"Reference_tiles\reference_crown3.jpg"
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



    #visualize_scores(img, all_regions)
    #visualize_classification(img, tile_grid, crown_grid, all_regions)
    #run_score_comparison_test(image_path, classifier, crown_detector)   
    #visualize_confusion_matrix(classifier.y_true, classifier.y_pred, classifier.classes)
    run_crown_detection_accuracy_test(image_path, classifier, crown_detector, 'ground_truth.csv')


if __name__ == "__main__":
    main()


