import cv2
import numpy as np

# Preprocessing function for images
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optionally apply other preprocessing steps like histogram equalization or blurring
    # gray = cv2.equalizeHist(gray)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# Function to split image into tiles
def split_into_tiles(image, rows, cols):
    height, width, _ = image.shape
    tile_height = height // rows
    tile_width = width // cols
    tiles = []
    for i in range(rows):
        for j in range(cols):
            tile = image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            tiles.append(tile)
    return tiles

# Extract SIFT features
def extract_sift_features(image):
    gray = preprocess_image(image)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Matching SIFT features using Lowe's ratio test
def match_sift_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return good_matches

# Function to detect terrain type in a tile based on SIFT features
def detect_terrain_sift(tile, reference_images, threshold=10):
    _, tile_descriptors = extract_sift_features(tile)

    if tile_descriptors is None:
        return 'ukendt'  # No descriptors found, return 'unknown'

    best_match = None
    max_matches = 0

    # Loop through each terrain type and reference images
    for terrain_type, ref_images in reference_images.items():
        for ref_image in ref_images:
            _, ref_descriptors = extract_sift_features(ref_image)
            if ref_descriptors is not None:
                matches = match_sift_features(tile_descriptors, ref_descriptors)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    best_match = terrain_type

    # If matches are below threshold, return 'unknown'
    if max_matches < threshold:
        return 'ukendt'

    return best_match if best_match else 'ukendt'


# Example: Reference images for terrain types (you need to load the reference images)
reference_images = {
    'skov': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_skov.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_skov2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_skov3.jpg')
    ],
    'mark': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mark.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mark2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mark3.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mark4.jpg')

    ],
    'odemark': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark3.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark4.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark5.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark6.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark7.jpg')
        
    ],
    'vand': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_vand.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_vand2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_vand3.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_vand4.jpg')
    ],
    'eng': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_eng.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_eng2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_eng3.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_eng4.jpg')
    ],
    'mine': [
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mine.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mine2.jpg'),
        cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mine3.jpg')
    ]

}

# Load an example tile map image
tile_map_image = cv2.imread(r'')

# Split the image into tiles (define number of rows and columns)
rows, cols = 4, 4  # Adjust based on the image layout
tiles = split_into_tiles(tile_map_image, rows, cols)

# Classify each tile
for idx, tile in enumerate(tiles):
    terrain_type = detect_terrain_sift(tile, reference_images)
    print(f"Tile {idx}: {terrain_type}")






