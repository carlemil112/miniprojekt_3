import cv2
import numpy as np

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

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_sift_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def detect_terrain_sift(tile, reference_images, threshold=10):
    _, tile_descriptors = extract_sift_features(tile)

    if tile_descriptors is None:
        return 'ukendt'  # If no descriptors found, return 'ukendt'

    best_match = None
    max_matches = 0

    for terrain_type, ref_image in reference_images.items():
        _, ref_descriptors = extract_sift_features(ref_image)
        if ref_descriptors is not None:
            matches = match_sift_features(tile_descriptors, ref_descriptors)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_match = terrain_type

    if max_matches < threshold:
        return 'ukendt'  # If the number of matches is below the threshold, return 'ukendt'

    return best_match if best_match else 'ukendt'


reference_images = {
    'skov': cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_skov.jpg'),
    'mark': cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_mark.jpg'),
    'odemark': cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_odemark.jpg'),
    'vand': cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference Tiles\reference_vand.jpg'),
}

image = cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Cropped and perspective corrected boards\1.jpg')

if image is None:
    print("Error loading the image.")
else:
    tiles = split_into_tiles(image, 5, 5)

    for idx, tile in enumerate(tiles):
        terrain = detect_terrain_sift(tile, reference_images, threshold=10)
        print(f"Tile {idx + 1}: Terræntype = {terrain}")

        cv2.imshow(f"Tile {idx + 1}", tile)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
