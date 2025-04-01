import cv2
import numpy as np

# Indlæs referencebilledet af kronen
reference_image_path = r'Reference Tiles/reference_crown.jpg'
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# Indlæs det fulde bræt (bræt-billedet med 5x5 tiles)
board_image_path = r'Cropped and perspective corrected boards\1.jpg'  # Ændr til stien for dit bræt-billede
board_image = cv2.imread(board_image_path, cv2.IMREAD_GRAYSCALE)

# SIFT detektor og matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Funktion til at forbedre kontrast og anvende støjreduktion
def preprocess_image(image):
    # Fjern støj med GaussianBlur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Forbedr kontrasten med histogramudligning
    equalized = cv2.equalizeHist(blurred)
    
    return equalized

# Funktion til at finde matches mellem tile og referencebillede
def match_tile_with_reference(tile_img, reference_img, threshold=5):
    # Forbehandling af billeder
    tile_img = preprocess_image(tile_img)
    reference_img = preprocess_image(reference_img)

    # Find SIFT nøglepunkter og deskriptorer for både tile og reference
    kp1, des1 = sift.detectAndCompute(tile_img, None)
    kp2, des2 = sift.detectAndCompute(reference_img, None)

    if des1 is None or des2 is None:
        return False  # Ingen match, hvis der ikke er deskriptorer

    # Matcher deskriptorerne
    matches = bf.match(des1, des2)

    # Sortér efter bedste matches
    matches = sorted(matches, key=lambda x: x.distance)

    # Vælg de bedste matches (juster threshold)
    good_matches = [m for m in matches if m.distance < 300]

    # Returner True, hvis antallet af gode matches overstiger threshold
    return len(good_matches) > threshold

# Funktion til at opdele brættet i 5x5 tiles
def split_image_into_tiles(image, grid_size=(5, 5)):
    tiles = []
    h, w = image.shape
    tile_h, tile_w = h // grid_size[0], w // grid_size[1]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            tile = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            tiles.append(tile)
    return tiles

# Opdel bræt-billedet i 5x5 tiles
tiles = split_image_into_tiles(board_image)

# Iterér over hver tile og match mod referencekronen
tile_results = {}
for idx, tile_img in enumerate(tiles):
    if match_tile_with_reference(tile_img, reference_image):
        tile_results[f'Tile {idx}'] = 'Krone fundet'
    else:
        tile_results[f'Tile {idx}'] = 'Ingen krone fundet'

# Print resultaterne for hver tile
for tile, result in tile_results.items():
    print(f'{tile}: {result}')
