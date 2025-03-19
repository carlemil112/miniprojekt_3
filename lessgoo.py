import cv2
import numpy as np
from matplotlib import pyplot as plt

# Indlæs referencebillede af kongekrone
crown_img = cv2.imread(r'Reference Tiles/reference_crown.jpg', cv2.IMREAD_GRAYSCALE)

# Indlæs full board billede, der skal opdeles i tiles
board_img = cv2.imread(r'Cropped and perspective corrected boards\1.jpg', cv2.IMREAD_GRAYSCALE)

# SIFT detektor
sift = cv2.SIFT_create()

# BFMatcher objekt
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Find nøglepunkter og deskriptorer for reference kronebilledet
keypoints_1, descriptors_1 = sift.detectAndCompute(crown_img, None)

# Dimensioner for grid (antager 5x5 grid)
grid_size = 5
board_height, board_width = board_img.shape

# Beregn dimensionerne for hver tile
tile_height = board_height // grid_size
tile_width = board_width // grid_size

# Dictionary for at gemme resultaterne
tile_results = {}

# Iterer over grid for at dele brættet op i tiles
tile_num = 1  # Bruges til at navngive tiles
for row in range(grid_size):
    for col in range(grid_size):
        # Skær tile ud baseret på dets position i grid
        tile_img = board_img[row * tile_height:(row + 1) * tile_height, col * tile_width:(col + 1) * tile_width]

        # Find nøglepunkter og deskriptorer for tile-billedet
        keypoints_2, descriptors_2 = sift.detectAndCompute(tile_img, None)

        if descriptors_2 is not None:
            # Matcher deskriptorer mellem kronebilledet og tile
            matches = bf.match(descriptors_1, descriptors_2)

            # Sorter matches efter distance (lavere er bedre)
            matches = sorted(matches, key=lambda x: x.distance)

            # Brug kun de bedste matches
            good_matches_threshold = 5  # Juster denne værdi baseret på dine tests
            good_matches = [m for m in matches if m.distance < 300]

            # Gem resultatet (True hvis tile indeholder en kongekrone)
            tile_results[f'Tile {tile_num}'] = len(good_matches) >= good_matches_threshold
        else:
            tile_results[f'Tile {tile_num}'] = False

        # Visualiser de bedste 10 matches for hver tile, hvis der er en krone
        if tile_results[f'Tile {tile_num}']:
            matches = bf.match(descriptors_1, descriptors_2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Visualiser de bedste 10 matches
            match_img = cv2.drawMatches(crown_img, keypoints_1, tile_img, keypoints_2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            plt.figure(figsize=(10, 7))
            plt.imshow(match_img)
            plt.title(f'Matches for Tile {tile_num}')
            plt.show()

        tile_num += 1

# Print resultatet af hver tile
for tile, has_crown in tile_results.items():
    if has_crown:
        print(f'{tile} indeholder en kongekrone.')
    else:
        print(f'{tile} indeholder ikke en kongekrone.')
