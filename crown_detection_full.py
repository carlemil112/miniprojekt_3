import cv2
import numpy as np
from matplotlib import pyplot as plt

    # Indlæs referencebillede af kongekrone
crown_img = cv2.imread('Reference Tiles/reference_crown.jpg', cv2.IMREAD_GRAYSCALE)

# Liste af tile billeder, der skal sammenlignes med reference kongekrone
tile_images = [
    r'Reference Tiles\reference_odemark2.jpg',
    r'Reference Tiles\reference_skov2.jpg',
    r'Reference Tiles\reference_mine.jpg',
    # Tilføj flere tiles efter behov
]


# SIFT detektor
sift = cv2.SIFT_create()

# BFMatcher objekt
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
# Find nøglepunkter og deskriptorer for reference kronebilledet
keypoints_1, descriptors_1 = sift.detectAndCompute(crown_img, None)


# Dictionary for at gemme resultaterne
tile_results = {}

# Iterer over tile billederne
for tile_path in tile_images:
    tile_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
    
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
        tile_results[tile_path] = len(good_matches) >= good_matches_threshold
    else:
        tile_results[tile_path] = False

# For at visualisere de bedste matches for hvert tile
for tile_path in tile_images:
    if tile_results[tile_path]:
        tile_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
        keypoints_2, descriptors_2 = sift.detectAndCompute(tile_img, None)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Visualiser de bedste 10 matches
        match_img = cv2.drawMatches(crown_img, keypoints_1, tile_img, keypoints_2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(match_img)
        plt.title(f'Matches for {tile_path}')
        plt.show()


# Print resultatet af hver tile
for tile, has_crown in tile_results.items():
    if has_crown:
        print(f'{tile} indeholder en kongekrone.')
    else:
        print(f'{tile} indeholder ikke en kongekrone.')




# Kontrastforbedring
