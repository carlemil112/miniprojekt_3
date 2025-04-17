import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_cropped_images(cropped_images):
    """
    Viser alle beskårne billeder ved hjælp af matplotlib.
    
    Args:
    - cropped_images: Liste over beskårne billeder.
    """
    num_images = len(cropped_images)
    plt.figure(figsize=(10, 5))
    
    # Gå igennem og vis hvert billede
    for i, img in enumerate(cropped_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Konverter fra BGR (OpenCV) til RGB (matplotlib)
        plt.axis('off')  # Skjul akserne
    
    plt.show()


def crop_outer_region(tile, crop_fraction=0.3):
    """
    Beskær midten af tile og lav et hul i midten, så kun den ydre del af tile forbliver synlig.
    
    Args:
    - tile: Input tile (billede).
    - crop_fraction: Hvor meget af midten skal beskæres (f.eks. 0.2 fjerner 20% af midten).
    
    Return:
    - cropped_tile: Billede med et "hul" i midten.
    """
    height, width, _ = tile.shape
    crop_h = int(height * crop_fraction)
    crop_w = int(width * crop_fraction)
    
    # Opret en sort maske, der vil have et hul i midten
    mask = np.ones_like(tile) * 255  # Start med en hvid (255) billede
    mask[crop_h:height-crop_h, crop_w:width-crop_w] = 0  # Lav et sort "hul" i midten
    
    # Brug masken til at beholde den ydre del af tile
    cropped_tile = cv2.bitwise_and(tile, mask)
    
    return cropped_tile

def calculate_rgb_median(tile):
    """
    Beregn medianen af RGB-værdierne for en tile.
    
    Args:
    - tile: Input tile (billede).
    
    Return:
    - median_rgb: Median RGB værdier som en tuple.
    """
    median_r = np.median(tile[:, :, 0])  # Red kanal
    median_g = np.median(tile[:, :, 1])  # Green kanal
    median_b = np.median(tile[:, :, 2])  # Blue kanal
    
    return (median_r, median_g, median_b)

# Eksempel reference tiles (du kan læse dem fra billeder eller arrays)
reference_tiles_forest = [cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference_tiles\reference_skov.jpg')]  # Tilføj alle forest-tiles her
reference_tiles_plain = [cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference_tiles\reference_eng.jpg')]   # Tilføj alle plain-tiles her
reference_tiles_water = [cv2.imread(r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Reference_tiles\reference_vand.jpg')] # Tilføj alle water-tiles her

# Kategorisering af reference tiles
reference_tiles = {
    "forest": reference_tiles_forest,
    "plain": reference_tiles_plain,
    "water": reference_tiles_water
}

# Liste til opbevaring af RGB-medianer og labels
tile_rgb_medians = []
tile_labels = []
cropped_images = []  # Opret en liste til de beskårne billeder

# Gå igennem alle reference tiles
for terrain_type, tiles in reference_tiles.items():
    for tile in tiles:
        # Beskær midten og lav et hul
        cropped_tile = crop_outer_region(tile, crop_fraction=0.2)
        
        # Beregn RGB median for cropped tile
        median_rgb = calculate_rgb_median(cropped_tile)
        
        # Tilføj median RGB værdier til listen med labels
        tile_rgb_medians.append(median_rgb)
        tile_labels.append(terrain_type)
        
        # Tilføj beskåret billede til listen
        cropped_images.append(cropped_tile)

# Nu har vi RGB-medianer og labels klar til KNN-træning
print("RGB medianer for reference tiles:", tile_rgb_medians)
print("Labels:", tile_labels)

# Nu kan du vise de beskårne billeder
display_cropped_images(cropped_images)
