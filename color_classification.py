import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from sklearn.model_selection import train_test_split

images = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

image_files = [f for f in os.listdir(images) if f.endswith(('.jpg'))]

# Split into 80% train and 20% test
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

def image_preprocessing(image_path):
    path = os.path.join(images, image_path)
    image = cv.imread(path)

    if image is None:
        print(f"Could not load image")
        return None, None

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return image, hsv_image

# Process all training images
train_images = [image_preprocessing(f) for f in train_files]

#knn klassifisering med median. Kig på kanter. Temple matching, hawk og sift. 

'''

# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    
    # Retrieve list of images
    image_list = glob.glob(os.path.join(images, "*.jpg"))

    if not image_list:  # If no images found
        print("No images found in the specified directory.")
        exit()

    image_path = image_list[0]  # Select the first image

    if not os.path.isfile(image_path):  # Ensure valid file path
        print("Image not found")
        exit()
    
    # Read and process the image
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(f"Total tiles detected: {len(tiles)}")
    
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain(tile))
            print("=====")

# Break a board into tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Determine the type of terrain in a tile
def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))  
    print(f"H: {hue}, S: {saturation}, V: {value}")
    
    if 21 < hue < 106 and 75 < saturation < 255 and 42 < value < 198:
        return "Field"
    if 36.5 < hue < 48 and 117.5 < saturation < 168 and 39 < value < 50.5:
        return "Forest"
    if 106 < hue < 108 and 246.5 < saturation < 253 and 134 < value < 161.5:
        return "Lake"
    if 39 < hue < 44 and 207 < saturation < 228 and 107 < value < 131:
        return "Grassland"
    if 20 < hue < 23 and 102.5 < saturation < 145 and 85 < value < 114:
        return "Swamp"
    if 22 < hue < 24 and 61.7 < saturation < 95 and 31.25 < value < 51.75:
        return "Mine"
    if 25.25 < hue < 32.75 and 54.25 < saturation < 98.5 and 68 < value < 117.5:
        return "Home"
    
    return "Unknown"

if __name__ == "__main__":
    main()



import cv2 as cv
import numpy as np
import os
import glob
from collections import defaultdict

images = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

# Main function containing the backbone of the program
def main():
    print("+-------------------------------+")
    print("| King Domino Points Calculator |")
    print("+-------------------------------+")
    
    # Retrieve list of images
    image_list = glob.glob(os.path.join(images, "*.jpg"))

    if not image_list:  # If no images found
        print("No images found in the specified directory.")
        exit()

    image_path = image_list[0]  # Select the first image

    if not os.path.isfile(image_path):  # Ensure valid file path
        print("Image not found")
        exit()
    
    # Read and process the image
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    
    # Initialize dictionary to store terrain counts
    terrain_counts = defaultdict(int)

    print(f"Total tiles detected: {len(tiles) * len(tiles[0])}")  # Print total tile count

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            terrain = get_terrain(tile)
            terrain_counts[terrain] += 1  # Count the terrain type
            print(f"Tile ({x}, {y}): {terrain}")
            print("=====")

    # Print summary of terrain classification
    print("\nSummary of Terrain Classification:")
    for terrain, count in terrain_counts.items():
        print(f"{terrain}: {count} tiles")

# Break a board into tiles
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Determine the type of terrain in a tile
def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))  
    print(f"H: {hue}, S: {saturation}, V: {value}")
    
    if 21 < hue < 106 and 75 < saturation < 255 and 42 < value < 198:
        return "Field"
    if 36.5 < hue < 48 and 117.5 < saturation < 168 and 39 < value < 50.5:
        return "Forest"
    if 106 < hue < 108 and 246.5 < saturation < 253 and 134 < value < 161.5:
        return "Lake"
    if 39 < hue < 44 and 207 < saturation < 228 and 107 < value < 131:
        return "Grassland"
    if 20 < hue < 23 and 102.5 < saturation < 145 and 85 < value < 114:
        return "Swamp"
    if 22 < hue < 24 and 61.7 < saturation < 95 and 31.25 < value < 51.75:
        return "Mine"
    if 25.25 < hue < 32.75 and 54.25 < saturation < 98.5 and 68 < value < 117.5:
        return "Home"
    
    return "Unknown"

if __name__ == "__main__":
    main()

'''