import cv2 as cv
import numpy as np
import os
import glob

images = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

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
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))  # Consider using median instead of mean
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