import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Taken from P0, with small tweaks:
def load_images(image_path, target_size=(500, 500)):
    images = []
    if not os.path.isdir(image_path):
        print("Image not found")
        return images

    for file in os.listdir(image_path):
        path = os.path.join(image_path, file)
        img = cv.imread(path)
        if img is not None:
                img = cv.resize(img, target_size)
                images.append(img)
        else:
            print(f"Failed to load")

    return images

#Breaks a board into tiles. We could also use the reference tile folder instead.  
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles


def get_histogram(tile, bins):
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue = hsv[:,:,0]

    hist = cv.calcHist([hue], [0], None, [bins], [0, 180])
    hist = cv.normalize(hist, hist).flatten()

    return hist


def main():
    image_path = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Reference_tiles"
    images = load_images(image_path, target_size=(500, 500))
    print(f"Loaded {len(images)} images.")

    all_histograms = []
    bins = 30

    for img in images:
        tiles = get_tiles(img)
        for row in tiles:
            for tile in row:
                hist = get_histogram(tile, bins=bins)
                all_histograms.append(hist)

    print(f"Total histograms generated: {len(all_histograms)}")

    plt.plot(all_histograms[0])
    plt.title(f"Hue Histogram (bins={bins}) of First Tile")
    plt.xlabel("Hue Bin")
    plt.ylabel("Normalized Frequency")
    plt.show()


if __name__ == "__main__":
    main()



#knn klassifisering med median. Kig på kanter. Til krone: Temple matching, hawk og sift. 