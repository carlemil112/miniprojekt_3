import cv2
import numpy as np

# Funktion til at vise billeder
def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2 
import glob
import os

# Definer begge fillokationer
path_1 = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"
path_2 = r"C:\Users\carle\Desktop\python_work\Miniprojekt_3\Cropped and perspective corrected boards"

# Først, forsøg at finde billeder i path_1
image_files = glob.glob(os.path.join(path_1, "*.jpg"))

# Hvis ingen billeder blev fundet i path_1, prøv path_2
if not image_files:
    image_files = glob.glob(os.path.join(path_2, "*.jpg"))

# Check if any images are found
if image_files:
    # Select the first image from the list
    image_path = image_files[0]
    
    # Read the image
    image = cv2.imread(image_path)
        

# Konverter billedet til HSV-farverum
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definer farveområder for landskaber (gul, grøn, blå, brun)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

lower_brown = np.array([10, 100, 100])
upper_brown = np.array([20, 255, 255])

# Farveområde for kroner (lysere gul)
lower_crown = np.array([20, 150, 150])
upper_crown = np.array([30, 255, 255])

# Lav masker for hvert farveområde
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
crown_mask = cv2.inRange(hsv_image, lower_crown, upper_crown)

# Kombiner alle landskabsfarvemasker til et samlet billede
combined_mask = yellow_mask | green_mask | blue_mask | brown_mask

# Find konturerne af kronerne
contours, _ = cv2.findContours(crown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tegn konturerne af kronerne på billedet
output_image = image.copy()
cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)

# Vis resultatet
show_image("Identificerede Kroner", output_image)

# Tæl hvor mange pixels hver landskabstype fylder
yellow_pixels = cv2.countNonZero(yellow_mask)
green_pixels = cv2.countNonZero(green_mask)
blue_pixels = cv2.countNonZero(blue_mask)
brown_pixels = cv2.countNonZero(brown_mask)

# Udskriv resultatet
print(f"Antal gule pixels (marker): {yellow_pixels}")
print(f"Antal grønne pixels (skove): {green_pixels}")
print(f"Antal blå pixels (vand): {blue_pixels}")
print(f"Antal brune pixels (ørken/bjerg): {brown_pixels}")

# Vis de segmenterede landskaber
show_image("Landskaber Segmenteret", combined_mask)

# Vis de oprindelige masker for hver landskabstype
show_image("Gul Mark", yellow_mask)
show_image("Grøn Skov", green_mask)
show_image("Blå Vand", blue_mask)
show_image("Brun Ørken/Bjerg", brown_mask)

# Vis kronemasken (til visuel kontrol)
show_image("Kroner Segmenteret", crown_mask)
