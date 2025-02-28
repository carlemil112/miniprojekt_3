import cv2 

images = r"C:\Users\anne\Desktop\Daki\s2\projekter\miniprojekt_3\miniprojekt_3\Cropped and perspective corrected boards"

# Convert to HSV

hsv_images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

