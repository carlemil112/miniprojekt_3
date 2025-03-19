import cv2
import numpy as np

crown_img = cv2.imread(r'Reference Tiles\reference_crown.jpg', cv2.IMREAD_GRAYSCALE)

tile_img = cv2.imread(r'Reference Tiles\reference_vand3.jpg', cv2.IMREAD_GRAYSCALE)


sift = cv2.SIFT_create()


keypoints_1, descriptors_1 = sift.detectAndCompute(crown_img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(tile_img, None)

#Matcher for at sammenligne deskriptorer:
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#Matches mellem deskriptorer fra kronebilledet og tilebilledet:
matches = bf.match(descriptors_1, descriptors_2)

#Sorter matches efter afstand (lavest er bedst):
matches = sorted(matches, key=lambda x: x.distance)

#Visualisering af bedste matches:
match_img = cv2.drawMatches(crown_img, keypoints_1, tile_img, keypoints_2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


#Visning af resultat:
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

