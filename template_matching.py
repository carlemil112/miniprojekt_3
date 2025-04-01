import cv2
import numpy as np

# Indlæs og konverter til HSV
board_img = cv2.imread(r'Cropped and perspective corrected boards/5.jpg')
board_hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

# Indlæs reference billeder for forskellige rotationer
template_0 = cv2.imread(r'Reference Tiles/reference_crown_small1.jpg')
template_90 = cv2.rotate(template_0, cv2.ROTATE_90_CLOCKWISE)
template_180 = cv2.rotate(template_0, cv2.ROTATE_180)
template_270 = cv2.rotate(template_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Konverter alle reference billeder til HSV
templates = [
    cv2.cvtColor(template_0, cv2.COLOR_BGR2HSV),
    cv2.cvtColor(template_90, cv2.COLOR_BGR2HSV),
    cv2.cvtColor(template_180, cv2.COLOR_BGR2HSV),
    cv2.cvtColor(template_270, cv2.COLOR_BGR2HSV)
]
h, w = templates[0].shape[:2]

# Optimerede parametre
threshold = 0.5  # Øget tærskel for færre false positives
min_scale = 0.95    # Skaleringsinterval
max_scale = 1.05
scale_steps = 3

# Non-maximum suppression til at fjerne overlap
def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], 
            np.where(overlap > overlapThresh)[0])))
    return boxes[pick]

# Funktion til at detektere kroner på en tile med flere templates
def detect_crowns(tile):
    detected_boxes = []
    
    for scale in np.linspace(min_scale, max_scale, scale_steps):
        for template in templates:
            resized = cv2.resize(template, (int(w*scale), int(h*scale)))
            resized_h, resized_w = resized.shape[:2]
            
            result = cv2.matchTemplate(tile, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > threshold:
                detected_boxes.append((max_loc[0], max_loc[1],
                                       max_loc[0] + resized_w, max_loc[1] + resized_h))
    
    return non_max_suppression(detected_boxes)

# Split board i tiles
tile_height = board_img.shape[0] // 5
tile_width = board_img.shape[1] // 5

for i in range(5):
    for j in range(5):
        y1 = i * tile_height
        y2 = y1 + tile_height
        x1 = j * tile_width
        x2 = x1 + tile_width
        
        tile = board_hsv[y1:y2, x1:x2]
        
        # Detekter kroner med den nye metode
        crowns = detect_crowns(tile)
        
        if len(crowns) > 0:
            cv2.rectangle(board_img, (x1, y1), (x2, y2), (0,255,0), 2)
            for (bx1, by1, bx2, by2) in crowns:
                cv2.rectangle(board_img, 
                              (x1 + bx1, y1 + by1),
                              (x1 + bx2, y1 + by2),
                              (0,0,255), 2)

cv2.imshow('Result', board_img)
cv2.waitKey(0)
cv2.destroyAllWindows()