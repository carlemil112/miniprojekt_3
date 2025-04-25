import cv2
import numpy as np
import glob

class CrownDetector:
    def __init__(self, template_paths, threshold=0.5, min_scale=0.95, max_scale=1.05, scale_steps=3):
        self.threshold = threshold
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_steps = scale_steps
        self.templates = self.load_and_prepare_templates(template_paths)

        # Gem original template højde og bredde for skalering
        self.h, self.w = self.templates[0].shape[:2]

    def load_and_prepare_templates(self, template_paths):
        templates = []
        for path in template_paths:
            base_template = cv2.imread(path)
            if base_template is None:
                raise FileNotFoundError(f"Template not found: {path}")
            # Opret rotationer
            rotations = [
                base_template,
                cv2.rotate(base_template, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(base_template, cv2.ROTATE_180),
                cv2.rotate(base_template, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ]
            for rot in rotations:
                templates.append(cv2.cvtColor(rot, cv2.COLOR_BGR2HSV))
        return templates

    def non_max_suppression(self, boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
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
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return boxes[pick]

    def detect_crowns(self, tile):
        detected_boxes = []
        for scale in np.linspace(self.min_scale, self.max_scale, self.scale_steps):
            for template in self.templates:
                resized = cv2.resize(template, (int(self.w * scale), int(self.h * scale)))
                result = cv2.matchTemplate(tile, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > self.threshold:
                    detected_boxes.append((
                        max_loc[0], max_loc[1],
                        max_loc[0] + resized.shape[1],
                        max_loc[1] + resized.shape[0]
                    ))
        return self.non_max_suppression(detected_boxes)

    def process_board_image(self, image_path):
        board_img = cv2.imread(image_path)
        board_hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)
        tile_height = board_img.shape[0] // 5
        tile_width = board_img.shape[1] // 5

        for i in range(5):
            for j in range(5):
                y1, y2 = i * tile_height, (i + 1) * tile_height
                x1, x2 = j * tile_width, (j + 1) * tile_width
                tile = board_hsv[y1:y2, x1:x2]
                crowns = self.detect_crowns(tile)

                if len(crowns) > 0:
                    cv2.rectangle(board_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for (bx1, by1, bx2, by2) in crowns:
                        cv2.rectangle(board_img,
                                      (x1 + bx1, y1 + by1),
                                      (x1 + bx2, y1 + by2),
                                      (0, 0, 255), 2)

        return board_img

# Brug
if __name__ == "__main__":
    # Angiv paths til dine templates
    template_paths = [
        r"Reference_tiles\reference_crown_small1.jpg",
        r"Reference_tiles\reference_crown2.jpg",
        r"Reference_tiles\reference_crown.jpg"
    ]
    
    detector = CrownDetector(template_paths)

    image_paths = glob.glob(r"Cropped and perspective corrected boards\*.jpg")[:1]

    for path in image_paths:
        result_img = detector.process_board_image(path)
        cv2.imshow(f"Result for {path}", result_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()