import cv2 as cv
import numpy as np
import os

def callback(input):
    pass

def canny_edge():
    root = os.getcwd()
    img_path = os.path.join(root, r'C:\Users\carle\Desktop\python_work\Miniprojekt_3\Cropped and perspective corrected boards\1.jpg')
    img = cv.imread(img_path)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    win_name = 'canny'
    cv.namedWindow(win_name)
    cv.createTrackbar('min_thres', win_name, 0, 255, callback)
    cv.createTrackbar('max_thres', win_name, 0, 255, callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        min_thres = cv.getTrackbarPos('min_thres', win_name)
        max_thres = cv.getTrackbarPos('max_thres', win_name)
        

        canny_edge_img = cv.Canny(img, min_thres, max_thres)
        cv.imshow(win_name, canny_edge_img)


        contours, _ = cv.findContours(canny_edge_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


        img_contours = img.copy()
        for contour in contours:
            if cv.contourArea(contour) > 100:  #filter til små contours
           
                epsilon = 0.02 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True) #approximer kronerne til et polygon
                if len(approx) > 6:  # mængden af kanter i polygonet
                    cv.drawContours(img_contours, [approx], 0, (0, 255, 0), 3)  # tegn contours

       
        cv.imshow('Contours', img_contours)

    cv.destroyAllWindows()

if __name__ == '__main__':
    canny_edge()
