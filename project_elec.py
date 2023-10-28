import numpy as np
import cv2

from tkinter import Tk
from tkinter.filedialog import askopenfilenames


def find_moon(img):
    global blurred
    #ugly AF circle finding
    contrast_value = 100
    circles = None
    while circles is None or len(circles) != 1:
        ret,contrast = cv2.threshold(img,contrast_value,255,cv2.THRESH_BINARY)
        image = cv2.normalize(src=contrast, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 1000, param1=250, param2=70, minRadius=5, maxRadius=0)
        contrast_value += 10

        if contrast_value > 255:
            return []
    
    blurred = image
    return [int(n) for n in circles[0][0]]

def get_luminosity(circle, img):
    global rect
    #crop to keep moon
    cx = int(circle[0])
    cy = int(circle[1])
    r = int(circle[2])
    rect = img[cy-r:cy+r, cx-r:cx+r]

    #attempt to fill holes & iregularities in moon
    rect = cv2.morphologyEx(rect, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

    #evaluate luminosity of each half
    means_l = []
    means_r = []
    for i in rect:
        means_l.append(np.mean(i[:len(i)//2]))
        means_r.append(np.mean(i[len(i)//2:]))
    left = np.mean(means_l)/204
    right = np.mean(means_r)/204
    
    return left, right

def get_lunar_day(left, right):
    lum = 1-((left+right)/2)
    if left > right:
        day = (14.75*lum)+14.75
    else:
        day = 14.75*lum
    return day

def superpose(img1, img2):
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            img2[i][j] = img1[i][j]

def get_final_image(img):
    global blurred

    #pre-processing
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(greyscale, (5, 5), 1.4)

    #moon finding stuff
    moon_pos = find_moon(blurred)

    #skip if no moon is found ==TODO==: replace by a popup
    if len(moon_pos) == 0:
        return []

    luminosity = get_luminosity(moon_pos, blurred)
    left = luminosity[0]
    right = luminosity[1]
    day = get_lunar_day(left, right)

    tmp_rect = cv2.resize(rect, (len(rect)//2, len(rect[0])//2)) 
    superpose(tmp_rect, img)

    #construct output image
    cx, cy, r = moon_pos
    cv2.line(img, (0,0), (0,len(tmp_rect)), (255, 0, 255), 2)
    cv2.line(img, (len(tmp_rect)//2,0), (len(tmp_rect)//2,len(tmp_rect)), (255, 0, 255), 2)
    cv2.line(img, (len(tmp_rect),0), (len(tmp_rect),len(tmp_rect)), (255, 0, 255), 2)
    cv2.line(img, (0,0), (len(tmp_rect), 0), (255, 0, 255), 2)
    cv2.line(img, (0,len(tmp_rect)), (len(tmp_rect), len(tmp_rect)), (255, 0, 255), 2)
    cv2.putText(img, f'Day: {int(day)} L: {round(left*100,2)}% R: {round(right*100,2)}%', (10, len(tmp_rect) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA) 
    cv2.circle(img, (cx,cy), r, (255, 0, 255), 3)

    return img

if __name__ == "__main__":
    Tk().withdraw()
    files = askopenfilenames()
    
    for f in files:
        initial = cv2.imread(f)
        global rect, blurred

        out_img = get_final_image(initial)
        if len(out_img) == 0:
            continue

        #show img and save it
        cv2.namedWindow(f, cv2.WINDOW_NORMAL)
        cv2.imshow(f, out_img)
        file = "".join(f.split(".")[:-1])+"_out."+f.split(".")[-1]
        print(file)
        cv2.imwrite(file, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()                                                                                                    