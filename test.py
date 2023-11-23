import numpy as np
from math import dist
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2

def sobel(img):
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    wheighted = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    binary = cv2.adaptiveThreshold(wheighted, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,size,0)
    return binary

def find_circle(img):
    circle_max = min(img.shape[0],img.shape[1])
    circle_min = int(circle_max*0.05)

    #edge detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges = sobel(binary)
    edges = cv2.dilate(edges, kernel, 1)

    #correlation convolutions
    candidates = []
    for i in range(circle_max,circle_min,-10):
        r = i//2
        #generate circle
        circle = np.zeros((i, i), dtype=np.uint8)
        cv2.circle(circle, (r,r), r, (255, 255, 255), 1)

        #correlate
        matchup = cv2.matchTemplate(edges, circle, cv2.TM_CCORR)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(matchup) #DO NOT TOUTCH!
        top_x = maxLoc[0]
        top_y = maxLoc[1]

        #eval circle luminosity
        rect = edges[top_y:top_y+i, top_x:top_x+i]
        filt = cv2.bitwise_and(circle,rect)
        lum = np.mean(filt)*100

        #save values (center x, center y, radius, luminosity)
        candidates.append([top_x+r, top_y+r, r, int(lum)])
    
    candidates = np.array(candidates)# do not remove np array is required
    luminosities = candidates[:,3]

    #find local maximums
    peaks, _ = find_peaks(luminosities, distance=50, height=np.mean(luminosities)/2)

    #the biggest local max is our circle
    data = candidates[peaks[0]]
    r = data[2]
    cx = data[0]
    cy = data[1]
    return (cx,cy,r)

if __name__ == "__main__":
    initial = cv2.imread("moons/moon5.png")

    #pre pocessing
    greyscale = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyscale, (7, 7), 0)
    size = min(initial.shape[0],initial.shape[1]) - 1 + (min(initial.shape[0],initial.shape[1])%2)
    binary = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,size,0)

    #clean artefacts
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(size*0.01),int(size*0.01)))
    for i in range(5):
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1)

    cx, cy, r = find_circle(binary)

    #display
    output = initial
    output = cv2.circle(output, (cx,cy), r, (255, 255, 0), 1)
    output = cv2.circle(output, (cx,cy), 2, (255, 255, 0), -1)

    #add luminosity graph (slow things down ALOT)
    # for i in range(len(out)):
    #     color = (255, 0, 255)
    #     if i == peaks[0]:
    #         color = (255, 255, 0)
    #     output = cv2.line(output, (i,len(output)), (i,len(output)-(out[:,3][i]//8)), color, 1)

    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()