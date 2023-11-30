import numpy as np
from tkinter import *
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter import messagebox
import cv2

def preprocessing(img):
    #pre pocessing
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(greyscale, (7, 7), 0)
    size = min(img.shape[0],img.shape[1]) - 1 + (min(img.shape[0],img.shape[1])%2)
    binary = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,size,0)

    #clean artefacts
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(size*0.01),int(size*0.01)))
    for i in range(5):
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1)

    return binary

def sobel(img):
    """
    return the binarized sobel image
    sobel is done in both direction then weighted
    """
    size = min(img.shape[0],img.shape[1]) - 1 + (min(img.shape[0],img.shape[1])%2)
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    wheighted = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    binary = cv2.adaptiveThreshold(wheighted, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,size,0)
    return binary

def find_circle(img):
    global edges
    circle_max = min(img.shape[0],img.shape[1])
    circle_min = int(circle_max*0.025)

    #edge detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    edges = sobel(binary)
    edges = cv2.dilate(edges, kernel, 1)

    #correlation convolutions
    candidates = []
    for i in range(circle_max,circle_min,-max(10,int((circle_max-circle_min)*0.01))):
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
        lum = np.mean(filt)*50

        #save values (center x, center y, radius, luminosity)
        candidates.append([top_x+r, top_y+r, r, int(lum)])
    
    candidates = np.array(candidates)# do not remove np array is required
    luminosities = candidates[:,3]

    #find local maximums
    #, height=np.mean(luminosities)/2
    #peaks, _ = find_peaks(luminosities, prominence=1.5, distance=50)
    peaks = sort_array(luminosities)

    global circles #required for graph display
    circles = candidates

    return candidates[peaks]

def sort_array(arr):
    """
    return an array of local maximums, sorted on the 'spikiness' of the points
    """
    result = []
    for i in range(3, len(arr) - 3):
        if all(arr[i] > x for x in arr[i-3:i]) and all(arr[i] > x for x in arr[i+1:i+4]):
            result.append(i)
    result.sort(key=lambda x: arr[x] - ((sum(arr[x-3:x]) + sum(arr[x+1:x+4])) / 6), reverse=True)
    return result

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
    if right > left:
        day = 14.75*((left+right)/2)
    else:
        day = 14.75*(1-((left+right)/2))+14.75
    return day

def superpose(img1, img2):
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            img2[i][j] = img1[i][j]

def output_img(img, graph, circle=None, day=None, left=None, right=None):
    size = min(img.shape[0],img.shape[1]) - 1 + (min(img.shape[0],img.shape[1])%2)
    
    if circle is not None:
        cv2.circle(img, (circle[0],circle[1]), circle[2], (255, 255, 0), 1)
        cv2.circle(img, (circle[0],circle[1]), 2, (255, 255, 0), -1)
        text_size = int(size*0.07)

    if (day is not None) or (day is not None) or (day is not None):
        tmp_rect = cv2.resize(rect, (int(size*0.25), int(size*0.25))) 
        superpose(tmp_rect, img)
        cv2.line(img, (0,0), (0, len(tmp_rect)), (255, 0, 255), 1)
        cv2.line(img, (len(tmp_rect)//2,0), (len(tmp_rect)//2,len(tmp_rect)), (255, 0, 255), 1)
        cv2.line(img, (len(tmp_rect),0), (len(tmp_rect),len(tmp_rect)), (255, 0, 255), 1)
        cv2.line(img, (0,0), (len(tmp_rect), 0), (255, 0, 255), 1)
        cv2.line(img, (0, len(tmp_rect)), (len(tmp_rect), len(tmp_rect)), (255, 0, 255), 1)
        cv2.putText(img, f'Day: {int(day)} L: {round(left*100,2)}% R: {round(right*100,2)}%', (5, int(size*0.25)+(text_size//2)), cv2.FONT_HERSHEY_SIMPLEX, text_size/100, (255, 0, 255), 1, cv2.LINE_AA)

    #add luminosity graph (slow things down ALOT)
    graph = np.array(graph)
    max_lum = max(graph[:,3])
    for i in range(len(graph)):
        color = (255, 0, 255)
        if circle is not None and np.array_equal(graph[i],data[0]):
            color = (255, 255, 0)
        val = int(len(img) - ((graph[i][3]/max_lum)*(len(img)/4)))
        cv2.line(img, (i*2,len(img)), (i*2,val), color, 1)
    return img

# =====================================/ MAIN /==================================================================

if __name__ == "__main__":
    global circles
    global binary
    global edges

    Tk().withdraw()
    files = askopenfilenames()

    for f in files:
        try:
            initial = cv2.imread(f)
            size = max(len(initial),len(initial[0]))
            if (size > 2000):
                scale = 1-((size-2000)/size)
                initial = cv2.resize(initial, (int(len(initial[0])*scale), int(len(initial)*scale)))
            binary = preprocessing(initial)
            data = find_circle(binary)
            if len(data) > 0:
                left, right = get_luminosity(data[0], binary)
                day = get_lunar_day(left, right)
                output = output_img(initial, circles, data[0], day, left, right)
            else:
                output = output_img(initial, circles)
            cv2.namedWindow(f, cv2.WINDOW_NORMAL)
            cv2.imshow(f, output)
        except TypeError:
            messagebox.showinfo("Not an Image", f"{f} is not an image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()