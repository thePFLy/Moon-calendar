import geocoder
import datetime
import ephem
import cv2
import numpy as np

def rotate(img, angle, center=None):
    height, width = img.shape[0:2]
    if center is None:
        center = (width//2, height//2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    dimensions = (width, height)
    return cv2.warpAffine(img, rotation_matrix, dimensions)

if __name__ == "__main__":
    g = geocoder.ip('me').latlng[0]

    moon_res = 500
    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    img = np.zeros((moon_res, moon_res, 3), dtype=np.uint8)
    cv2.rectangle(img,(0,0),(moon_res, moon_res),(255,0,255),-1)

    phase = round(((datetime.date.today() - datetime.date(1999, 8, 11)).days % 29.53059)/29.53059,2)
    print(phase)

    main_color = (255,255,255)
    secondary_color = (0,0,0)
    quarter_phase = 0 #0 to 1 float
    half_phase_start = 270 #angle (90 or 270)

    if phase <= 0.25:
        quarter_phase = 1 - phase

    elif phase > 0.25 and phase <= 0.5:
        quarter_phase = phase - 0.25

    elif phase > 0.5 and phase <= 0.75:
        quarter_phase = 1 - (phase - 0.5)

    else:
        quarter_phase = phase - 0.75

    if phase > 0.25 and phase <= 0.75:
        main_color = (0,0,0)
        secondary_color = (255,255,255)
        half_phase_start = 90

    print(quarter_phase)

    cv2.circle(img, (moon_res//2,moon_res//2), moon_res//2, main_color, -1)
    cv2.ellipse(img, (moon_res//2,moon_res//2), (int((moon_res//2)*quarter_phase), moon_res//2), 0, 0, 360, secondary_color, -1)
    cv2.ellipse(img, (moon_res//2,moon_res//2), (moon_res//2, moon_res//2), half_phase_start, 0, 180, secondary_color, -1)

    img = rotate(img, g)

    cv2.imshow("out", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


