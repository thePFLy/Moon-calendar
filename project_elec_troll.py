import geocoder
import datetime
import ephem
import cv2
import numpy as np

if __name__ == "__main__":
    g = geocoder.ip('me')
    print(g.latlng)

    date = ephem.Date(datetime.date.today())
    last_moon = ephem.previous_new_moon(date)

    img = np.zeros((500,500), dtype=int)
    cv2.imshow("out", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

    print(date - last_moon)


