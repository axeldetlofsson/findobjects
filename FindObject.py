import cv2
import numpy as np
#from cv2 import cv

#method = CV_TM_SQDIFF_NORMED
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Read the images from the file
small_image = cv2.imread('files/small.png', cv2.IMREAD_GRAYSCALE)
#large_image = cv2.imread('files/big.png')


cap = cv2.VideoCapture("files/lolvid.mp4")
w, h = small_image.shape[::-1]

while True:
    ret, frame = cap.read()

    if not ret is False:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    res = cv2.matchTemplate(gray_frame, small_image, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.7)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
