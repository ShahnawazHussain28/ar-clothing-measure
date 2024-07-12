from typing import List
import cv2
import numpy as np
from utils import draw_points_on_image, get_measurements, draw_measurements

img = cv2.imread("tshirt.jpg")
# resize the image by a factor of 0.5
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

canny = cv2.Canny(blurred, 100, 200)

# dilate and erode
kernel = np.ones((2, 2))
dilated = cv2.dilate(canny, kernel, iterations=3)
eroded = cv2.erode(dilated, kernel, iterations=3)

# detect contour

contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contour: List = contours[0]

blank_image = np.zeros(gray.shape, np.uint8)
blank_image = cv2.drawContours(blank_image, [contour], 0, (255, 255, 255), 1)

goodFeaturesToTrack = cv2.goodFeaturesToTrack(eroded, 100, 0.1, 5)
goodFeaturesToTrack = np.int_(goodFeaturesToTrack)

boundingbox = cv2.boundingRect(goodFeaturesToTrack)
blank_image = cv2.rectangle(
    blank_image,
    (boundingbox[0], boundingbox[1]),
    (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
    (255, 255, 255),
    1,
)

points = goodFeaturesToTrack.reshape(-1, 2)

measurements = get_measurements(points)

for key in measurements:
    img = draw_measurements(img, measurements[key])


blank_image = draw_points_on_image(blank_image, points, (255, 255, 255), 5)

cv2.imshow("imageContour", blank_image)
cv2.imshow("img", img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
