from typing import List
import cv2
import numpy as np

img = cv2.imread("tshirt.jpg")
# resize the image by a factor of 0.5
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((15, 15), np.uint8)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
thres = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contour: List = contours[0]


def get_bottom_left_point(cont):
    compare_val = 0
    bl = cont[0][0]
    index = 0
    for i, point in enumerate(cont):
        ymx = point[0][1] - point[0][0]
        if ymx > compare_val:
            index = i
            compare_val = ymx
            bl = point[0]
    return [bl, index]


bottom_left, bl_idx = get_bottom_left_point(contour)
while True:
    # draw circle at bottom left
    img = cv2.circle(img, contour[bl_idx][0], 5, (0, 0, 255), -1)
    bl_idx = (bl_idx + 1) % len(contour)
    cv2.imshow("image", img)
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


# print(bottom_left)
#
# blank_image = np.zeros(img.shape, np.uint8)
#
# cv2.imshow("imageOrig", img)
# img = cv2.drawContours(blank_image, [contour], 0, (0, 255, 0), 3)
# cv2.imshow("imageEdge", img)
# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()
