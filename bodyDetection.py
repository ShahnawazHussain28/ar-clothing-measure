from cvzone.PoseModule import PoseDetector
import cv2

from utils import (
    ellipse_circumference,
    get_metric_per_pixel_customer,
    get_belly,
)

detector_front = PoseDetector(
    staticMode=False,
    modelComplexity=2,
    smoothLandmarks=True,
    enableSegmentation=True,
    smoothSegmentation=True,
    detectionCon=0.5,
    trackCon=0.5,
)
detector_side = PoseDetector(
    staticMode=False,
    modelComplexity=2,
    smoothLandmarks=True,
    enableSegmentation=True,
    smoothSegmentation=True,
    detectionCon=0.5,
    trackCon=0.5,
)

HEIGHT = 168

front = cv2.imread("./images/front2.jpg")
side = cv2.imread("./images/side3.jpg")
front = cv2.resize(front, (int(front.shape[1] * 0.25), int(front.shape[0] * 0.25)))
side = cv2.resize(side, (int(side.shape[1] * 0.25), int(side.shape[0] * 0.25)))
front = detector_front.findPose(front)
front_lmlist, _, front_results = detector_front.findPosition(front)
side = detector_side.findPose(side)
side_lmlist, _, side_results = detector_side.findPosition(side)


if front_lmlist is not None and side_lmlist is not None:
    mpp_front = get_metric_per_pixel_customer(front_lmlist, HEIGHT)
    mpp_side = get_metric_per_pixel_customer(side_lmlist, HEIGHT)

    chest_front = (
        front_lmlist[24][0],
        int((front_lmlist[24][1] - front_lmlist[12][1]) * 0.333 + front_lmlist[12][1]),
    )
    belly_front = (
        front_lmlist[24][0],
        int((front_lmlist[24][1] - front_lmlist[12][1]) * 0.666 + front_lmlist[12][1]),
    )
    chest_side = (
        side_lmlist[24][0],
        int((side_lmlist[24][1] - side_lmlist[12][1]) * 0.333 + side_lmlist[12][1]),
    )
    belly_side = (
        side_lmlist[24][0],
        int((side_lmlist[24][1] - side_lmlist[12][1]) * 0.666 + side_lmlist[12][1]),
    )

    belly_major = get_belly(belly_front, front_results.segmentation_mask, mpp_front) / 2
    belly_minor = get_belly(belly_side, side_results.segmentation_mask, mpp_side) / 2
    belly_circumference = ellipse_circumference(belly_major, belly_minor)
    belly = (2 * belly_major) + (belly_circumference / 2)

    print(belly_major, belly_minor, belly)
    front = cv2.circle(front, chest_front, 5, (0, 0, 0), -1)
    front = cv2.circle(front, belly_front, 5, (0, 0, 0), -1)
    side = cv2.circle(side, chest_side, 5, (0, 0, 0), -1)
    side = cv2.circle(side, belly_side, 5, (0, 0, 0), -1)

cv2.imshow("segmentation", front_results.segmentation_mask)
cv2.imshow("segmentation side", side_results.segmentation_mask)
cv2.imshow("Image front", front)
cv2.imshow("Image side", side)
cv2.waitKey(0)
