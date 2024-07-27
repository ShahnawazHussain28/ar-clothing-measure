from cvzone.PoseModule import PoseDetector
import cv2

from utils import get_metric_per_pixel_customer, get_shoulder_length

cap = cv2.VideoCapture(0)

detector = PoseDetector(
    staticMode=False,
    modelComplexity=2,
    smoothLandmarks=True,
    enableSegmentation=True,
    smoothSegmentation=True,
    detectionCon=0.5,
    trackCon=0.5,
)

HEIGHT = 168

while True:
    success, img = cap.read()
    if not success or img is None:
        continue
    img = detector.findPose(img)
    lmList, bboxInfo, results = detector.findPosition(
        img, draw=True, bboxWithHands=False
    )
    if lmList:
        center = bboxInfo["center"]
        head_pos = lmList[0][1]
        metric_per_pixel = get_metric_per_pixel_customer(lmList, HEIGHT)
        shoulder_length = get_shoulder_length(lmList, metric_per_pixel)
        print(shoulder_length)
        cv2.imshow("segmentation", results.segmentation_mask)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
