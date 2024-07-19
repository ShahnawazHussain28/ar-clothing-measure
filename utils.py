import cv2
from cv2.typing import MatLike, Point
import numpy as np


def draw_points_on_image(
    image: MatLike,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
):
    """Draws points on image"""
    for point in points:
        x, y = point
        image = cv2.circle(image, (x, y), radius, color, -1)
    return image


def draw_measurements(
    image: MatLike, measurement: dict, color: tuple[int, int, int] = (255, 180, 180)
):
    """Draws measurements on image"""
    image = draw_points_on_image(image, [measurement["start"]], color, 5)
    image = draw_points_on_image(image, [measurement["end"]], color, 5)
    image = cv2.line(image, measurement["start"], measurement["end"], color, 2)
    mid: Point = (np.array(measurement["start"]) + np.array(measurement["end"])) // 2
    image = cv2.putText(
        image,
        f"{int(measurement['distance'])}",
        (mid[0] + 2, mid[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return image


def get_bottom_left_right(points):
    """Returns bottom left and right points"""
    bottom_left = max(points, key=lambda p: p[1] - p[0])
    bottom_right = max(points, key=lambda p: p[0] + p[1])
    return bottom_left, bottom_right


def get_left_shoulder(points):
    """Returns left shoulder point"""
    left_shoulder = min(points, key=lambda p: p[0] - p[1])
    return left_shoulder


def get_right_shoulder(points):
    """Returns right shoulder point"""
    right_shoulder = min(points, key=lambda p: p[0] + p[1])
    return right_shoulder


def calculate_width(points, metric_per_pixel):
    """Returns width"""
    bottom_left, bottom_right = points["bottom_left"], points["bottom_right"]
    dist = (
        np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
        / metric_per_pixel
    )
    return {"start": bottom_left, "end": bottom_right, "distance": dist}


def calculate_shoulder_length(points, metric_per_pixel):
    """Returns shoulder length"""
    left_shoulder = get_left_shoulder(points)
    right_shoulder = get_right_shoulder(points)
    dist = (
        np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        / metric_per_pixel
    )
    return {"start": left_shoulder, "end": right_shoulder, "distance": dist}


def get_collar_points(points: list[tuple[int, int]]):
    """Returns top 2 points as collar points"""
    p = sorted(points, key=lambda p: p[1])
    collars = [p[0], p[1]]
    collars = sorted(collars, key=lambda p: p[0])
    return collars


def calculate_collar(points, metric_per_pixel):
    """Returns length"""
    left_collar, right_collar = points["left_collar"], points["right_collar"]
    dist = (
        np.linalg.norm(np.array(left_collar) - np.array(right_collar))
        / metric_per_pixel
    )
    return {"start": left_collar, "end": right_collar, "distance": dist}


def calculate_length(points, metric_per_pixel):
    """Returns length"""
    bottom_left, collar_left = points["bottom_left"].copy(), points["left_collar"]
    bottom_left[0] = collar_left[0]
    dist = (
        np.linalg.norm(np.array(collar_left) - np.array(bottom_left)) / metric_per_pixel
    )
    return {"start": collar_left, "end": bottom_left, "distance": dist}


def get_measurements(points, metric_per_pixel):
    """Returns measurements from points"""
    measurements = {}
    poi = {}
    poi["bottom_left"], poi["bottom_right"] = get_bottom_left_right(points)
    poi["left_collar"], poi["right_collar"] = get_collar_points(points)

    measurements["width"] = calculate_width(poi, metric_per_pixel)
    measurements["collar"] = calculate_collar(poi, metric_per_pixel)
    measurements["length"] = calculate_length(poi, metric_per_pixel)
    # measurements["shoulder_length"] = calculate_shoulder_length(points)
    return measurements
