import cv2
import numpy as np
from PIL import Image
import webcolors


def getColorFromCloth(image, contour):
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, mask)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    return get_main_color(masked_image)


def closest_color(requested_color):
    min_colors = {}
    try:
        for key, name in webcolors.css2_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]
    except Exception:
        return "Unidentified Color"


def get_main_color(image: np.ndarray):
    img = Image.fromarray(image)
    dom_color = sorted(img.getcolors(2**24), reverse=True)[1][1]
    return closest_color(dom_color)
