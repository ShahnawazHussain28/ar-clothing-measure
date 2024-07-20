import cv2
import numpy as np
from flask import Flask, jsonify, request

from main import get_measurements_from_image

app = Flask(__name__)

print("v-0.1.0")


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/get-measurements", methods=["POST"])
def process_image():
    print(request.files)
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    image_file = request.files["image"]
    if not image_file.mimetype.startswith("image/"):
        return jsonify({"error": "Invalid image format"}), 400
    image = cv2.imdecode(
        np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED
    )
    measurements = get_measurements_from_image(image)
    if measurements is None:
        return (
            jsonify(
                {"error": "Could not detect dress in the image. Please re-upload."}
            ),
            400,
        )
    res = {}
    for key in measurements:
        res[key] = measurements[key]["distance"]
    return jsonify(res), 200
