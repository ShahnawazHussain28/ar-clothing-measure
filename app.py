import cv2
import numpy as np
from flask import Flask, jsonify, request

from main import get_measurements_from_image

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/get-measurements", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    print(request.files)
    image_file = request.files["image"]
    print(image_file)
    if not image_file.mimetype.startswith("image/"):
        return jsonify({"error": "Invalid image format"}), 400
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    measurements = get_measurements_from_image(image)
    if measurements is None:
        return jsonify({"error": "No shirt found in image"}), 400
    return jsonify({"measurements": measurements})
