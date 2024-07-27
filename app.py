import cv2
import flask
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from main import get_measurements_from_image

app = Flask(__name__)
cors = CORS(app, origins="*")

print("v-0.1.0")


@app.route("/")
@cross_origin()
def hello_world():
    return "Hello World!"


@app.route("/get-measurements", methods=["POST"])
@cross_origin()
def process_image():
    print(request.files)
    headers = {"Access-Control-Allow-Origin": "*", "X-Content-Type-Options": "nosniff"}
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request"}), 400, headers
    image_file = request.files["image"]
    if not image_file.mimetype.startswith("image/"):
        return jsonify({"error": "Invalid image format"}), 400, headers
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
            headers,
        )
    res = {}
    for key in measurements:
        res[key] = round(measurements[key]["distance"], 2)
    print(res)
    return jsonify(res), 200, headers
