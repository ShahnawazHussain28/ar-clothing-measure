import cv2
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin

from main import get_measurements_from_image

app = Flask(__name__)
cors = CORS(app, origins="*")

VERSION = "v-0.3.0"


@app.route("/")
@cross_origin()
def hello_world():
    return "Hello World! " + VERSION


@app.route("/get-measurements", methods=["POST"])
@cross_origin()
def process_image():
    try:
        if "image" not in request.files:
            return str({"error": "No image file found in request"}), 400

        image_file = request.files["image"]
        if not image_file.mimetype.startswith("image/"):
            return str({"error": "Invalid image format"}), 400

        img_string = image_file.read()
        image = cv2.imdecode(np.fromstring(img_string, np.uint8), cv2.IMREAD_UNCHANGED)

        measurements = get_measurements_from_image(image)
        if "error" in measurements:
            return str({"error": measurements["message"]}), 400

        res = {}
        for key, value in measurements.items():
            res[key] = round(value["distance"], 2)
        return str(res), 200

    except Exception as e:
        return str({"error": str(e)}), 500
