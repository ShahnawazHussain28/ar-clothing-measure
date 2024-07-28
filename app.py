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
            erres = {"error": "No image file found in request"}
            return str(erres), 400

        image_file = request.files["image"]
        if not image_file.mimetype.startswith("image/"):
            erres = {"error": "Invalid image format"}
            return str(erres), 400

        img_string = image_file.read()
        image = cv2.imdecode(np.fromstring(img_string, np.uint8), cv2.IMREAD_UNCHANGED)

        measurements = get_measurements_from_image(image)
        if measurements is None:
            erres = {"error": "Could not detect dress in the image. Please re-upload."}
            return str(erres), 400

        res = {}
        for key in measurements:
            res[key] = round(measurements[key]["distance"], 2)
        return str(res), 200

    except Exception as e:
        erres = {"error": str(e)}
        return str(erres), 500
