# api/index.py
# Vercel Python Function entry point for emotion detection
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
from fer import FER

app = Flask(__name__)
detector = FER(mtcnn=True)

UPLOAD_FORM = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Live Emotion Detection</title>
  <style>
    body { background: #181c20; color: #fff; font-family: 'Segoe UI', sans-serif; text-align: center; }
    h2 { margin-top: 30px; }
    #container { display: inline-block; position: relative; }
    #video, #overlay { border-radius: 12px; box-shadow: 0 4px 24px #0008; }
    #overlay { position: absolute; left: 0; top: 0; }
    #result { margin-top: 18px; font-size: 1.1em; }
  </style>
</head>
<body>
  <h2>🧠 Live Emotion Detection</h2>
  <div id="container">
    <video id="video" width="640" height="480" autoplay muted></video>
    <canvas id="overlay" width="640" height="480"></canvas>
  </div>
  <div id="result"></div>
  <script>
    // ...existing code...
  </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    results = detector.detect_emotions(img)
    return jsonify(results)

# Vercel will use the 'app' object as the entry point
