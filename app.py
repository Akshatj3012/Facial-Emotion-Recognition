# Import necessary libraries
from flask import Flask, request, jsonify, render_template_string  # Flask for web app, request for file upload, jsonify for API response
import numpy as np  # For image array manipulation
import cv2  # OpenCV for image decoding
from fer import FER  # FER for facial emotion recognition

# Initialize Flask app
app = Flask(__name__)

# Initialize the FER detector with MTCNN for better face detection
# mtcnn=True uses a more accurate face detector
# You can set mtcnn=False for faster but less accurate detection
# See: https://github.com/justinshenk/fer

detector = FER(mtcnn=True)

# HTML and JavaScript for the frontend UI
# This template:
# - Shows a webcam video
# - Draws bounding boxes and emotion labels on a canvas overlay
# - Sends frames to the backend for emotion detection
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
    // Get references to video and canvas elements
    const video = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const resultDiv = document.getElementById('result');

    // Request access to the user's webcam
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then(stream => { video.srcObject = stream; })
      .catch(err => { resultDiv.innerText = 'Camera access denied.'; });

    // Draw bounding boxes and emotion labels on the canvas
    function drawResults(results) {
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!Array.isArray(results)) return;
      results.forEach(face => {
        const [x, y, w, h] = face.box; // Face bounding box
        const emotions = face.emotions; // Emotion scores
        // Get the top emotion
        const top = Object.entries(emotions).sort((a,b)=>b[1]-a[1])[0];
        const label = `${top[0][0].toUpperCase()+top[0].slice(1)} (${Math.round(top[1]*100)}%)`;
        // Color by emotion
        const colorMap = {
          angry: '#ff3333', disgust: '#990099', fear: '#6600cc', happy: '#00e676', sad: '#2979ff', surprise: '#ffd600', neutral: '#bdbdbd'
        };
        ctx.strokeStyle = colorMap[top[0]] || '#00e5ff';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h); // Draw bounding box
        ctx.font = '20px Segoe UI';
        ctx.fillStyle = ctx.strokeStyle;
        // Draw label background for readability
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 28, textWidth + 10, 26);
        ctx.fillStyle = '#181c20';
        ctx.fillText(label, x + 5, y - 10); // Draw label text
      });
    }

    // Capture a frame from the video, send to backend, and draw results
    function sendFrame() {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = video.videoWidth;
      tempCanvas.height = video.videoHeight;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      tempCanvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        fetch('/predict', { method: 'POST', body: formData })
          .then(res => res.json())
          .then(data => {
            drawResults(data);
            resultDiv.innerText = data.length ? `${data.length} face(s) detected` : 'No face detected';
          })
          .catch(() => { resultDiv.innerText = 'Error detecting emotion.'; });
      }, 'image/jpeg');
    }

    // Send frames to backend at 2 FPS
    setInterval(sendFrame, 1000/2); // 2 FPS for smoother experience
  </script>
</body>
</html>
'''

# Route for the main page (serves the HTML UI)
@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_FORM)

# Route for emotion prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    # Read the uploaded image as a numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    # Detect faces and emotions in the image
    results = detector.detect_emotions(img)
    # Return the results as JSON
    return jsonify(results)

# Run the app locally (not used on Vercel)
if __name__ == "__main__":
    app.run(debug=True)
