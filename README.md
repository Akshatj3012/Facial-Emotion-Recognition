# Live Emotion Detection Web App

This project is a real-time emotion detection web application using Flask, OpenCV, and the FER (Facial Emotion Recognition) library. It allows users to use their webcam in the browser, detects faces and their emotions live, and displays bounding boxes and emotion labels on the video stream.

## Features
- **Live Webcam Detection:** Uses the browser's webcam to capture video frames in real time.
- **Face Detection & Emotion Recognition:** Detects faces and classifies emotions (angry, disgust, fear, happy, sad, surprise, neutral) using the FER library.
- **Attractive UI:** Modern, responsive interface with color-coded bounding boxes and emotion labels.
- **REST API:** `/predict` endpoint accepts image frames and returns detected faces and emotions as JSON.
- **Deployable on Vercel:** Ready for serverless deployment.

## How It Works
1. The frontend uses HTML5 and JavaScript to access the user's webcam and capture frames.
2. Each frame is sent to the Flask backend via AJAX POST to `/predict`.
3. The backend uses FER to detect faces and emotions in the frame and returns the results as JSON.
4. The frontend draws bounding boxes and emotion labels on a canvas overlay above the video.

## File Structure
- `app.py` - Main Flask application with both backend and frontend code.
- `requirements.txt` - Python dependencies.
- `vercel.json` - Vercel deployment configuration.

## API
### `POST /predict`
- **Request:** Multipart form with an image file (JPEG/PNG).
- **Response:** JSON array of detected faces, each with:
  - `box`: `[x, y, w, h]` (bounding box coordinates)
  - `emotions`: `{emotion: score, ...}`

## Running Locally
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the app:
   ```sh
   python app.py
   ```
3. Open your browser at `http://localhost:5000`.

## Deployment
- Deploy the project folder to Vercel. The `vercel.json` configures the Python runtime and entry point.

## Requirements
- Python 3.7+
- Flask
- OpenCV (`opencv-python`)
- numpy
- fer

## Credits
- [FER: Facial Emotion Recognition](https://github.com/justinshenk/fer)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)

---

**Made with ❤ for real-time emotion AI!**
