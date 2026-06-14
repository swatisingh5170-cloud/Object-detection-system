# Object Detection Web App

## Project Overview

This repository contains a simple Streamlit web app that uses YOLOv8 for object detection on webcam frames. The app is designed to capture webcam video, run inference on each frame, and display detected objects with bounding boxes.

## Repository Content

- `app.py` - main Streamlit app for webcam-based object detection
- `README.md` - project documentation and usage notes
- `yolo_detection model/` - contains model files and a webcam test script
  - `models/best.pt` - trained YOLOv8 model weights
  - `webcam_test.py` - separate OpenCV webcam test script

## Things Learned

- how to load and run a YOLOv8 model using Ultralytics
- basic OpenCV webcam capture and color conversion
- building a Streamlit UI with buttons and image display
- how Streamlit reruns scripts on user interaction
- the importance of correct model file paths for app startup

## Future Function

Planned improvements for the app:

- a proper live webcam stream instead of a blocking loop
- add frame-by-frame control and display FPS
- provide model selection and confidence threshold sliders
- improve error handling when the webcam or model file is missing
- support webcam access directly in the browser using a WebRTC integration

## Difficulty I Faced

- `st.button` may not appear if the script fails before reaching that line
- the original app uses a blocking `while` loop, which is not compatible with Streamlit's rerun model
- the model path must be correct or the app crashes before UI rendering
- installing extra dependencies for browser-based webcam streaming can be tricky on Windows

## Tech Used

- Python
- Streamlit
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- `matplotlib` (optional visualization support)

## Development Pipeline

1. write `app.py` to capture webcam frames and show a simple UI
2. load YOLOv8 model weights from `yolo_detection model/models/best.pt`
3. run object detection on each frame and annotate results
4. test locally with Streamlit using `python -m streamlit run app.py`
5. debug issues such as missing model files or webcam access failures
6. document the app behavior and future improvements in `README.md`
