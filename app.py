import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Object Detection")
st.title("Real-Time Object Detection")
st.write("Press the button below to start the webcam detection.")


#Load the model trained on a dataset from roboflow , trained on COLAB 
model = YOLO("yolo_detection model/models/best.pt")

#Button to start webcam feed
start = st.button("Start Webcam Detection")

if start:
    cam = cv2.VideoCapture(0, cv2.CAP_ANY) #initialize camera from openCV
    stop = st.button("stop webcam detection") #Stop button to interrupt camera feed and end it 

    frame_placeholder = st.empty() #empty placeholder for webcam frame

    while cam.isOpened(): #Loop for capturing and detecting until camera is open 

        success , frame = cam.read() #reading the frame from camera feed 

        if not success: #End the webcam feed if interrupted unwillingly 
            st.write("Failed to capture video")
            break

        if stop: #willingly interrupt and end camera feed 
            cam.release()
            break

        results = model.predict(frame, conf=0.5) #Predict the object(S) in frame using the trained model

        annotated_frame= results[0].plot() #store the result in annotated frame 

        annotated_frame=cv2.cvtColor( #replacing the color - OpenCV BRG ( Blue green red) format use karta hai
            annotated_frame,          #aur Matplotlib ya streamlit RGB(Red green blue format) isliye conversion is important
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image( #replacing the empty placeholder with the annotated frame with predicted classes and bounding box
            annotated_frame,
            channels ="RGB",
            use_container_width = True
        )

    cam.release()