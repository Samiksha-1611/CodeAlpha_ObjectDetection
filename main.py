import streamlit as st
from ultralytics import YOLO
from image_detection import run_image_detection
from video_detection import run_video_detection
from webcam_detection import run_webcam_detection

st.set_page_config(page_title="YOLO model by JIT students", layout="wide")
st.title("YOLO Detection App")

model = YOLO("yolov8n.pt")

option = st.sidebar.selectbox(
    "Select Mode",
    ["Image", "Video (Upload)", "Webcam (Real-time)"]
)

if option == "Image":
    run_image_detection(model)

elif option == "Video (Upload)":
    run_video_detection(model)

elif option == "Webcam (Real-time)":
    run_webcam_detection(model)
