# 🚀 YOLOv8 Object Detection + Deep SORT Tracking (Streamlit App)

This project is a complete object detection and tracking system using **YOLOv8** and **Deep SORT**, wrapped inside an easy-to-use **Streamlit web application**.  
It supports **image detection**, **video detection**, and **real-time webcam tracking** with unique object IDs.

---

## ⭐ Features

### 🔍 1. Image Detection  
Upload an image and detect objects instantly using YOLOv8.  
(Implemented in `image_detection.py`) :contentReference[oaicite:0]{index=0}  

### 🎞 2. Video Detection + Deep SORT Tracking  
Upload a video and get a new video file with tracked objects and unique IDs.  
(Implemented in `video_detection.py`) :contentReference[oaicite:1]{index=1}  

### 🎥 3. Real-time Webcam Tracking  
Track moving objects in real time with Deep SORT.  
(Implemented in `webcam_detection.py`) :contentReference[oaicite:2]{index=2}  

### ⚡ 4. Standalone Python Scripts  
- `realtime_tracking.py` → command-line live tracking (webcam/video) :contentReference[oaicite:3]{index=3}  
- `image_detection.py`, `video_detection.py`, `webcam_detection.py` → modular functions  
- `main.py` → Streamlit UI entry point :contentReference[oaicite:4]{index=4}  

### 🧠 Model Used  
- **YOLOv8n model** (`yolov8n.pt`)  

---

## 📂 Project Structure

.
├── main.py # Streamlit UI with mode selection
├── image_detection.py # Image detection module
├── video_detection.py # Video detection + Deep SORT
├── webcam_detection.py # Live webcam tracking
├── realtime_tracking.py # CLI-based YOLO tracking
├── check_version # Optional version check script
├── install.txt # Required pip installation commands
├── yolov8n.pt # YOLO model file


---

## 🛠 Installation

Install the required modules:  
(from `install.txt`) :contentReference[oaicite:5]{index=5}

```bash
pip install streamlit
pip install pillow
pip install ultralytics
pip install numpy
pip install opencv-python
pip install imageio
pip install deep-sort-realtime

🚀 How to Run the App
1️⃣ Start the Streamlit web app
streamlit run main.py

2️⃣ Choose a mode
Image Detection
Video Detection (Upload)
Webcam Detection (Real-time)

Image Detection Workflow(From image_detection.py)
Upload image
YOLO processes the frame
Annotated image displayed

Video Detection + Deep SORT Tracking(From video_detection.py)
Upload a video
YOLO detects objects per frame
Deep SORT assigns unique IDs
Video is processed frame by frame and displayed

Real-time Webcam Tracking(From webcam_detection.py)
Opens webcam
YOLO detects objects each frame
Deep SORT keeps track of object IDs
Annotated frames shown live

Running Standalone Real-Time Tracking Script
python realtime_tracking.py

Credits
Built using:
YOLOv8 (Ultralytics)
Deep SORT
OpenCV
Streamlit

