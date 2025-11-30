import cv2
from ultralytics import YOLO
import sys

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")  # make sure this path is correct

    # Decide source:
    # - No argument  -> webcam (0)
    # - 1st argument -> video file path
    if len(sys.argv) > 1:
        source = sys.argv[1]          # e.g. python realtime_tracking.py video.mp4
    else:
        source = 0                    # default: webcam

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot read frame.")
            break

        # Run YOLOv8 tracking (with internal tracker, e.g. ByteTrack)
        results = model.track(frame, persist=True, verbose=False, conf=0.5)


        # Draw bounding boxes, labels, and track IDs
        annotated_frame = results[0].plot()

        # Show the frame in a window
        cv2.imshow("YOLOv8 Object Detection & Tracking (press 'q' to quit)", annotated_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
