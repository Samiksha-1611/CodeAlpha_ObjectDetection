import streamlit as st
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_webcam_detection(model):
    st.subheader("Webcam Detection + Deep SORT Tracking (Live)")

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    if start:
        st.session_state.run_webcam = True
    if stop:
        st.session_state.run_webcam = False

    FRAME_WINDOW = st.empty()

    if not st.session_state.run_webcam:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.session_state.run_webcam = False
        return

    # 🔹 Deep SORT tracker for webcam stream
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    while st.session_state.run_webcam:
        ret, frame_bgr = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # YOLO detection
        yolo_results = model(frame_bgr, verbose=False)[0]
        boxes = yolo_results.boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                w = x2 - x1
                h = y2 - y1
                detections.append(([float(x1), float(y1), float(w), float(h)],
                                   float(conf),
                                   int(cls)))

        # Deep SORT update
        tracks = tracker.update_tracks(detections, frame=frame_bgr)

        # Draw tracked boxes
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = map(int, (l, t, r, b))

            det_class = getattr(track, "det_class", None)
            label = f"ID {track_id}"
            if det_class is not None and det_class in model.names:
                label += f" {model.names[int(det_class)]}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

    cap.release()
