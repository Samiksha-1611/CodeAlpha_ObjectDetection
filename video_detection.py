import streamlit as st
import cv2
import numpy as np
import tempfile
import imageio
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_video_detection(model):
    st.subheader("Video Detection + Deep SORT Tracking")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Save uploaded video to a temporary file
        tempof = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tempof.write(uploaded_file.read())
        tempof.close()
        video_path = tempof.name

        if st.button("Decode video objects (Deep SORT)"):
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()["fps"]

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_file.close()
            writer = imageio.get_writer(out_file.name, fps=fps)

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                total_frames = reader.count_frames()
            except Exception:
                total_frames = None   # some formats don't support this

            # 🔹 Create Deep SORT tracker
            tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

            for i, frame_rgb in enumerate(reader):
                # imageio gives RGB; convert to BGR for OpenCV/YOLO
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # 🔹 YOLOv8 detection (no internal tracking)
                yolo_results = model(frame_bgr, verbose=False)[0]
                boxes = yolo_results.boxes

                detections = []
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)

                    # Deep SORT expects: ([x, y, w, h], confidence, class)
                    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                        w = x2 - x1
                        h = y2 - y1
                        detections.append(([float(x1), float(y1), float(w), float(h)],
                                           float(conf),
                                           int(cls)))

                # 🔹 Update Deep SORT tracker
                tracks = tracker.update_tracks(detections, frame=frame_bgr)

                # 🔹 Draw boxes + IDs
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    l, t, r, b = track.to_ltrb()   # left, top, right, bottom
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

                # back to RGB for saving with imageio
                out_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                writer.append_data(out_frame_rgb)

                # Progress UI
                if total_frames:
                    progress = min((i + 1) / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {i+1}/{total_frames} ...")
                else:
                    status_text.text(f"Processing frame {i+1} ...")

            writer.close()
            status_text.text("Processing complete! Below is the Deep SORT tracked video:")
            st.video(out_file.name)
    else:
        st.info("Please upload a video file to start.")
