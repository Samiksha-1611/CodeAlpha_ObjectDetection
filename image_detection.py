import streamlit as st
from PIL import Image

def run_image_detection(model):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Now run detection
        results = model(image)
        st.write("Detection complete!")
        st.image(results.render()[0])  # Display YOLO output
    else:
        st.warning("Please upload an image.")

import streamlit as st
from PIL import Image
import numpy as np
import cv2 

def run_image_detection(model):
    uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="uploaded image", use_container_width=True)

        if st.button("Run Image Parsing"):
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result = model.predict(frame, verbose = False)
            annotated_image = result[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption = "Detected", use_container_width=True)