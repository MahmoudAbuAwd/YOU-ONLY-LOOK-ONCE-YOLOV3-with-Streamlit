import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Paths to YOLOv3 files (ensure these files are in the same directory)
WEIGHTS_PATH = "yolov3.weights"
CONFIG_PATH = "yolov3.cfg"
NAMES_PATH = "coco.names"

# Load class names
with open(NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLOv3 model
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    return net

net = load_model()

# Function to perform object detection on an image
def detect_objects_image(image, confidence_threshold=0.5, nms_threshold=0.4):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

# Streamlit UI
st.title("YOLOv3 Object Detection")
st.write("Upload an image or enable video detection for real-time object detection using YOLOv3.")

# Sidebar for options
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert("RGB"))

        # Perform object detection
        with st.spinner("Detecting objects..."):
            result_image = detect_objects_image(image.copy())

        # Display results
        st.image(result_image, caption="Detected Objects", channels="BGR", use_column_width=True)

elif option == "Video":
    st.write("Press the 'Start' button to start real-time object detection from your webcam.")
    start_button = st.button("Start Detection")

    if start_button:
        # Start webcam video stream
        cap = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture video frame. Exiting...")
                break

            # Perform object detection on video frame
            result_frame = detect_objects_image(frame.copy())

            # Convert BGR to RGB for Streamlit
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            stframe.image(result_frame, channels="RGB", use_column_width=True)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        st.success("Stopped video stream.")

st.write("Press 'q' in the video stream to stop detection.")
