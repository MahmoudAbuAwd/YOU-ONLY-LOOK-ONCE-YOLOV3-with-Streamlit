import cv2
import numpy as np

# Paths to YOLOv3 files (ensure these are correct)
WEIGHTS_PATH = "yolov3.weights"  # Path to yolov3.weights
CONFIG_PATH = "yolov3.cfg"       # Path to yolov3.cfg
NAMES_PATH = "coco.names"        # Path to coco.names

# Load COCO class names
with open(NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLOv3 network
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

# Use GPU if available (Optional: Requires OpenCV built with CUDA support)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Set parameters
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for predictions
NMS_THRESHOLD = 0.4         # Non-max suppression threshold

# Get output layer names
layer_names = net.getLayerNames()
output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start capturing video from webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process frames in real time
while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    height, width, _ = frame.shape

    # Convert the frame to a blob (preprocessing for YOLOv3)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the YOLO network
    outputs = net.forward(output_layer_names)

    # Extract detections
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Draw bounding boxes and labels
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the video frame with detections
    cv2.imshow("YOLOv3 Real-Time Object Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
