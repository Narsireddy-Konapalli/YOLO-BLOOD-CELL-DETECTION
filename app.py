import streamlit as st
import cv2
import numpy as np
import yaml
from PIL import Image
from yaml.loader import SafeLoader


st.title("YOLOv5 Object Detection App")

with open('data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)
labels = data_yaml['names']

yolo = cv2.dnn.readNetFromONNX('Model/weights/best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    rows, cols, _ = img.shape

    input_size = (640, 640)
    input_image = cv2.resize(img, input_size)

    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, input_size, swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()

    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    x_factor = cols / input_size[0]
    y_factor = rows / input_size[1]

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.5:
            class_score = row[5:].max()
            class_id = row[5:].argmax()
            if class_score > 0.3:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.3, 0.45).flatten()

    for ind in indices:
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        class_id = classes[ind]
        class_name = labels[class_id]
        text = f'{class_name}: {bb_conf}%'

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y - 30), (x + w, y), (255, 255, 255), -1)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="YOLOv5 Object Detection Result", use_column_width=True)
    st.write("Object detection completed and displayed!")
else:
    st.write("Please upload an image file.")
