import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression

app = Flask(__name__)

# Load YOLOv5 model
model_path = "path/to/your/model.pt"  # Specify the path to your trained YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, map_location=device)
model.eval()

def detect_objects(image):
    img = torch.from_numpy(image).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        detections = model(img)[0]
        detections = non_max_suppression(detections, conf_thres=0.3, iou_thres=0.45)

    return detections[0]

@app.route('/')
def index():
    return render_template('index.html')  # Create a simple HTML form for file upload

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detections = detect_objects(image)
        
        # Process detection results here and return the response as JSON
        # You can extract bounding box coordinates, class labels, etc. from 'detections'
        
        return jsonify({"message": "Detection completed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
