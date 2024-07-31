import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
import torch


app = Flask(__name__)

def preprocess_image(image_path):
    img = Image.open(image_path).resize((640, 640))
    img_array = np.array(img) / 255.0
    return img_array

def predict_image(image_path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, trust_repo=True)  # or yolov5n - yolov5x6 or custom
        im = image_path
        # Perform object detection
        results = model(im)
        # Display the image with bounding boxes on canvas
        img_with_boxes = results.render()[0]  # Get the image with bounding boxes
        img_with_boxes = Image.fromarray(img_with_boxes)  # Convert numpy array to PIL Image
        img_with_boxes = img_with_boxes.resize((400,400))
        return img_with_boxes
    except Exception as e:
        print("Error:", e)
        return None

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file")
        if file:
            input_file_path = os.path.join("static", "input_image.jpg")  # Path to save input image
            file.save(input_file_path)  # Save the input image
            image_with_boxes = predict_image(input_file_path)  # Detect objects and get the result image
            if image_with_boxes:
                result_file_path = os.path.join("static", "result.jpg")  # Path to save result image
                image_with_boxes.save(result_file_path)  # Save the result image
                return render_template("result.html")
            else:
                return render_template("index.html", message="Error processing image")
    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
