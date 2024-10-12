import io
import json
import requests
import numpy as np
import torch
from datetime import datetime
import time
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
import uvicorn

# Load a pre-trained ResNet model and set it to evaluation mode
model = models.resnet18(pretrained=True)
model.eval()

# Load class labels from the specified URL
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Initialize FastAPI application
app = FastAPI()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image for the model.
    Steps include resizing, normalization, and tensor conversion.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    return transform(image)

def get_prediction(image_tensor: torch.Tensor) -> str:
    """
    Make a prediction using the model and return the predicted class label.
    """
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    # Get the predicted class index and corresponding class label
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    return labels[predicted_class]

@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns the current timestamp in epoch format.
    """
    epoch_timestamp = int(time.time())  # Get current time in epoch format
    return {"status": "ok", "timestamp": epoch_timestamp}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an image file and return the predicted class.
    """
    # Read the image data directly into memory
    image_data = await file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    class_name = get_prediction(processed_image)

    return {"predicted_class": class_name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
